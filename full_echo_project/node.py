# =============================================================================
# [요약] 노드 구현 — Packet + Node (Q 테이블, 4종 라우팅 알고리즘)
# - Q-routing / AQFE / AQRERM / AQPRICE 만 포함하는 정리판
# - 구세대 변형 (PFE 계열, AQRERM 하드코딩 시리즈, RL 컨트롤러 연동 등) 의
#   전체 구현은 legacy_algorithm_files/node.py 에 보존
# =============================================================================
import math
import random
from collections import deque


# =====================================================================
# AQPRICE 상수
# - 노드 별 total_point (Full Echo 사용 예산) 의 초기값
# - gr, b_max, c 등은 self.params 에서 읽음 (main 스크립트에서 override 가능)
# =====================================================================
PFE_TOTAL_POINT_INITIAL = 0.0   # 시작 포인트. 0 이면 처음엔 Full Echo 불가, 모은 뒤 발동


class Packet:
    def __init__(self, src, dst, created_at):
        self.src = src
        self.dst = dst
        self.created_at = created_at
        self.queue_entry_tick = created_at  # 현재 큐 진입 tick (큐 이동마다 갱신)
        self.route_memory = []              # 방문 노드 리스트 (AQRERM / AQPRICE 의 Route Memory)


class Node:
    def __init__(self, node_id, neighbors, algorithm, params, num_nodes):
        self.id = node_id
        self.neighbors = neighbors          # 인접 노드 ID 리스트
        self.algorithm = algorithm
        self.params = params

        self.queue = deque()               # 처리 대기 패킷
        self.incoming = []                 # 이번 tick 도착 패킷 (다음 tick에 queue로 이동)

        # Q 테이블: Q[dst][nbr] = 예상 전달 시간, 초기값 1.0
        self.Q = {
            d: {n: 1.0 for n in neighbors}
            for d in range(num_nodes) if d != node_id
        }

        # 신뢰도 표 (CQ-Routing, Kumar & Miikkulainen 1999) — aqprice_conf 가 켜졌을 때만 쓰인다.
        # C[dst][nbr] ∈ [0, 1] : Q[dst][nbr] 를 얼마나 믿을 수 있는가.
        # 이웃이 곧 목적지인 칸은 1 로 고정, 나머지는 0 에서 시작.
        # 매 tick 감쇠(C ← λ·C)는 갱신 tick 을 저장해 두고 조회할 때 λ^(경과 tick) 을 곱해 계산한다.
        self.C = {
            d: {n: (1.0 if n == d else 0.0) for n in neighbors}
            for d in range(num_nodes) if d != node_id
        }
        self.C_tick = {
            d: {n: 0 for n in neighbors}
            for d in range(num_nodes) if d != node_id
        }

        self.T_est = 0.0
        self.T_max = 1.0

        # AQPRICE: 매 tick 포인트 적립 활성 여부
        # 활성 시 simulator tick 루프가 모든 노드에 tick_accumulate_point() 호출 →
        # 큐가 비어 라우팅 안 하는 tick 에도 gr 만큼 적립이 진행됨.
        self.tick_accum_enabled = (algorithm == 'aqprice')

        # AQPRICE: echo 응답 시 받은 이웃 큐 길이 캐시 (실시간 직접 읽기 대체)
        self.last_known_queue = {n: 0 for n in neighbors}

        # AQPRICE: 누적 포인트 (Full Echo 사용 예산)
        self.total_point = PFE_TOTAL_POINT_INITIAL

        # AQPRICE 진단 카운터 — simulator 가 stat_interval 시점에 읽고 0 으로 리셋
        # full_echo_ratio = pfe_window_full_echo_count / pfe_window_route_count
        self.pfe_window_full_echo_count = 0
        self.pfe_window_route_count     = 0

        # 에코 비용 진단 — 실행 전체 누적 (리셋 없음). simulator 가 종료 시 합산해 노출.
        # echo_query_count = 이웃에게 전달시간 추정을 조회한 총 횟수 (best_estimate 호출 수)
        # route_call_count = 이 노드가 라우팅을 수행한 총 횟수 (지표의 분모)
        # 라우팅 결정당 이웃 조회 수 = 전체 echo_query_count 합 / 전체 route_call_count 합
        self.echo_query_count = 0
        self.route_call_count = 0

        # 경로 변경률(next-hop 변경률) 진단 — 안정 구간에서 선택이 자주 뒤집히는지 확인.
        #   같은 목적지 d 에 대해 직전에 고른 next_hop 을 기억했다가, 이번 결정에서
        #   달라졌으면 변경 1회로 집계한다. 변경률 = 변경 횟수 / 라우팅 결정 횟수.
        #   모든 알고리즘에서 집계하므로 알고리즘 간 경로 진동 정도를 비교할 수 있다.
        self.last_choice = {}              # dst -> 최근에 고른 next_hop
        self.switch_count = 0              # 실행 전체 누적 변경 횟수 (리셋 없음)
        self.window_switch_count   = 0     # 윈도우 단위 변경 횟수 (simulator 가 읽고 리셋)
        self.window_decision_count = 0     # 윈도우 단위 라우팅 결정 횟수 (모든 알고리즘)

    # -------------------------------------------------------------------------
    # T_est 업데이트 (AQFE / AQRERM / AQPRICE)
    # T_est = 모든 목적지에 대해 min_y Q[d][y] 의 평균 (AQRERM 정의)
    # -------------------------------------------------------------------------
    def update_T_est(self):
        if not self.Q or not self.neighbors:
            self.T_est = 0.0
            return
        self.T_est = sum(
            min(self.Q[d][n] for n in self.neighbors)
            for d in self.Q
        ) / len(self.Q)
        if self.T_est > self.T_max:
            self.T_max = self.T_est

    # -------------------------------------------------------------------------
    # AQPRICE per-tick 포인트 적립 — 매 tick 모든 노드에 호출 (simulator tick 루프).
    # tick_accum_enabled=False 인 노드는 즉시 return.
    # 라우팅 호출 여부와 무관하게 큐가 비어 있어도 적립 진행 →
    # 한가한 노드도 풀에코 예산을 천천히 축적 가능.
    # -------------------------------------------------------------------------
    def tick_accumulate_point(self, current_tick):
        if not self.tick_accum_enabled:
            return
        gr    = self.params.get('pfe_gr', 0.133)
        b_max = self.params.get('pfe_b_max', 0.5)
        self.total_point = min(b_max, self.total_point + gr)

    # -------------------------------------------------------------------------
    # 이 노드에서 dst까지의 최선 추정치 반환
    # exclude_node: 이 노드를 이웃 후보에서 제외 (Route Memory용)
    # -------------------------------------------------------------------------
    def best_estimate(self, dst, exclude_node=None):
        if dst == self.id:
            return 0.0
        if dst not in self.Q:
            return float('inf')
        candidates = {
            n: self.Q[dst][n]
            for n in self.neighbors
            if n != exclude_node
        }
        if not candidates:
            # exclude 후 후보가 비면 fallback: exclude_node 도 포함해서 다시 시도
            # (dead-end 노드 — 실제 라우팅도 같은 fallback 으로 작동하므로 일관됨)
            candidates = {n: self.Q[dst][n] for n in self.neighbors}
        if not candidates:
            return float('inf')
        return min(candidates.values())

    # -------------------------------------------------------------------------
    # 큐 페널티 — 이웃 y 를 거칠 때 그 노드의 큐에서 기다릴 시간의 대리값 (c · 큐길이)
    #
    # 단 y 가 곧 목적지이면 0 을 준다. simulator 는 next_hop == dst 인 순간 그 자리에서
    # 전달을 끝내므로, 패킷이 목적지의 큐에 들어가지도 않고 목적지의 처리 기회를 쓰지도
    # 않는다. 따라서 목적지의 큐 길이는 이 패킷의 전달시간에 아무 영향이 없다.
    # 벌점을 주면 목적지를 코앞에 두고 다른 이웃으로 우회하는 선택이 생기는데,
    # 직행이 1 tick 인 반면 우회는 최소 2 tick 이므로 어떤 경우에도 손해다.
    # -------------------------------------------------------------------------
    @staticmethod
    def queue_penalty(neighbor, dst, c_q, queue_len):
        if neighbor == dst:
            return 0.0
        return c_q * queue_len

    # -------------------------------------------------------------------------
    # 신뢰도 (CQ-Routing, Kumar & Miikkulainen 1999 식 2·3) — AQPRICE 실험용, aqprice_conf 토글
    #
    # conf_value   : 지금 tick 기준 신뢰도. 마지막 갱신 뒤 지난 tick 수만큼 λ 를 곱한다 (식 3 위쪽).
    # best_estimate_conf : best_estimate 와 같은 최솟값 Q + 그 칸의 신뢰도 (에코에 함께 실어 보냄).
    # conf_lr      : 학습률 = max(C_est, 1 - C_old) (식 2). 내 칸을 못 믿거나 받은 값을 믿을수록 크다.
    # conf_set     : Q 를 갱신한 칸의 신뢰도를 C_old + lr·(C_est - C_old) 로 옮긴다 (식 3 아래쪽).
    # -------------------------------------------------------------------------
    def conf_value(self, dst, nbr, tick):
        if nbr == dst:
            return 1.0
        lam = self.params.get('conf_lambda', 0.95)
        return self.C[dst][nbr] * (lam ** (tick - self.C_tick[dst][nbr]))

    def best_estimate_conf(self, dst, tick, exclude_node=None):
        if dst == self.id:
            return 0.0, 1.0
        if dst not in self.Q:
            return float('inf'), 0.0
        cands = [n for n in self.neighbors if n != exclude_node]
        if not cands:
            cands = list(self.neighbors)   # best_estimate 와 같은 fallback
        if not cands:
            return float('inf'), 0.0
        n_best = min(cands, key=lambda n: self.Q[dst][n])
        return self.Q[dst][n_best], self.conf_value(dst, n_best, tick)

    def conf_lr(self, dst, nbr, c_est, tick):
        return max(c_est, 1.0 - self.conf_value(dst, nbr, tick))

    def conf_set(self, dst, nbr, c_est, lr, tick):
        if nbr == dst:
            return                      # 이웃이 곧 목적지인 칸은 1 로 고정
        c_old = self.conf_value(dst, nbr, tick)
        self.C[dst][nbr] = c_old + lr * (c_est - c_old)
        self.C_tick[dst][nbr] = tick

    # -------------------------------------------------------------------------
    # Dual Reinforcement Q-Routing 역방향 갱신 (AQPRICE 실험용, aqprice_dual 토글)
    #
    # 이 노드(x)가 출발지 src 인 패킷을 다음 홉 nbr (= 선택한 y_star) 로 보낼 때,
    # 'x 에서 src 까지의 추정' 을 함께 보낸다고 본다 (원 논문처럼 패킷을 받는 노드만 갱신).
    # nbr 은 그 값으로 자기 Q[src][x] ('x 에게 넘기면 src 까지 몇 틱') 를 갱신한다.
    #   목표값 = q_n (nbr 의 큐 대기) + 1 (nbr → x 한 홉) + x 의 src 추정
    # q_n : 원 논문 식 (4) 의 q_y (갱신하는 노드 자신의 큐 대기 시간) 에 해당한다.
    #   정방향 갱신이 q (자기 큐 대기) 를 넣으므로, 역방향에서 빼면 x 칸만 q 만큼 싸 보인다.
    #   노드는 tick 당 패킷 1 개를 처리하므로 nbr 의 현재 큐 길이를 예상 대기 틱으로 쓴다
    #   (이 패킷은 다음 tick 에 nbr 큐의 맨 뒤에 붙는다).
    # x 의 추정은 원 논문 식 (3) 처럼 x 의 모든 이웃 중 최솟값이다 (nbr 도 포함).
    # x 의 최선 경로가 nbr 을 거치면 nbr 이 'x 를 거쳐 자기에게 돌아오는 길' 의 값을 배울 수 있다.
    # (실험 1~4 는 nbr 을 빼고 계산했다. experiments.md 참고)
    # 학습률 mode:
    #   'eta'  : 정방향 y_star 갱신과 같은 eta
    #   'eta2' : nbr 자신의 R_x · k (정방향 비-y_star 갱신과 같은 형태)
    # 값 하나를 얹어 보내는 것이라 이웃 조회 수(echo_query_count)는 늘리지 않는다.
    # -------------------------------------------------------------------------
    def dual_backward_update(self, nbr, src, all_nodes, mode):
        if nbr == src:
            return                      # nbr 의 Q 테이블에는 자기 자신 행이 없다
        nbr_node = all_nodes[nbr]
        if src not in nbr_node.Q:
            return
        t_x = self.best_estimate(src)   # 모든 이웃 중 최솟값. x 가 src 자신이면 0
        if not math.isfinite(t_x):
            return
        if mode == 'eta':
            lr = self.params['eta']
        elif mode == 'eta2':
            rx_n = nbr_node.T_est / nbr_node.T_max if nbr_node.T_max > 0 else 0.0
            lr = rx_n * self.params['k']
        else:
            raise ValueError(f"aqprice_dual 은 None / 'eta' / 'eta2' 중 하나여야 한다: {mode!r}")
        q_n = len(nbr_node.queue)
        old = nbr_node.Q[src][self.id]
        nbr_node.Q[src][self.id] = old + lr * (q_n + 1 + t_x - old)

    # -------------------------------------------------------------------------
    # 라우팅: 패킷을 받아 다음 홉 반환, Q 테이블 업데이트
    # -------------------------------------------------------------------------
    def route(self, packet, current_tick, all_nodes):
        self.route_call_count += 1
        self.window_decision_count += 1
        if self.algorithm == 'q_routing':
            next_hop = self._route_q(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqfe':
            next_hop = self._route_aqfe(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm':
            next_hop = self._route_aqrerm(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqprice':
            next_hop = self._route_aqprice(packet, current_tick, all_nodes)
        else:
            raise ValueError(f"unknown algorithm: {self.algorithm} "
                             f"(정리판은 q_routing/aqfe/aqrerm/aqprice 만 지원, "
                             f"그 외 변형은 legacy_algorithm_files/ 참조)")

        # 경로 변경률 집계: 같은 목적지에 대해 직전 선택과 달라졌으면 변경으로 카운트.
        # (해당 목적지의 첫 라우팅 결정은 비교 대상이 없어 변경으로 세지 않고 기록만 한다)
        dst = packet.dst
        prev = self.last_choice.get(dst)
        if prev is not None and prev != next_hop:
            self.switch_count += 1
            self.window_switch_count += 1
        self.last_choice[dst] = next_hop
        return next_hop

    # -------------------------------------------------------------------------
    # Q-routing (Boyan & Littman 1994)
    # y* 만 단일 업데이트, echo 없음
    # -------------------------------------------------------------------------
    def _route_q(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])

        q = current_tick - packet.queue_entry_tick
        s = 1
        t = all_nodes[y_star].best_estimate(dst)
        self.echo_query_count += 1  # y_star 하나만 조회 (echo 없음)

        self.Q[dst][y_star] += eta * (q + s + t - self.Q[dst][y_star])

        return y_star

    # -------------------------------------------------------------------------
    # AQFE (Shilova et al. 2016) — Adaptive Q-routing with Full Echo
    # 매 라우팅마다 모든 이웃 echo, 비-y_star 는 eta2 = (T_est/T_max)·k 로 업데이트
    # -------------------------------------------------------------------------
    def _route_aqfe(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']

        self.update_T_est()

        # 모든 이웃의 t 값 수집 (Full Echo)
        t_values = {n: all_nodes[n].best_estimate(dst) for n in self.neighbors}
        self.echo_query_count += len(self.neighbors)  # 모든 이웃 조회 (Full Echo)

        # 선택은 갱신 전 Q-table 로 한다. 이번 에코로 받은 t_values 는 Q 갱신에만 쓰이고
        # 현재 패킷의 다음 홉에는 반영되지 않는다 (다음 패킷부터 영향).
        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = (self.T_est / self.T_max) * k if self.T_max > 0 else 0.0

        # 선택된 이웃: eta로 업데이트
        self.Q[dst][y_star] += eta * (q + s + t_values[y_star] - self.Q[dst][y_star])

        # 나머지 이웃: eta2로 업데이트
        for n in self.neighbors:
            if n != y_star:
                self.Q[dst][n] += eta2 * (q + s + t_values[n] - self.Q[dst][n])

        return y_star

    # -------------------------------------------------------------------------
    # AQRERM (Kavalerov et al. 2017) — Random Echo + Route Memory
    # y* 는 항상 echo, 나머지는 확률 p = T_est/T_max 로 echo.
    # Route Memory: 최근 L 개 방문 노드를 후보에서 제외.
    # -------------------------------------------------------------------------
    def _route_aqrerm(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # Route Memory: 방문한 노드 제외
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # AQRERM 큐 페널티 토글 — c 효과 비교 실험용.
        # aqrerm_c > 0 이면 선택식에 이웃 큐 길이를 페널티로 더함 (AQPRICE 의 c 와 동일 개념).
        # 기본 0 → 순수 Q 선택 (기존 AQRERM 동작 그대로).
        aqrerm_c = self.params.get('aqrerm_c', 0.0)
        if aqrerm_c > 0:
            y_star = min(candidates,
                         key=lambda n: self.Q[dst][n] + self.queue_penalty(
                             n, dst, aqrerm_c, len(all_nodes[n].queue)))
        else:
            y_star = min(candidates, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * k

        # y*는 항상 echo, 나머지는 확률 p로 echo
        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)
        self.echo_query_count += len(echo_set)  # 실제 echo 한 이웃 수 (부분 echo)

        for n in echo_set:
            # Route Memory: y=n에서 t 추정 시 현재 노드 x 제외
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # Route Memory 갱신 (L=0 이면 항상 빈 리스트)
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQPRICE — Adaptive Q-Routing with Point-Regulated Inline Congestion Echo
    #
    # 포인트 예산 게이트 + pre-echo (echo → 선정 → 학습) + 큐 페널티 (c·queue).
    #   - 적립: simulator tick 루프가 tick_accumulate_point() 로 매 tick gr 적립
    #           (큐가 비어 라우팅 안 하는 tick 에도 적립 진행)
    #   - 가격: sale_pt = max(0.1, 1 - R_x) — 혼잡(불안정)할수록 Full Echo 가 싸짐
    #   - total_point >= 가격 이면:
    #       1) Full Echo 발동 → 모든 이웃의 fresh t_n + queue 수집
    #       2) Score = t_n + c·queue_n 으로 y_star 선정 (Route Memory 필터 적용)
    #       3) y_star: eta 로 update, 그 외 이웃: eta2 = R_x·k 로 균등 update
    #   - 포인트 부족 시 (Q-routing 모드):
    #       1) echo 미수행
    #       2) stale 캐시 (Q + c·last_known_queue) 로 y_star 선정
    #       3) y_star 만 update
    # -------------------------------------------------------------------------
    def _route_aqprice(self, packet, current_tick, all_nodes):
        self.pfe_window_route_count += 1

        dst   = packet.dst
        eta   = self.params['eta']
        k     = self.params['k']
        # AQPRICE 정식 구성은 라우트 메모리 미포함 → 전용 키 aqprice_L 로 읽고 기본 0(=미사용).
        # 일반 스크립트는 aqprice_L 을 안 넘기므로 자동 L=0. ablation 스크립트만 명시적으로 aqprice_L 세팅.
        # (AQRERM 은 그대로 self.params['L'] 을 읽어 라우트 메모리 유지)
        L     = self.params.get('aqprice_L', 0)
        c_q   = self.params['c']
        # 요청 노드 제외 토글 — 이웃이 최소 Q 를 낼 때 질문자(self)를 뺄지 여부.
        # 기본 True(제외 = 기존 동작, AQRERM 라우트 메모리에서 온 조건).
        # ablation 스크립트가 aqprice_exclude_requester=False 로 두면 Q-routing/AQFE 처럼 질문자도 포함.
        exclude_req = self.id if self.params.get('aqprice_exclude_requester', True) else None
        # 포인트 게이트 토글 — 예산·가격 판정을 건너뛸지 여부. 제거 실험 전용.
        #   True(기본, AQPRICE 정식): total_point >= 가격 일 때만 Full Echo.
        #   False: 판정 없이 항상 Full Echo 를 수행 (Inline·c·L 은 그대로 유지).
        # 파라미터로 gr / b_max 를 크게 주는 방식과 달리, 한 tick 에 여러 패킷을 처리해도
        # 반드시 매 결정 Full Echo 가 되므로 '예산만 제거한 조건' 이 정확히 성립한다.
        point_gate = self.params.get('aqprice_point_gate', True)
        # Dual Reinforcement 역방향 갱신 토글 — None(기본, AQPRICE 정식) / 'eta' / 'eta2' (학습률).
        # 켜져 있으면 포인트 부족(Q-routing 모드)에서는 항상 다음 홉 y_star 한 곳을 갱신한다.
        dual_mode = self.params.get('aqprice_dual', None)
        # Full Echo 모드에서의 역방향 갱신 범위 (dual_mode 가 켜져 있을 때만 의미 있음)
        #   None(기본) : Full Echo 모드에서는 갱신하지 않음
        #   'all'      : 에코를 받은 이웃 전부
        #   'ystar'    : 패킷을 받는 y_star 하나
        dual_fe = self.params.get('aqprice_dual_fe', None)
        # 신뢰도 학습률 토글 (CQ-Routing)
        #   None(기본) : AQPRICE 정식 (y_star 는 eta, 선택하지 않은 이웃은 eta2)
        #   'eta'      : y_star 갱신만 신뢰도 학습률, 선택하지 않은 이웃은 eta2 유지
        #   'eta2'     : 선택하지 않은 이웃 갱신만 신뢰도 학습률, y_star 는 eta 유지 (두 모드 모두)
        #   'both'     : y_star 와 선택하지 않은 이웃 모두 신뢰도 학습률
        # 켜져 있으면 Q-value 를 고칠 때마다 그때 쓴 학습률로 그 Q-value 의 신뢰도도 고친다.
        conf_mode = self.params.get('aqprice_conf', None)
        if conf_mode not in (None, 'eta', 'eta2', 'both'):
            raise ValueError(f"aqprice_conf 는 None / 'eta' / 'eta2' / 'both' 중 하나여야 한다: {conf_mode!r}")
        conf_star  = conf_mode in ('eta', 'both')    # y_star 학습률을 신뢰도로
        conf_other = conf_mode in ('eta2', 'both')   # 선택하지 않은 이웃 학습률을 신뢰도로

        self.update_T_est()

        # R_x 는 현재 노드의 불안정도 지표. 0 이면 안정적, 1 에 가까울수록 불안정.
        R_x = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # 동적 가격 (Full Echo 1 회 비용)
        sale_pt = max(0.1, 1.0 - R_x)
        c_pt = self.params.get('pfe_c', sale_pt)

        q = current_tick - packet.queue_entry_tick
        s = 1
        eta2 = R_x * k

        # Route Memory 필터
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        if (not point_gate) or (self.total_point >= c_pt):
            # === Pre-echo 모드 ===
            # 게이트가 꺼진 조건에서는 차감 없이 항상 이 분기를 탄다 (예산 개념 자체가 없는 상태).
            if point_gate:
                self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            # 1차: 모든 이웃의 fresh t_n 수집 (Full Echo)
            # 신뢰도가 켜져 있으면 각 이웃이 최솟값 Q 의 신뢰도(c_values)도 함께 보낸다.
            if conf_mode is None:
                t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=exclude_req)
                            for n in self.neighbors}
            else:
                t_values, c_values = {}, {}
                for n in self.neighbors:
                    t_values[n], c_values[n] = all_nodes[n].best_estimate_conf(
                        dst, current_tick, exclude_node=exclude_req)
            self.echo_query_count += len(self.neighbors)  # Full Echo: 모든 이웃 조회

            # Inline 토글 — 이번 에코로 얻은 최신 정보를 '현재 패킷 선택'에 쓸지 여부.
            #   Inline(기본, True): 최신 t_n + 최신 큐로 선택 (AQPRICE 정식).
            #   post-echo(False): 에코는 하되 선택에는 안 쓰고, 기존 Q + 이전 큐 캐시로 먼저 선택
            #     (AQFE/AQRERM 처럼 '선택 먼저 → 에코는 갱신용'). 에코 정보는 다음 패킷부터 반영.
            #   Inline 이 실제로 성능을 바꾸는지 순서만 바꿔 분리 검증하는 토글.
            aqprice_inline = self.params.get('aqprice_inline', True)

            if aqprice_inline:
                # 최신 큐로 캐시 갱신 후, 그 최신값으로 선택
                for n in self.neighbors:
                    self.last_known_queue[n] = len(all_nodes[n].queue)

                def _score(n):
                    if not math.isfinite(t_values[n]):
                        return float('inf')
                    return t_values[n] + self.queue_penalty(
                        n, dst, c_q, self.last_known_queue[n])

                y_star = min(candidates, key=_score)
            else:
                # 기존 Q + 이전 큐 캐시로 먼저 선택 (이번 에코 정보 사용 안 함)
                y_star = min(
                    candidates,
                    key=lambda n: self.Q[dst][n] + self.queue_penalty(
                        n, dst, c_q, self.last_known_queue[n])
                )
                # 선택 후 큐 캐시 갱신 (Q 갱신·다음 패킷용)
                for n in self.neighbors:
                    self.last_known_queue[n] = len(all_nodes[n].queue)

            # 3차: y_star eta, 비-y_star eta2 균등 update
            # (신뢰도가 켜져 있으면 aqprice_conf 에 따라 학습률을 max(C_est, 1-C_old) 로 바꾼다)
            if conf_star:
                lr_star = self.conf_lr(dst, y_star, c_values[y_star], current_tick)
            else:
                lr_star = eta
            self.Q[dst][y_star] += lr_star * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
            if conf_mode is not None:
                self.conf_set(dst, y_star, c_values[y_star], lr_star, current_tick)
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    if conf_other:
                        lr_n = self.conf_lr(dst, n, c_values[n], current_tick)
                    else:
                        lr_n = eta2
                    self.Q[dst][n] += lr_n * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
                    if conf_mode is not None:
                        self.conf_set(dst, n, c_values[n], lr_n, current_tick)

            # Dual 역방향 갱신 (Full Echo 모드) — aqprice_dual_fe 로 범위를 고른다.
            # 갱신하는 칸이 이웃의 Q[src] 행이라 이번 결정의 선택·정방향 갱신에는 영향 없음.
            if dual_mode is not None:
                if dual_fe == 'all':
                    for n in self.neighbors:
                        self.dual_backward_update(n, packet.src, all_nodes, dual_mode)
                elif dual_fe == 'ystar':
                    self.dual_backward_update(y_star, packet.src, all_nodes, dual_mode)
                elif dual_fe is not None:
                    raise ValueError(f"aqprice_dual_fe 는 None / 'all' / 'ystar' 중 하나여야 한다: {dual_fe!r}")
        else:
            # === Q-routing 모드 (포인트 부족) ===
            # stale 캐시 기반 선정
            y_star = min(
                candidates,
                key=lambda n: self.Q[dst][n] + self.queue_penalty(
                    n, dst, c_q, self.last_known_queue[n])
            )
            # y_star 만 update ('eta'/'both' 는 신뢰도 학습률, 'eta2' 는 eta 를 쓰되 신뢰도는 추적)
            if conf_mode is None:
                t = all_nodes[y_star].best_estimate(dst, exclude_node=exclude_req)
                lr_star = eta
            else:
                t, c_star = all_nodes[y_star].best_estimate_conf(
                    dst, current_tick, exclude_node=exclude_req)
                lr_star = (self.conf_lr(dst, y_star, c_star, current_tick)
                           if conf_star else eta)
            self.echo_query_count += 1  # 포인트 부족(Q-routing 모드): y_star 하나만 조회
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += lr_star * (
                q + s + t - self.Q[dst][y_star]
            )
            if conf_mode is not None:
                self.conf_set(dst, y_star, c_star, lr_star, current_tick)
            # Dual 역방향 갱신 — 포인트 부족(Q-routing 모드)일 때만, 이 패킷을 받을 y_star 한 곳.
            # Full Echo 모드는 이미 이웃 전부를 갱신하므로 역방향 갱신을 더하지 않는다.
            # (원 논문 DRQ 는 결정당 한 칸만 갱신하는 Q-routing 에 갱신을 하나 더한 것)
            if dual_mode is not None:
                self.dual_backward_update(y_star, packet.src, all_nodes, dual_mode)

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star
