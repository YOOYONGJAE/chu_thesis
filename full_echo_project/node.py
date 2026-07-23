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
        # route_call_count = 이 노드가 라우팅 결정을 내린 총 횟수
        # 결정당 에코 이웃 수 = 전체 echo_query_count 합 / 전체 route_call_count 합
        self.echo_query_count = 0
        self.route_call_count = 0

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
    # 라우팅: 패킷을 받아 다음 홉 반환, Q 테이블 업데이트
    # -------------------------------------------------------------------------
    def route(self, packet, current_tick, all_nodes):
        self.route_call_count += 1
        if self.algorithm == 'q_routing':
            return self._route_q(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqfe':
            return self._route_aqfe(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm':
            return self._route_aqrerm(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqprice':
            return self._route_aqprice(packet, current_tick, all_nodes)
        else:
            raise ValueError(f"unknown algorithm: {self.algorithm} "
                             f"(정리판은 q_routing/aqfe/aqrerm/aqprice 만 지원, "
                             f"그 외 변형은 legacy_algorithm_files/ 참조)")

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

        if self.total_point >= c_pt:
            # === Pre-echo 모드 ===
            self.total_point -= c_pt
            self.pfe_window_full_echo_count += 1

            # 1차: 모든 이웃의 fresh t_n + queue 수집
            t_values = {n: all_nodes[n].best_estimate(dst, exclude_node=exclude_req)
                        for n in self.neighbors}
            self.echo_query_count += len(self.neighbors)  # Full Echo: 모든 이웃 조회
            for n in self.neighbors:
                self.last_known_queue[n] = len(all_nodes[n].queue)

            # 2차: fresh Score = t_n + c·queue_n 으로 y_star 선정
            def _score(n):
                if math.isfinite(t_values[n]):
                    return t_values[n] + c_q * self.last_known_queue[n]
                else:
                    return float('inf')

            y_star = min(candidates, key=_score)

            # 3차: y_star eta, 비-y_star eta2 균등 update
            self.Q[dst][y_star] += eta * (
                q + s + t_values[y_star] - self.Q[dst][y_star]
            )
            for n in self.neighbors:
                if n == y_star:
                    continue
                if math.isfinite(t_values[n]):
                    self.Q[dst][n] += eta2 * (
                        q + s + t_values[n] - self.Q[dst][n]
                    )
        else:
            # === Q-routing 모드 (포인트 부족) ===
            # stale 캐시 기반 선정
            y_star = min(
                candidates,
                key=lambda n: self.Q[dst][n] + c_q * self.last_known_queue[n]
            )
            # y_star 만 update
            t = all_nodes[y_star].best_estimate(dst, exclude_node=exclude_req)
            self.echo_query_count += 1  # 포인트 부족(Q-routing 모드): y_star 하나만 조회
            self.last_known_queue[y_star] = len(all_nodes[y_star].queue)
            self.Q[dst][y_star] += eta * (
                q + s + t - self.Q[dst][y_star]
            )

        # Route Memory 갱신
        if L == 0:
            packet.route_memory = []
        else:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star
