import math
import random
from collections import deque
from echo_controller import EchoController


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _signed_ratio01(x, scale):
    """0 → 0.5, 양수 → >0.5, 음수 → <0.5 (tanh 기반)"""
    if scale == 0:
        return 0.5
    return 0.5 + 0.5 * math.tanh(x / scale)


class Packet:
    def __init__(self, src, dst, created_at):
        self.src = src
        self.dst = dst
        self.created_at = created_at
        self.queue_entry_tick = created_at  # 현재 큐 진입 tick (큐 이동마다 갱신) : 의미는 "이 tick까지 이 노드 큐에 있었음"
        self.route_memory = []              # 방문 노드 리스트 (AQRERM용)


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

        # ΔQ_min: 목적지별 직전 Q_min 저장
        self.prev_Q_min = {d: 1.0 for d in range(num_nodes) if d != node_id}

        # TD_error_ema: TD 에러의 지수이동평균
        self.td_error_ema = 0.0
        self.td_ema_alpha = 0.1  # EMA 감쇠 계수

        # route_switching_recent: 최근 500 tick 내 y* 변경 기록 [(tick, y_star), ...]
        self.y_star_history = deque()

        # echo_age_avg: 목적지 d, 이웃 n별 마지막 echo tick
        self.last_echo_tick = {
            d: {n: 0 for n in neighbors}
            for d in range(num_nodes) if d != node_id
        }

        # AQLRERM: echo 응답 시 받은 이웃 큐 길이 캐시 (실시간 직접 읽기 대체)
        self.last_known_queue = {n: 0 for n in neighbors}

    # -------------------------------------------------------------------------
    # T_est 업데이트 (AQFE / AQRERM)
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
            # 이 한 줄이 Q 테이블의 inf 폭주를 차단해서 평균/분산 통계까지 보호함
            candidates = {n: self.Q[dst][n] for n in self.neighbors}
        if not candidates:
            return float('inf')
        return min(candidates.values())

    # -------------------------------------------------------------------------
    # 라우팅: 패킷을 받아 다음 홉 반환, Q 테이블 업데이트
    # -------------------------------------------------------------------------
    def route(self, packet, current_tick, all_nodes):
        if self.algorithm == 'q_routing':
            return self._route_q(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqfe':
            return self._route_aqfe(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm':
            return self._route_aqrerm(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqrerm_no_mem':
            return self._route_aqrerm_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqlrerm':
            return self._route_aqlrerm(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqlrerm_7000_no_mem':
            return self._route_aqlrerm_7000_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm == 'aqlrerm_all_no_mem':
            return self._route_aqlrerm_all_no_mem(packet, current_tick, all_nodes)
        elif self.algorithm in ('aqlrerm_l_train', 'aqlrerm_l_close'):
            return self._route_aqlrerm_l_train(packet, current_tick, all_nodes)
        elif self.algorithm in ('learned_aqrerm', 'bandit_aqrerm'):
            return self._route_learned_aqrerm(packet, current_tick, all_nodes)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _route_q(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])

        q = current_tick - packet.queue_entry_tick
        s = 1
        t = all_nodes[y_star].best_estimate(dst)

        self.Q[dst][y_star] += eta * (q + s + t - self.Q[dst][y_star])

        return y_star

    def _route_aqfe(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']

        self.update_T_est()

        # 모든 이웃의 t 값 수집 (Full Echo)
        t_values = {n: all_nodes[n].best_estimate(dst) for n in self.neighbors}

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = (self.T_est / self.T_max) * eta * k if self.T_max > 0 else 0.0

        # 선택된 이웃: eta로 업데이트
        self.Q[dst][y_star] += eta * (q + s + t_values[y_star] - self.Q[dst][y_star])

        # 나머지 이웃: eta2로 업데이트
        for n in self.neighbors:
            if n != y_star:
                self.Q[dst][n] += eta2 * (q + s + t_values[n] - self.Q[dst][n])

        return y_star

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

        eta2 = p * eta * k

        # y*는 항상 echo, 나머지는 확률 p로 echo
        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

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
    # AQRERM_no_mem: AQRERM에서 Route Memory만 끈 변형 (디버깅용)
    # - 방문 노드 후보 제외 X
    # - 이웃 t 추정 시 self.id 제외 X
    # - packet.route_memory도 갱신하지 않음
    # -------------------------------------------------------------------------
    def _route_aqrerm_no_mem(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        y_star = min(self.neighbors, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * eta * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        return y_star

    # -------------------------------------------------------------------------
    # AQLRERM: AQRERM + 큐 길이 페널티를 라우팅 결정에 반영
    # score(n) = Q[dst][n] + c * len(neighbor.queue)
    # Q 갱신식과 echo 메커니즘은 AQRERM과 동일
    # -------------------------------------------------------------------------
    def _route_aqlrerm(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']
        c = self.params['c']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        # 캐시된 이웃 큐 길이로 score 계산 (echo 응답 때 받은 stale 값 사용)
        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * eta * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            # echo 응답에 piggyback된 큐 길이 캐시 갱신
            self.last_known_queue[n] = len(all_nodes[n].queue)
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
    # AQLRERM_no_mem: 디버깅용
    # - current_tick < memory_cut_tick: 일반 AQLRERM 그대로 (L = params['L'])
    # - current_tick >= memory_cut_tick: effective L = 0 으로 route memory 무효화
    #   * visited 필터링 비활성화 (모든 이웃이 후보)
    #   * packet.route_memory 항상 빈 리스트로 유지
    #   (best_estimate의 exclude_node=self.id 효과는 그대로 유지)
    # -------------------------------------------------------------------------
    def _route_aqlrerm_7000_no_mem(self, packet, current_tick, all_nodes):
        # memory_cut_tick (보통 7000) 이전엔 L=params['L'], 이후엔 L=0 으로 전환
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        c = self.params['c']
        memory_cut_tick = self.params.get('memory_cut_tick', 0)
        L = 0 if current_tick >= memory_cut_tick else self.params['L']

        if L == 0:
            packet.route_memory = []

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * eta * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        if L > 0:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory

        return y_star

    # -------------------------------------------------------------------------
    # AQLRERM_ALL_NO_MEM: 시뮬레이션 시작부터 끝까지 L=0 (route memory 영구 비활성)
    # - memory_cut_tick 분기 없음 — 항상 L=0
    # - packet.route_memory 항상 빈 리스트, visited 필터링 의미 없음
    # - 그 외 echo, Q 업데이트, last_known_queue 캐싱은 AQLRERM 본체 그대로
    # -------------------------------------------------------------------------
    def _route_aqlrerm_all_no_mem(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        c = self.params['c']

        # L = 0 영구 강제 — route_memory 매번 빈 리스트로 유지
        packet.route_memory = []

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # visited 항상 비어 있으니 모든 이웃이 후보
        candidates = list(self.neighbors)
        if not candidates:
            return packet.dst   # 이웃 없으면 위로 (보호)

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * eta * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # route_memory 는 갱신 없이 빈 리스트 유지
        return y_star

    # -------------------------------------------------------------------------
    # AQLRERM_L_TRAIN: AQLRERM과 동일하되 route memory size L 을 글로벌 학습으로 결정
    # - y_star 선택: AQLRERM의 큐 페널티 (Q + c * last_known_queue)
    # - echo 확률 p: AQLRERM의 표준 p = T_est / T_max
    # - L: simulator 가 매 stat_interval 마다 글로벌 state 로 sampling 해서
    #      controller.cached_L 에 박아둔 값을 그대로 사용 (window 동안 일정)
    # - td_error_ema 만 글로벌 state 의 avg_TD_error 계산용으로 유지 갱신
    # - 그 외 보조 변수 (prev_Q_min, y_star_history, last_echo_tick) 는 갱신 생략
    # -------------------------------------------------------------------------
    def _route_aqlrerm_l_train(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        c = self.params['c']

        self.update_T_est()
        p = self.T_est / self.T_max if self.T_max > 0 else 0.0

        # L 은 컨트롤러의 window-cached 값을 직접 읽기 — actor 호출 없음
        controller = self.params['controller']
        L = controller.cached_L
        # L_history 누적 (per-routing 단위 진단 로그용)
        controller.L_history.append(L)

        # --- 이하 AQLRERM 본체와 동일 ---
        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(
            candidates,
            key=lambda n: self.Q[dst][n] + c * self.last_known_queue[n]
        )
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * eta * k

        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            self.last_known_queue[n] = len(all_nodes[n].queue)
            if n == y_star:
                self.Q[dst][n] += eta * (q + s + t_n - self.Q[dst][n])
            else:
                self.Q[dst][n] += eta2 * (q + s + t_n - self.Q[dst][n])

        # td_error_ema 갱신 — 글로벌 state 의 avg_TD_error 입력으로 쓰임
        td_error_ystar = q + s + all_nodes[y_star].best_estimate(dst, exclude_node=self.id) - self.Q[dst][y_star]
        if math.isfinite(td_error_ystar):
            self.td_error_ema = (1 - self.td_ema_alpha) * self.td_error_ema + self.td_ema_alpha * abs(td_error_ystar)

        # route_memory 갱신: 학습된 L 사용
        if L > 0:
            new_memory = packet.route_memory + [self.id]
            if len(new_memory) > L:
                new_memory = new_memory[-L:]
            packet.route_memory = new_memory
        else:
            packet.route_memory = []

        return y_star


    # 역할
    # 다음 홉을 결정한다.
    # echo_set을 확률적으로 선정하여 Q 테이블 업데이트
    def _route_learned_aqrerm(self, packet, current_tick, all_nodes):
        dst = packet.dst
        eta = self.params['eta']
        k = self.params['k']
        L = self.params['L']

        self.update_T_est()

        # --- 상태값 계산 ---
        q_values = list(self.Q[dst].values())

        # Q_min, Q_avg, Q_spread, Q_variance
        Q_min      = min(q_values)
        Q_avg      = sum(q_values) / len(q_values)
        Q_spread   = max(q_values) - Q_min
        Q_variance = sum((v - Q_avg) ** 2 for v in q_values) / len(q_values)

        # ΔQ_min: 현재 Q_min - 직전 Q_min
        delta_Q_min = Q_min - self.prev_Q_min[dst]
        self.prev_Q_min[dst] = Q_min

        # TD_error_ema: 직전 업데이트의 EMA (아직 업데이트 전이므로 현재값 사용)
        TD_error_ema = self.td_error_ema

        # queue_len: 현재 큐 길이
        queue_len = len(self.queue)

        # route_switching_recent: 최근 500 tick 내 y* 변경 횟수
        self.y_star_history = deque(
            [(t, y) for t, y in self.y_star_history if current_tick - t <= 500]
        )
        route_switching_recent = sum(
            1 for i in range(1, len(self.y_star_history))
            if self.y_star_history[i][1] != self.y_star_history[i-1][1]
        )

        # echo_age_avg: 목적지 d 기준 이웃별 마지막 echo 이후 경과 tick 평균
        echo_age_avg = sum(
            current_tick - self.last_echo_tick[dst][n]
            for n in self.neighbors
        ) / len(self.neighbors)

        # T_ratio: AQRERM의 p와 동일한 값 (참고용 상태값)
        T_ratio = self.T_est / self.T_max if self.T_max > 0 else 0.0

        T_max = self.T_max if self.T_max > 0 else 1.0
        state = [
            _clamp01(Q_min              / T_max),
            _clamp01(Q_avg              / T_max),
            _clamp01(Q_spread           / T_max),
            _clamp01(Q_variance ** 0.5  / T_max),
            _signed_ratio01(delta_Q_min,  T_max),
            _clamp01(TD_error_ema       / T_max),
            _clamp01(queue_len          / 10),
            _clamp01(route_switching_recent / 10),
            _clamp01(echo_age_avg       / 500),
            T_ratio,
        ]

        # 에코를 위해 에코 컨트롤러로 p 계산. 
        # 에코 컨트롤러에서 ACTOR가 state를 입력받아 p값을 예측하여 반환
        # 진행 중에 last state 가 저장됨
        p = self.params['controller'].predict(state)

        visited = set(packet.route_memory)
        candidates = [n for n in self.neighbors if n not in visited]
        if not candidates:
            candidates = self.neighbors

        y_star = min(candidates, key=lambda n: self.Q[dst][n])
        q = current_tick - packet.queue_entry_tick
        s = 1

        eta2 = p * eta * k

        # 선정된 이웃 y*는 항상 echo, 나머지는 확률 p로 echo
        echo_set = {y_star}
        for n in self.neighbors:
            if n != y_star and random.random() < p:
                echo_set.add(n)

        # 확률적으로 선정된 echo_set에 대해 Q 테이블 업데이트
        for n in echo_set:
            t_n = all_nodes[n].best_estimate(dst, exclude_node=self.id)
            td_error = q + s + t_n - self.Q[dst][n]
            if n == y_star:
                self.Q[dst][n] += eta * td_error
            else:
                self.Q[dst][n] += eta2 * td_error

            # last_echo_tick 갱신
            self.last_echo_tick[dst][n] = current_tick

        # TD_error_ema 갱신 (y*의 TD error 기준)
        td_error_ystar = q + s + all_nodes[y_star].best_estimate(dst, exclude_node=self.id) - self.Q[dst][y_star]
        self.td_error_ema = (1 - self.td_ema_alpha) * self.td_error_ema + self.td_ema_alpha * abs(td_error_ystar)

        # y_star_history 갱신
        self.y_star_history.append((current_tick, y_star))

        new_memory = packet.route_memory + [self.id]
        if len(new_memory) > L:
            new_memory = new_memory[-L:]
        packet.route_memory = new_memory

        # 라우팅 결정 (다음 홉 반환)
        return y_star
