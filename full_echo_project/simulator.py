import math
import random
import numpy as np
from node import Node, Packet
from echo_controller import EchoController, LTrainController, LCloseController
from bandit_controller import BanditController

# aqlrerm_l_train 전용 reward shaping 가중치
# score = d_window + BETA_QUEUE * total_qlen + GAMMA_PEND * pending_count
# reward = (prev_score - score) / max(prev_score, 1e-6)
BETA_QUEUE = 1
GAMMA_PEND = 1


class Simulator:
    def __init__(self, algorithm, params, seed, topology):
        """
        algorithm: 'q_routing' | 'aqfe' | 'aqrerm'
        params: {'eta': float, 'k': float, 'L': int}
        seed: 난수 시드 (rng_traffic, rng_order 분리)
        topology: {'num_nodes': int, 'adjacency': dict}
        """
        self.algorithm = algorithm
        self.params = params.copy()
        self.num_nodes = topology['num_nodes']
        adjacency = {k: list(v) for k, v in topology['adjacency'].items()}

        # 학습 기반 알고리즘용 컨트롤러 생성
        if algorithm == 'learned_aqrerm':
            self.controller = EchoController()
            self.params['controller'] = self.controller
        elif algorithm == 'bandit_aqrerm':
            self.controller = BanditController()
            self.params['controller'] = self.controller
        elif algorithm == 'aqlrerm_l_train':
            self.controller = LTrainController()
            self.params['controller'] = self.controller
        elif algorithm == 'aqlrerm_l_close':
            self.controller = LCloseController()
            self.params['controller'] = self.controller
        else:
            self.controller = None

        self.nodes = [
            Node(i, adjacency[i], algorithm, self.params, self.num_nodes)
            for i in range(self.num_nodes)
        ]
        self.rng_traffic = random.Random(seed + 1000)  # 패킷 src/dst 생성용
        self.rng_order   = random.Random(seed + 2000)  # 노드 처리 순서 셔플용

        self.link_usage = {}
        for u, neighbors in adjacency.items():
            for v in neighbors:
                if (v, u) not in self.link_usage:
                    self.link_usage[(u, v)] = 0

    # -------------------------------------------------------------------------
    # 시뮬레이션 실행
    # lam          : Poisson 파라미터 (네트워크 전체 tick당 평균 패킷 생성 수)
    # total_ticks  : 총 tick 수
    # stat_interval: 통계 집계 주기 (tick)
    # 반환: ADT 리스트 (stat_interval tick마다 1개)
    # -------------------------------------------------------------------------
    def cut_link(self, u, v):
        if v in self.nodes[u].neighbors:
            self.nodes[u].neighbors.remove(v)
        if u in self.nodes[v].neighbors:
            self.nodes[v].neighbors.remove(u)

    def _build_global_state(self, d_window, total_qlen, pending_count,
                            total_generated, total_delivered,
                            prev_total_qlen, prev_pending_count,
                            current_tick):
        """
        16차원 글로벌 state 벡터를 만들어 LTrainController.set_window_L 에 전달.
        - 각 항목은 대략 [0, 1] 범위로 정규화 (학습 안정성 위한 스케일 정렬)
        - math.isfinite 로 inf/NaN 값을 거른 뒤 통계 계산 (dead-end Q 오염 안전망)
        - 정규화 분모는 추정치 — 시나리오 따라 튜닝 가능
        """
        nodes = self.nodes

        # 노드별 큐 길이
        qlens = [len(n.queue) for n in nodes]
        max_qlen = max(qlens) if qlens else 0
        std_qlen = float(np.std(qlens)) if qlens else 0.0

        # 노드별 T_ratio, TD_error, Q_spread 수집 (finite 만)
        t_ratios = []
        td_errors_abs = []
        q_spreads = []
        for n in nodes:
            if n.T_max > 0 and math.isfinite(n.T_max) and math.isfinite(n.T_est):
                tr = n.T_est / n.T_max
                if math.isfinite(tr):
                    t_ratios.append(tr)
            if math.isfinite(n.td_error_ema):
                td_errors_abs.append(abs(n.td_error_ema))
            # 노드 안에서 모든 dst 의 Q_spread (max-min) 평균
            node_dst_spreads = []
            for q_dict in n.Q.values():
                vals = [v for v in q_dict.values() if math.isfinite(v)]
                if vals:
                    node_dst_spreads.append(max(vals) - min(vals))
            if node_dst_spreads:
                q_spreads.append(float(np.mean(node_dst_spreads)))

        avg_T_ratio  = float(np.mean(t_ratios))      if t_ratios else 0.0
        max_T_ratio  = float(max(t_ratios))          if t_ratios else 0.0
        std_T_ratio  = float(np.std(t_ratios))       if t_ratios else 0.0
        avg_Q_spread = float(np.mean(q_spreads))     if q_spreads else 0.0
        max_Q_spread = float(max(q_spreads))         if q_spreads else 0.0
        avg_TD_error = float(np.mean(td_errors_abs)) if td_errors_abs else 0.0

        # 누적/델타 지표
        delivered_rate = total_delivered / max(total_generated, 1)
        pending_ratio  = pending_count   / max(total_generated, 1)
        queue_delta    = total_qlen      - prev_total_qlen
        pending_delta  = pending_count   - prev_pending_count

        # 절단 후 경과 시간
        cut_tick = getattr(self, '_cut_tick', None)
        if cut_tick is None or current_tick < cut_tick:
            ticks_after_cut = 0
        else:
            ticks_after_cut = current_tick - cut_tick

        total_ticks = getattr(self, '_total_ticks', 14000)
        prev_L      = self.controller.cached_L if self.controller else 0
        l_max       = float(max(self.controller.ACTIONS)) if self.controller else 1.0

        # 16 차원 글로벌 state — 정규화 분모는 대략적 스케일
        state = [
            d_window         / 100.0,                       # 1. ADT
            total_qlen       / 200.0,                       # 2. 전체 큐 합
            max_qlen         / 30.0,                        # 3. 최악 노드 큐
            std_qlen         / 10.0,                        # 4. 부하 불균등성
            pending_ratio,                                  # 5. 미배달 비율 (0~1)
            delivered_rate,                                 # 6. 배달률 (0~1)
            queue_delta      / 200.0,                       # 7. 큐 변화 (signed)
            pending_delta    / max(total_generated, 1),     # 8. 미배달 변화 (signed)
            avg_T_ratio,                                    # 9. 평균 포화도
            max_T_ratio,                                    # 10. 최악 포화도
            std_T_ratio      / 0.5,                         # 11. 포화 불균등성
            avg_Q_spread     / 100.0,                       # 12. 평균 Q 분산
            max_Q_spread     / 100.0,                       # 13. 최악 Q 분산
            avg_TD_error     / 10.0,                        # 14. 평균 학습 잔차
            ticks_after_cut  / float(total_ticks),          # 15. 절단 후 경과 비율
            prev_L           / max(l_max, 1.0),             # 16. 직전 L
        ]
        return state

    def run(self, lam, total_ticks=10000, stat_interval=100, link_cuts=None):
        # link_cuts: [(tick, u, v)] — 단일 cut 만 사용 (없으면 None / 빈 리스트)
        # 호환을 위해 list 형태 그대로 받되, 첫 항목만 꺼내 단일 변수로 보관
        if link_cuts:
            cut_tick, cut_u, cut_v = link_cuts[0]
        else:
            cut_tick = cut_u = cut_v = None
        # 글로벌 state 계산용 (ticks_after_cut, 시뮬레이션 진행도)
        self._cut_tick    = cut_tick
        self._total_ticks = total_ticks

        adt_series = []
        queue_len_series = []   # stat_interval 시점의 모든 노드 queue 길이 합
        window_delivered = []

        # 도달 여부 카운터 (selection bias 진단용)
        total_generated = 0
        total_delivered = 0

        # 시뮬레이션 시작 전, learned_aqrerm의 prev_state 초기화
        prev_d_window = None
        prev_score    = None   # aqlrerm_l_train 전용 composite score 추적

        # 글로벌 state 의 delta 항목 계산용 (이전 window 값 기억)
        prev_total_qlen    = 0
        prev_pending_count = 0

        for tick in range(total_ticks):

            # 0. 링크 차단 — 예약된 단일 cut tick 도달 시 1회 실행
            if tick == cut_tick:
                self.cut_link(cut_u, cut_v)
                print(f"  [tick {tick}] 링크 ({cut_u}, {cut_v}) 차단")

            # 1. 지난 tick까지 들어온 패킷들을 각 노드 queue로 이동 (이번 tick에 도착한 패킷은 이번 tick 처리 대상에서 제외)
            for node in self.nodes:
                for pkt in node.incoming:
                    pkt.queue_entry_tick = tick
                    node.queue.append(pkt)
                node.incoming = []

            # 2. 각 노드에서 패킷 1개 처리 (매 tick 랜덤 순서)
            order = list(range(self.num_nodes))
            self.rng_order.shuffle(order) # 노드 처리 순서 랜덤 셔플
            for i in order:
                node = self.nodes[i]
                if not node.queue:
                    continue

                pkt = node.queue.popleft()

                if not node.neighbors:
                    continue

                # 라우팅 결정! (다음 홉 반환)
                # 다음 홉은 node.route()에서 알고리즘별로 결정하여 반환
                # aqlrerm_l_train 의 결정 로직 : 컨트롤러가 선택한 L에 따라 ADT 예측값이 가장 근접한 이웃을 다음 홉으로 선택
                next_hop = node.route(pkt, tick, self.nodes) 

                # 목적지에 도착하면 전달 시간 계산하여 window_delivered에 추가, 아니면 다음 홉으로 이동
                if next_hop == pkt.dst:
                    delivery_time = tick + 1 - pkt.created_at
                    window_delivered.append(delivery_time)
                    total_delivered += 1
                else:
                    link = (i, next_hop) if (i, next_hop) in self.link_usage else (next_hop, i)
                    self.link_usage[link] += 1
                    self.nodes[next_hop].incoming.append(pkt)

            # 3. 새 패킷 생성 -> 이번 tick에는 incoming에만 넣고 끝
            n_packets = np.random.poisson(lam) # 포아송 분포에 따라 tick당 생성할 패킷 수 결정
            
            for _ in range(n_packets):
                src = self.rng_traffic.randint(0, self.num_nodes - 1) # 출발 노드 랜덤 선택. 0번 부터 num_nodes-1 번 중에 하나 선택 
                dst = self.rng_traffic.randint(0, self.num_nodes - 2) # 도착 노드 랜덤 선택. 0번 부터 num_nodes-2 번 중에 하나 선택
                if dst >= src: # 출발지보다 낮은 것은 그대로 두고, 출발지와 같거나 높은 것은 +1 해서 목적지를 출발지와 겹치지 않도록 조정
                    dst += 1
                pkt = Packet(src=src, dst=dst, created_at=tick)
                self.nodes[src].incoming.append(pkt) # 생성한 패킷을 출발 노드의 incoming 리스트에 추가. 실제 노드의 queue로 이동하는 것은 다음 tick의 1번 단계에서 처리됨
                total_generated += 1 # 이번 tick에 생성된 패킷 수 카운트

            # 4. 통계 집계 및 컨트롤러 학습
            # stat_interval마다 100 tick 동안 배달된 패킷들의 평균 전달 시간 계산하여 ADT 시리즈에 추가
            if (tick + 1) % stat_interval == 0: # 현재는 stat_interver이 100이므로 100 tick마다 통계 집계
                if window_delivered:
                    d_window = np.mean(window_delivered) # 100 tick 동안 배달된 패킷들의 평균 전달 시간
                    adt_series.append(d_window) # ADT 시리즈에 추가
                else: # 100 tick 동안 배달된 패킷이 없으면 ADT 계산 불가 -> NaN 기록 (컨트롤러 학습에서는 보상 계산 시 d_window 존재 여부로 처리)
                    d_window = 0. 
                    adt_series.append(float('nan')) # ADT 계산 불가 시 NaN 기록

                # 모든 노드 queue 길이 합산 (현재 시점 네트워크 적체량)
                total_qlen = sum(len(node.queue) for node in self.nodes)
                queue_len_series.append(total_qlen)

                # learned_aqrerm: 100 tick마다 reward로 train
                # if self.controller is not None and self.controller.last_state is not None:
                #     reward = -d_window
                #     next_state = self.controller.last_state  # 현재 상태를 next_state로 사용
                #     self.controller.train(reward, next_state)

                if self.controller is not None:
                    if self.algorithm in ('aqlrerm_l_train', 'aqlrerm_l_close'):

                        # 현재까지 네트워크에 남아 있는 패킷 수
                        pending_count = total_generated - total_delivered

                        # 이전 window 대비 변화량
                        queue_delta = total_qlen - prev_total_qlen
                        pending_delta = pending_count - prev_pending_count

                        # 0 나눗셈 방지용 기준값
                        d_ref = max(getattr(self, "d_ref", 1.0), 1.0)
                        q_ref = max(getattr(self, "q_ref", 1.0), 1.0)
                        p_ref = max(getattr(self, "p_ref", 1.0), 1.0)

                        # 기준값을 너무 급하게 바꾸지 않기 위한 EMA 업데이트
                        # EMA: Exponential Moving Average, 최근 값에 더 큰 비중을 주는 이동평균
                        self.d_ref = 0.95 * d_ref + 0.05 * max(d_window, 1.0)
                        self.q_ref = 0.95 * q_ref + 0.05 * max(total_qlen, 1.0)
                        self.p_ref = 0.95 * p_ref + 0.05 * max(pending_count, 1.0)

                        # 단위가 다른 값들을 정규화
                        d_norm = d_window / self.d_ref
                        q_norm = total_qlen / self.q_ref
                        p_norm = pending_count / self.p_ref

                        # 악화 방향만 별도로 반영
                        q_delta_norm = max(queue_delta, 0) / self.q_ref
                        p_delta_norm = max(pending_delta, 0) / self.p_ref

                        # 합성 cost: 작을수록 좋은 상태
                        # pending_delta를 강하게 둔 이유:
                        # 링크 단절 후 회복 실패는 "현재 많이 남아 있음"보다
                        # "계속 더 쌓이고 있음"이 더 위험한 신호이기 때문
                        score = (
                            1.0 * d_norm
                            + 0.5 * q_norm
                            + 0.5 * p_norm
                            + 1.0 * q_delta_norm
                            + 2.0 * p_delta_norm
                        )
                        # 1) 글로벌 state 로 다음 window 의 L 을 먼저 sampling
                        #    (train 보다 먼저 호출해야 latest_state 가 fresh 한 상태로
                        #     train 의 advantage 계산에 들어감)
                        if self.algorithm == 'aqlrerm_l_train':
                            global_state = self._build_global_state(
                                d_window, total_qlen, pending_count,
                                total_generated, total_delivered,
                                prev_total_qlen, prev_pending_count,
                                tick + 1,
                            )
                            self.controller.set_window_L(global_state, tick + 1) # 컨트롤러가 글로벌 state 보고 L 결정하도록 호출

                        # 2) 그 다음에 reward 계산 + train
                        if prev_score is not None:
                            reward = -score
                            self.controller.train(reward)
                        prev_score = score

                        # close 전용: 적응형 stress 판정 + cached_L 갱신
                        # update_stress 가 (d_window, total_qlen, pending_count) 로 instability 계산,
                        # set_window_L 이 새로 갱신된 stressed flag 로 cached_L = L_NORMAL or L_STRESS 결정
                        if self.algorithm == 'aqlrerm_l_close':
                            self.controller.update_stress(d_window, total_qlen, pending_count)
                            self.controller.set_window_L(None, tick + 1)

                        # 다음 stat_interval 의 delta 계산용 prev 값 갱신
                        prev_total_qlen    = total_qlen
                        prev_pending_count = pending_count
                    else:
                        # 그 외 컨트롤러 (learned_aqrerm, bandit_aqrerm) 는 기존 ADT 보상
                        if prev_d_window is not None:
                            reward = (prev_d_window - d_window) / max(prev_d_window, 1e-6)
                            self.controller.train(reward)
                        prev_d_window = d_window

                window_delivered = []

        # 시뮬레이션 종료 시점에 네트워크에 남아있는 미배달 패킷 수
        self.queue_len_series = queue_len_series
        self.total_generated = total_generated
        self.total_delivered = total_delivered
        self.undelivered_count = sum(
            len(node.queue) + len(node.incoming) for node in self.nodes
        )
        return adt_series
