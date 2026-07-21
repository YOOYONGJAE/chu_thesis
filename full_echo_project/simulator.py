# =============================================================================
# [요약] tick 기반 네트워크 시뮬레이터 — 패킷 생성 / 라우팅 / 통계 집계
# - 4종 (Q-routing / AQFE / AQRERM / AQPRICE) 전용 정리판
# - 진단 시리즈: ADT / 큐 길이 / T_est / T_max / AQPRICE 포인트·에코 발동 비율
# - RL 컨트롤러 연동, T_max 가속 감쇠, AdE 진단 등은
#   legacy_algorithm_files/simulator.py 에 보존
# =============================================================================
import math
import random
import numpy as np
from node import Node, Packet


class Simulator:
    def __init__(self, algorithm, params, seed, topology):
        """
        algorithm: 'q_routing' | 'aqfe' | 'aqrerm' | 'aqprice'
        params: {'eta': float, 'k': float, 'L': int, 'c': float, ...}
        seed: 난수 시드 (rng_traffic, rng_order 분리)
        topology: {'num_nodes': int, 'adjacency': dict}
        """
        self.algorithm = algorithm
        self.params = params.copy()
        self.num_nodes = topology['num_nodes']
        adjacency = {k: list(v) for k, v in topology['adjacency'].items()}

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

    # =========================================================================
    # [링크 절단 기능 — 이번 연구 마감 범위 제외]
    # 연구 진행 중 링크 장애 발생 시 각 알고리즘의 라우팅 적응력을 확인하기 위해
    # 추가된 기능입니다. 이번 논문 마감 범위에는 포함하지 않으므로 주석 처리하여
    # 제외합니다. 재활성화가 필요하면 아래 주석을 해제하고, run() 내의 link_cuts
    # 관련 주석 처리 구간도 함께 해제하십시오.
    # =========================================================================
    # def cut_link(self, u, v):
    #     if v in self.nodes[u].neighbors:
    #         self.nodes[u].neighbors.remove(v)
    #     if u in self.nodes[v].neighbors:
    #         self.nodes[v].neighbors.remove(u)

    # -------------------------------------------------------------------------
    # 시뮬레이션 실행
    # lam          : Poisson 파라미터 (네트워크 전체 tick당 평균 패킷 생성 수)
    # total_ticks  : 총 tick 수
    # stat_interval: 통계 집계 주기 (tick)
    # link_cuts    : [(tick, u, v)] — 현재 비활성 (링크 절단 기능 주석 처리 참조)
    # 반환: ADT 리스트 (stat_interval tick마다 1개)
    # -------------------------------------------------------------------------
    def run(self, lam, total_ticks=10000, stat_interval=100, link_cuts=None):
        # =====================================================================
        # [링크 절단 초기화 — 이번 연구 마감 범위 제외]
        # link_cuts 파라미터를 실제로 처리하는 초기화 블록입니다.
        # cut_link() 메서드 주석 참조. 재활성화 시 아래 주석을 해제하십시오.
        # =====================================================================
        # if link_cuts:
        #     cut_tick, cut_u, cut_v = link_cuts[0]
        # else:
        #     cut_tick = cut_u = cut_v = None
        cut_tick = cut_u = cut_v = None  # 링크 절단 비활성 — 항상 None 고정

        adt_series = []
        queue_len_series = []   # stat_interval 시점의 모든 노드 queue 길이 합
        t_est_series     = []   # stat_interval 시점의 네트워크 평균 T_est (finite 만)
        t_max_series     = []   # stat_interval 시점의 네트워크 평균 T_max (finite 만)
        # AQPRICE 진단 시계열 — stat_interval 윈도우 단위로 측정
        # 다른 알고리즘에선 카운터/total_point 가 0 이므로 시계열도 모두 0
        pfe_total_point_series     = []  # 윈도우 종료 시점의 네트워크 평균 누적 포인트
        pfe_full_echo_ratio_series = []  # 윈도우 동안 Full Echo 발동 / 라우팅 호출
        window_delivered = []

        # =====================================================================
        # [절단 전/후 link_usage 분리 추적 — 이번 연구 마감 범위 제외]
        # 링크 절단 시점을 기준으로 트래픽 분포 변화를 추적하는 집계 변수입니다.
        # 재활성화 시 아래 주석을 해제하십시오.
        # =====================================================================
        # link_usage_pre_cut  = {k: 0 for k in self.link_usage}
        # link_usage_post_cut = {k: 0 for k in self.link_usage}

        # 도달 여부 카운터
        total_generated = 0
        total_delivered = 0

        for tick in range(total_ticks):

            # =====================================================================
            # [링크 차단 적용 — 이번 연구 마감 범위 제외]
            # 지정된 tick에 링크를 실제로 제거하는 블록입니다.
            # 재활성화 시 아래 주석을 해제하고 cut_link() 및 초기화 블록도 함께
            # 해제하십시오.
            # =====================================================================
            # if tick == cut_tick:
            #     self.cut_link(cut_u, cut_v)
            #     print(f"  [tick {tick}] 링크 ({cut_u}, {cut_v}) 차단")

            # 1. 지난 tick까지 들어온 패킷들을 각 노드 queue로 이동
            #    (이번 tick에 도착한 패킷은 이번 tick 처리 대상에서 제외)
            for node in self.nodes:
                for pkt in node.incoming:
                    pkt.queue_entry_tick = tick
                    node.queue.append(pkt)
                node.incoming = []

            # 2. 각 노드에서 패킷 1개 처리 (매 tick 랜덤 순서)
            order = list(range(self.num_nodes))
            self.rng_order.shuffle(order)
            for i in order:
                node = self.nodes[i]
                if not node.queue:
                    continue

                pkt = node.queue.popleft()

                if not node.neighbors:
                    continue

                # 라우팅 결정 (다음 홉 반환)
                next_hop = node.route(pkt, tick, self.nodes)

                # 링크 사용량 기록 — 목적지 도착(마지막 홉) 포함 모든 링크 통과 카운트
                link = (i, next_hop) if (i, next_hop) in self.link_usage else (next_hop, i)
                self.link_usage[link] += 1
                # =====================================================================
                # [절단 전/후 분리 카운트 — 이번 연구 마감 범위 제외]
                # 링크 사용량을 절단 시점 기준으로 pre/post 로 나눠 기록합니다.
                # 재활성화 시 아래 주석과 link_usage_pre/post_cut 초기화 블록을
                # 함께 해제하십시오.
                # =====================================================================
                # if cut_tick is None or tick < cut_tick:
                #     link_usage_pre_cut[link] += 1
                # else:
                #     link_usage_post_cut[link] += 1

                # 목적지 도착 시 전달 시간 기록, 아니면 다음 홉으로 이동
                if next_hop == pkt.dst:
                    delivery_time = tick + 1 - pkt.created_at
                    window_delivered.append(delivery_time)
                    total_delivered += 1
                else:
                    self.nodes[next_hop].incoming.append(pkt)

            # 2.5 AQPRICE per-tick 포인트 적립 — 모든 노드 매 tick
            #     (적립 비활성 노드는 즉시 no-op. 큐가 비어 라우팅 안 한 노드도 적립 진행)
            for node in self.nodes:
                node.tick_accumulate_point(tick)

            # 3. 새 패킷 생성 → 이번 tick에는 incoming에만 넣고 끝
            n_packets = np.random.poisson(lam)
            for _ in range(n_packets):
                src = self.rng_traffic.randint(0, self.num_nodes - 1)
                dst = self.rng_traffic.randint(0, self.num_nodes - 2)
                if dst >= src:
                    dst += 1
                pkt = Packet(src=src, dst=dst, created_at=tick)
                self.nodes[src].incoming.append(pkt)
                total_generated += 1

            # 4. 통계 집계
            if (tick + 1) % stat_interval == 0:
                if window_delivered:
                    adt_series.append(np.mean(window_delivered))
                else:
                    adt_series.append(float('nan'))

                # 모든 노드 queue 길이 합산 (현재 시점 네트워크 적체량)
                total_qlen = sum(len(node.queue) for node in self.nodes)
                queue_len_series.append(total_qlen)

                # 네트워크 평균 T_est / T_max — finite 값만 모아 평균
                t_est_vals = [n.T_est for n in self.nodes if math.isfinite(n.T_est)]
                t_max_vals = [n.T_max for n in self.nodes if math.isfinite(n.T_max)]
                t_est_series.append(float(np.mean(t_est_vals)) if t_est_vals else 0.0)
                t_max_series.append(float(np.mean(t_max_vals)) if t_max_vals else 0.0)

                # AQPRICE 진단: 윈도우 동안 누적된 노드 카운터를 합산해 비율 계산 후 리셋
                tp_sum          = sum(n.total_point for n in self.nodes)
                fe_count_sum    = sum(n.pfe_window_full_echo_count for n in self.nodes)
                route_count_sum = sum(n.pfe_window_route_count     for n in self.nodes)
                pfe_total_point_series.append(tp_sum / len(self.nodes) if self.nodes else 0.0)
                pfe_full_echo_ratio_series.append(
                    fe_count_sum / route_count_sum if route_count_sum > 0 else 0.0
                )
                for n in self.nodes:
                    n.pfe_window_full_echo_count = 0
                    n.pfe_window_route_count     = 0

                window_delivered = []

        # 결과 노출
        self.queue_len_series = queue_len_series
        self.t_est_series     = t_est_series
        self.t_max_series     = t_max_series
        self.pfe_total_point_series     = pfe_total_point_series
        self.pfe_full_echo_ratio_series = pfe_full_echo_ratio_series
        self.total_generated = total_generated
        self.total_delivered = total_delivered
        self.undelivered_count = sum(
            len(node.queue) + len(node.incoming) for node in self.nodes
        )

        # 에코 비용 진단 — 실행 전체 노드 카운터 합산
        # echo_cost = total_echo_queries / total_route_calls = 결정당 평균 조회 이웃 수
        self.total_echo_queries = sum(node.echo_query_count for node in self.nodes)
        self.total_route_calls  = sum(node.route_call_count for node in self.nodes)
        # =====================================================================
        # [절단 전/후 link_usage 결과 노출 — 이번 연구 마감 범위 제외]
        # 재활성화 시 아래 주석과 link_usage_pre/post_cut 초기화·집계 블록을
        # 함께 해제하십시오.
        # =====================================================================
        # self.link_usage_pre_cut  = link_usage_pre_cut
        # self.link_usage_post_cut = link_usage_post_cut
        return adt_series
