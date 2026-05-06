import random
import numpy as np
from node import Node, Packet
from echo_controller import EchoController


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

        # learned_aqrerm 전용 EchoController 생성
        if algorithm == 'learned_aqrerm':
            self.controller = EchoController()
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

    def run(self, lam, total_ticks=10000, stat_interval=100, link_cuts=None):
        # link_cuts: [(tick, u, v), ...] — 해당 tick에 링크를 끊음
        if link_cuts is None:
            link_cuts = []
        adt_series = []
        window_delivered = []

        for tick in range(total_ticks):

            # 0. 링크 차단
            for cut_tick, u, v in link_cuts:
                if tick == cut_tick:
                    self.cut_link(u, v)
                    print(f"  [tick {tick}] 링크 ({u}, {v}) 차단")

            # 1. 지난 tick까지 쌓인 incoming만 queue로 이동
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

                next_hop = node.route(pkt, tick, self.nodes)

                if next_hop == pkt.dst:
                    delivery_time = tick + 1 - pkt.created_at
                    window_delivered.append(delivery_time)
                else:
                    link = (i, next_hop) if (i, next_hop) in self.link_usage else (next_hop, i)
                    self.link_usage[link] += 1
                    self.nodes[next_hop].incoming.append(pkt)

            # 3. 새 패킷 생성 -> 이번 tick에는 incoming에만 넣고 끝
            n_packets = np.random.poisson(lam)
            for _ in range(n_packets):
                src = self.rng_traffic.randint(0, self.num_nodes - 1)
                dst = self.rng_traffic.randint(0, self.num_nodes - 2)
                if dst >= src:
                    dst += 1
                pkt = Packet(src=src, dst=dst, created_at=tick)
                self.nodes[src].incoming.append(pkt)

            # 4. 통계 집계 및 컨트롤러 학습
            if (tick + 1) % stat_interval == 0:
                if window_delivered:
                    d_window = np.mean(window_delivered)
                    adt_series.append(d_window)
                else:
                    d_window = 0.0
                    adt_series.append(float('nan'))

                # learned_aqrerm: 100 tick마다 reward로 train
                if self.controller is not None and self.controller.last_state is not None:
                    reward = -d_window
                    next_state = self.controller.last_state  # 현재 상태를 next_state로 사용
                    self.controller.train(reward, next_state)

                window_delivered = []

        return adt_series
