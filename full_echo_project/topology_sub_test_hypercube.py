# =============================================================================
# [요약] 5 차원 하이퍼큐브 토폴로지 정의 — 32 노드 / 80 링크
# - 출처: Choi & Yeung, "Predictive Q-Routing: A Memory-based Reinforcement
#   Learning Approach to Adaptive Traffic Control", NeurIPS 1995, 4.2 절
#   ("a 32-node hypercube"). 같은 계열의 Kumar & Miikkulainen 1997 (DRQ),
#   1999 (CDRQ) 는 128 노드 7 차원 하이퍼큐브를 사용.
# - 정의: 노드 번호를 5 자리 2 진수로 썼을 때, 정확히 한 자리만 다른 두 노드를 잇는다.
#   예) 노드 5 (00101) 의 이웃 = 4 (00100), 7 (00111), 1 (00001), 13 (01101), 21 (10101)
# - 정의만으로 모양이 하나로 정해지므로 그림에서 링크를 옮겨 적는 과정이 없다.
#
# [구조 특징]
# - 모든 노드의 이웃 수 = 5 (차원 수). 링크 수 = 32 × 5 / 2 = 80
# - 두 노드 사이 최단 홉 수 = 2 진수에서 다른 자리 수 (해밍 거리), 최대 5 홉
# - 거리 h 인 두 노드 사이에 길이가 같은 최단경로가 h! 개 있다 (예: 5 홉이면 120 개)
#   → 병목 하나를 피하는 망이 아니라 여러 길에 부하를 나누는 능력을 보는 망
#
# [주의 — 원 논문과의 차이]
# - PQ 논문은 하이퍼큐브 실험의 트래픽 패턴·부하를 자세히 적지 않았다.
#   여기서는 다른 토폴로지와 같이 시뮬레이터의 무작위 출발지·목적지 트래픽을 쓴다.
# =============================================================================
DIMENSION = 5
NUM_NODES = 2 ** DIMENSION   # 32

# 각 노드에서 자기보다 번호가 큰 이웃만 골라 링크를 한 번씩만 넣는다.
# n ^ (1 << b) 는 n 의 b 번째 자리만 뒤집은 번호 = 그 자리 방향의 이웃.
EDGES = [
    (n, n ^ (1 << b))
    for n in range(NUM_NODES)
    for b in range(DIMENSION)
    if n < n ^ (1 << b)
]


def build_adjacency():
    adj = {i: [] for i in range(NUM_NODES)}
    for u, v in EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj


ADJACENCY = build_adjacency()
