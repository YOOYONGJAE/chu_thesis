import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
from topology_nsfnet import NUM_NODES as NSFNET_NUM_NODES, ADJACENCY as NSFNET_ADJACENCY

SEED = 800

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}
TOPOLOGY_NSFNET = {'num_nodes': NSFNET_NUM_NODES, 'adjacency': NSFNET_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정 (AQRERM 논문 기준)
# -------------------------------------------------------------------------
ETA = 0.9
K = 0.5 / ETA       # eta*k = 0.5 이므로 k = 0.5/0.9 ≈ 0.556
L = 1
C = 0.5             # AQLRERM 큐 길이 페널티 가중치

PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

ALGORITHMS = ['q_routing', 'aqfe', 'aqrerm', 'aqlrerm']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqlrerm': 'AQLRERM',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'green',
          'aqlrerm': 'red',
          'learned_aqrerm': 'brown', 'bandit_aqrerm': 'purple'}

STAT_INTERVAL = 100

# -------------------------------------------------------------------------
# 단일 부하 실험: ADT vs tick 그래프
# -------------------------------------------------------------------------
def run_experiment(lam, total_ticks, ax, title, seed, topology):
    x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

    for algo in ALGORITHMS:
        random.seed(seed)
        np.random.seed(seed)
        print(f"  Running {LABELS[algo]} (lambda={lam})...")
        sim = Simulator(algorithm=algo, params=PARAMS, seed=seed, topology=topology)
        adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
        ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

    ax.set_title(title)
    ax.set_xlabel('Simulator Time')
    ax.set_ylabel('Average Delivery Time')
    ax.legend()
    ax.grid(True, alpha=0.3)


# -------------------------------------------------------------------------
# 토폴로지별 실험 실행 및 저장
# -------------------------------------------------------------------------
def run_topology(topology, name, filename):
    EXPERIMENTS = [
        {'lam': 1,   'total_ticks': 5000,  'title': 'λ=1'},
        {'lam': 2,   'total_ticks': 10000, 'title': 'λ=2'},
        {'lam': 3,   'total_ticks': 14000, 'title': 'λ=3'},
        {'lam': 3.7, 'total_ticks': 14000, 'title': 'λ=3.7'},
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(name)

    for ax, exp in zip(axes, EXPERIMENTS):
        print(f"\n=== {name} λ={exp['lam']} ===")
        run_experiment(lam=exp['lam'], total_ticks=exp['total_ticks'],
                       ax=ax, title=exp['title'],
                       seed=SEED, topology=topology)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    run_topology(TOPOLOGY_GRID,   '6x6 Grid',  'results_grid.png')
    run_topology(TOPOLOGY_NSFNET, 'NSFNET',    'results_nsfnet.png')
