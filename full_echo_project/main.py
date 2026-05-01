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
L = 3

PARAMS = {'eta': ETA, 'k': K, 'L': L}

ALGORITHMS = ['q_routing', 'aqfe', 'aqrerm']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'green'}

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(name)

    print(f"\n=== {name} 저부하 실험 (lambda=1) ===")
    run_experiment(lam=1, total_ticks=5000, ax=axes[0], title='Low Load (λ=1)',
                   seed=SEED, topology=topology)

    print(f"=== {name} 고부하 실험 (lambda=3) ===")
    run_experiment(lam=3, total_ticks=14000, ax=axes[1], title='High Load (λ=3)',
                   seed=SEED, topology=topology)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    run_topology(TOPOLOGY_GRID,   '6x6 Grid',  'results_grid.png')
    run_topology(TOPOLOGY_NSFNET, 'NSFNET',    'results_nsfnet.png')
