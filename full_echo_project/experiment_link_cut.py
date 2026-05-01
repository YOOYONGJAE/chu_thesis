import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A

SEED = 800
PARAMS = {'eta': 0.9, 'k': 0.556, 'L': 3}
TOPO = {'num_nodes': G_N, 'adjacency': G_A}

ALGORITHMS = ['q_routing', 'aqfe', 'aqrerm']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'green'}

STAT_INTERVAL = 100
CUT_TICK = 7000
CUT_LINK = (14, 15)

EXPERIMENTS = [
    {'lam': 1, 'total_ticks': 14000, 'title': f'Low Load (λ=1) — Link {CUT_LINK} cut at tick {CUT_TICK}'},
    {'lam': 3, 'total_ticks': 14000, 'title': f'High Load (λ=3) — Link {CUT_LINK} cut at tick {CUT_TICK}'},
]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, exp in zip(axes, EXPERIMENTS):
    print(f"\n=== λ={exp['lam']} ===")
    for algo in ALGORITHMS:
        random.seed(SEED)
        np.random.seed(SEED)
        print(f"  Running {LABELS[algo]}...")

        sim = Simulator(algorithm=algo, params=PARAMS, seed=SEED, topology=TOPO)
        adt = sim.run(
            lam=exp['lam'],
            total_ticks=exp['total_ticks'],
            stat_interval=STAT_INTERVAL,
            link_cuts=[(CUT_TICK, *CUT_LINK)]
        )

        x_axis = np.arange(1, len(adt) + 1) * STAT_INTERVAL
        ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

    ax.axvline(x=CUT_TICK, color='red', linestyle='--', linewidth=1.5, label=f'Link {CUT_LINK} cut')
    ax.set_title(exp['title'])
    ax.set_xlabel('Simulator Time')
    ax.set_ylabel('Average Delivery Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_link_cut.png', dpi=150)
print("\n결과 저장: results_link_cut.png")
