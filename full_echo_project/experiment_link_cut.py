import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A

SEED = 800
PARAMS = {'eta': 0.9, 'k': 0.556, 'L': 1, 'c': 0.2}
TOPO = {'num_nodes': G_N, 'adjacency': G_A}

ALGORITHMS = ['q_routing', 'aqfe', 'aqrerm', 'aqlrerm']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqlrerm': 'AQLRERM',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'green',
          'aqlrerm': 'red',
          'learned_aqrerm': 'brown', 'bandit_aqrerm': 'purple'}

STAT_INTERVAL = 100
CUT_TICK = 7000
LAM = 2

# 5가지 절단 시나리오 (λ=3 고정)
EXPERIMENTS = [
    {'lam': LAM, 'total_ticks': 14000,
     'cuts': [(CUT_TICK, 14, 15)],
     'title': f'λ={LAM} — Cut (14,15) [bottom bridge]'},
    {'lam': LAM, 'total_ticks': 14000,
     'cuts': [(CUT_TICK, 2, 3)],
     'title': f'λ={LAM} — Cut (2,3) [top bridge]'},
    {'lam': LAM, 'total_ticks': 14000,
     'cuts': [(CUT_TICK, 13, 14)],
     'title': f'λ={LAM} — Cut (13,14) [near bridge]'},
    {'lam': LAM, 'total_ticks': 14000,
     'cuts': [(CUT_TICK, 28, 29)],
     'title': f'λ={LAM} — Cut (28,29) [right side]'},
    {'lam': LAM, 'total_ticks': 14000,
     'cuts': [(CUT_TICK, 18, 19), (CUT_TICK, 22, 23)],
     'title': f'λ={LAM} — Cut (18,19)+(22,23) [both sides]'},
]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)   # 6번째 칸은 비움

for ax, exp in zip(axes_flat, EXPERIMENTS):
    print(f"\n=== {exp['title']} ===")
    for algo in ALGORITHMS:
        random.seed(SEED)
        np.random.seed(SEED)
        print(f"  Running {LABELS[algo]}...")

        sim = Simulator(algorithm=algo, params=PARAMS, seed=SEED, topology=TOPO)
        adt = sim.run(
            lam=exp['lam'],
            total_ticks=exp['total_ticks'],
            stat_interval=STAT_INTERVAL,
            link_cuts=exp['cuts']
        )
        qlen = sim.queue_len_series

        # 로그: ADT, total_queue_len 시리즈
        # print(f"    [{LABELS[algo]}]")
        # print(f"      ADT  : {' '.join(f'{x:5.1f}' for x in adt)}")
        # print(f"      qlen : {' '.join(f'{x:4d}' for x in qlen)}")

        x_axis = np.arange(1, len(adt) + 1) * STAT_INTERVAL
        ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

    ax.axvline(x=CUT_TICK, color='red', linestyle='--', linewidth=1.5, label='Link cut')
    ax.set_title(exp['title'])
    ax.set_xlabel('Simulator Time')
    ax.set_ylabel('Average Delivery Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_link_cut.png', dpi=150)
print("\n결과 저장: results_link_cut.png")
