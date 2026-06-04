import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_nsfnet import NUM_NODES as NSFNET_NUM_NODES, ADJACENCY as NSFNET_ADJACENCY

SEED = 800

TOPOLOGY_NSFNET = {'num_nodes': NSFNET_NUM_NODES, 'adjacency': NSFNET_ADJACENCY}

ETA = 0.9
K = 0.5             # AQRERM 논문 기준 — eta2 = k · R_x

BASE_PARAMS = {'eta': ETA, 'k': K}

ALGORITHM = 'aqrerm'
L_VALUES  = [0, 1, 3, 8, 10]

STAT_INTERVAL = 100

EXPERIMENTS = [
    {'lam': 1,   'total_ticks': 5000,  'title': 'λ=1'},
    {'lam': 2,   'total_ticks': 10000, 'title': 'λ=2'},
    {'lam': 3,   'total_ticks': 14000, 'title': 'λ=3'},
    {'lam': 3.7, 'total_ticks': 14000, 'title': 'λ=3.7'},
]

MD_PATH = 'result_aqrerm_l_nsfnet.md'


def run_one_lambda(ax, lam, total_ticks, md_file, colors):
    md_file.write(f"### λ={lam} ({total_ticks} ticks)\n\n")
    md_file.write("| L | generated | delivered | undelivered | delivery_rate |\n")
    md_file.write("|---|-----------|-----------|-------------|---------------|\n")

    x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

    print(f"\n=== AQRERM L-sweep (NSFNET) λ={lam} ===")
    for color, L in zip(colors, L_VALUES):
        random.seed(SEED)
        np.random.seed(SEED)
        params = {**BASE_PARAMS, 'L': L}
        print(f"  Running L={L}...")

        sim = Simulator(algorithm=ALGORITHM, params=params, seed=SEED, topology=TOPOLOGY_NSFNET)
        adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

        gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
        rate = (dlv / gen * 100) if gen > 0 else 0.0
        print(f"    L={L:2d}  generated={gen:6d}  delivered={dlv:6d}  "
              f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
        md_file.write(f"| {L} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

        ax.plot(x_axis, adt, label=f'L={L}', color=color)

    md_file.write("\n")
    ax.set_xlabel('Simulator Time')
    ax.set_ylabel('Average Delivery Time')
    ax.legend()
    ax.grid(True, alpha=0.3)


if __name__ == '__main__':
    # Okabe-Ito 색맹 안전 팔레트 (L=3은 vermillion 으로 강조)
    L_COLORS = {0: 'black', 1: '#0072B2', 3: '#D55E00', 8: '#009E73', 10: '#CC79A7'}
    colors = [L_COLORS[L] for L in L_VALUES]

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQRERM L-sweep on NSFNET\n\n')

        fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(24, 5), squeeze=False)
        axes = axes.flatten()  # EXPERIMENTS 가 1 개여도 1D 배열로 유지
        fig.suptitle("AQRERM L-sweep (NSFNET)")

        for ax, exp in zip(axes, EXPERIMENTS):
            run_one_lambda(ax, exp['lam'], exp['total_ticks'], md, colors)
            ax.set_title(exp['title'])

        plt.tight_layout()
        filename = "result_aqrerm_l_nsfnet.png"
        plt.savefig(filename, dpi=150)
        print(f"\n결과 저장: {filename}")
        plt.close()

    print(f"\n완료. 로그: {MD_PATH}")
