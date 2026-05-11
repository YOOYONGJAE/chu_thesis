import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A

SEED = 800
BASE_PARAMS = {'eta': 0.9, 'k': 0.556, 'L': 10}
TOPO = {'num_nodes': G_N, 'adjacency': G_A}

ALGORITHMS = ['aqrerm', 'aqlrerm']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqrerm_no_mem': 'AQRERM_no_mem',
          'aqlrerm': 'AQLRERM',
          'aqlrerm_no_mem': 'AQLRERM_no_mem',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'navy',
          'aqrerm_no_mem': 'magenta',
          'aqlrerm': 'darkorange',
          'aqlrerm_no_mem': 'cyan',
          'learned_aqrerm': 'brown', 'bandit_aqrerm': 'purple'}

STAT_INTERVAL = 100
CUT_TICK = 7000
TOTAL_TICKS = 14000

# 절단 시나리오: 두 다리만
CUT_SCENARIOS = [
    {'name': '(14,15) [bottom bridge]', 'cuts': [(CUT_TICK, 14, 15)]},
    {'name': '(2,3) [top bridge]',      'cuts': [(CUT_TICK, 2, 3)]},
]

# 부하: 세 가지
LAMBDAS = [1.5, 2, 2.5]

# c-sweep 설정
C_VALUES = [0.5]
MD_PATH = 'result_link_cut.md'


# -------------------------------------------------------------------------
# 한 c 값에 대한 실험: 2개 절단 x 3개 부하 = 6개 패널
# -------------------------------------------------------------------------
def run_one_c(c, md_file):
    params = {**BASE_PARAMS, 'c': c, 'memory_cut_tick': CUT_TICK}

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"6x6 Grid (c={c}, L={BASE_PARAMS['L']})")

    md_file.write(f"## c = {c}\n\n")

    for row, scenario in enumerate(CUT_SCENARIOS):
        md_file.write(f"### Cut {scenario['name']}\n\n")
        for col, lam in enumerate(LAMBDAS):
            ax = axes[row, col]
            md_file.write(f"#### λ={lam} ({TOTAL_TICKS} ticks)\n\n")
            md_file.write("| algo | generated | delivered | undelivered | delivery_rate |\n")
            md_file.write("|------|-----------|-----------|-------------|---------------|\n")

            print(f"\n=== c={c} Cut {scenario['name']} λ={lam} ===")
            for algo in ALGORITHMS:
                random.seed(SEED)
                np.random.seed(SEED)
                print(f"  Running {LABELS[algo]}...")

                sim = Simulator(algorithm=algo, params=params, seed=SEED, topology=TOPO)
                adt = sim.run(
                    lam=lam,
                    total_ticks=TOTAL_TICKS,
                    stat_interval=STAT_INTERVAL,
                    link_cuts=scenario['cuts']
                )

                gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                rate = (dlv / gen * 100) if gen > 0 else 0.0
                print(f"    {LABELS[algo]:18s} generated={gen:6d}  delivered={dlv:6d}  "
                      f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                md_file.write(f"| {LABELS[algo]} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

                x_axis = np.arange(1, len(adt) + 1) * STAT_INTERVAL
                ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

            md_file.write("\n")
            ax.axvline(x=CUT_TICK, color='red', linestyle='--', linewidth=1.5, label='Link cut')
            ax.set_title(f"λ={lam} — Cut {scenario['name']}")
            ax.set_xlabel('Simulator Time')
            ax.set_ylabel('Average Delivery Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"result_link_cut_c_{c}.png"
    plt.savefig(filename, dpi=150)
    print(f"결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# Link cut c-sweep (6x6 Grid)\n\n')
        for c in C_VALUES:
            run_one_c(c, md)
    print(f"\n모든 c-sweep 완료. 로그: {MD_PATH}")
