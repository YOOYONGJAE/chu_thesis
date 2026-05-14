import random
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from echo_controller import LTrainController
from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A

SEED = 800
BASE_PARAMS = {'eta': 0.9, 'k': 0.556, 'L': 3}
TOPO = {'num_nodes': G_N, 'adjacency': G_A}

ALGORITHMS = ['aqrerm', 'aqlrerm', 'aqlrerm_no_mem', 'aqlrerm_l_train', 'aqlrerm_l_close']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqrerm_no_mem': 'AQRERM_no_mem',
          'aqlrerm': 'AQLRERM',
          'aqlrerm_no_mem': 'AQLRERM_no_mem',
          'aqlrerm_l_train': 'AQLRERM_L_TRAIN',
          'aqlrerm_l_close': 'AQLRERM_L_CLOSE',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'navy',
          'aqrerm_no_mem': 'magenta',
          'aqlrerm': 'darkorange',
          'aqlrerm_no_mem': 'cyan',
          'aqlrerm_l_train': 'black',
          'aqlrerm_l_close': 'olive',
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
LAMBDAS = [1.5, 2, 2.3]

# c-sweep 설정
C_VALUES = [0.5]
MD_PATH = 'result_link_cut.md'


# -------------------------------------------------------------------------
# 한 c 값에 대한 실험: 2개 절단 x 3개 부하 = 6개 패널
# -------------------------------------------------------------------------
def run_one_c(c, md_file):
    params = {**BASE_PARAMS, 'c': c, 'memory_cut_tick': CUT_TICK}

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    train_l_range = f"{min(LTrainController.ACTIONS)}~{max(LTrainController.ACTIONS)}"
    fig.suptitle(f"6x6 Grid (c={c}, L={BASE_PARAMS['L']}, TRAIN_L={train_l_range})")

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
                # AQLRERM_L_TRAIN 은 학습 후보 L 범위도 같이 표기
                if algo == 'aqlrerm_l_train':
                    train_l_range = f"{min(LTrainController.ACTIONS)}~{max(LTrainController.ACTIONS)}"
                    print(f"  알고리즘 시작 >> {LABELS[algo]}... (TRAIN_L={train_l_range})")
                else:
                    print(f"  알고리즘 시작 >> {LABELS[algo]}...")

                # 시뮬레이터 생성, 링크 절단 시나리오 전달
                sim = Simulator(algorithm=algo, params=params, seed=SEED, topology=TOPO)
                # 링크 절단 시나리오 전달
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

                # ---- selected_L 진단 로그 (L_TRAIN / L_CLOSE) ----
                if algo in ('aqlrerm_l_train', 'aqlrerm_l_close') and sim.controller is not None:
                    L_hist = sim.controller.L_history
                    if L_hist:
                        total = len(L_hist)
                        mean_L = sum(L_hist) / total
                        counts = Counter(L_hist)
                        dist_str = "  ".join(
                            f"L={k}:{v:>6d}({v/total*100:5.1f}%)"
                            for k, v in sorted(counts.items())
                        )
                        n_chunks = 10
                        chunk_size = max(1, total // n_chunks)
                        chunk_means = [
                            float(np.mean(L_hist[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        chunks_str = " ".join(f"{m:5.2f}" for m in chunk_means)
                        print(f"      [L stats] n={total}  mean={mean_L:.2f}  "
                              f"min={min(L_hist)}  max={max(L_hist)}")
                        print(f"      [L dist ] {dist_str}")
                        print(f"      [L time ] {chunks_str}   (시간순 10등분 평균, per-routing)")

                    # per-window 분포 (window 마다 한 번씩 기록된 L)
                    L_w_hist = getattr(sim.controller, 'L_window_history', None)
                    if L_w_hist:
                        n_w = len(L_w_hist)
                        mean_w = sum(L_w_hist) / n_w
                        counts_w = Counter(L_w_hist)
                        dist_w_str = "  ".join(
                            f"L={k}:{v:>4d}({v/n_w*100:5.1f}%)"
                            for k, v in sorted(counts_w.items())
                        )
                        n_chunks = 10
                        chunk_size = max(1, n_w // n_chunks)
                        chunk_means_w = [
                            float(np.mean(L_w_hist[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        chunks_w_str = " ".join(f"{m:5.2f}" for m in chunk_means_w)
                        print(f"      [Lw stat] n={n_w}  mean={mean_w:.2f}  "
                              f"min={min(L_w_hist)}  max={max(L_w_hist)}")
                        print(f"      [Lw dist] {dist_w_str}")
                        print(f"      [Lw time] {chunks_w_str}   (시간순 10등분 평균, per-window)")

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
