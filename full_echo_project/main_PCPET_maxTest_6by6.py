import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

SEED = 200

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C_VALUE = 0.3   # c 는 고정. 이 스크립트는 pfe_b_max 만 sweep.

BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C_VALUE}

# 단일 알고리즘 — pfe_c_pre_echo_tick
ALGORITHM = 'pfe_c_pre_echo_tick'

# pfe_b_max sweep 설정
B_MAX_VALUES = [0.2, 0.3, 0.4]

# 적녹색약 친화 sequential colormap (viridis) — b_max 가 클수록 밝은 톤
B_COLORS = plt.cm.viridis(np.linspace(0, 1, len(B_MAX_VALUES)))

# 점선 패턴 — 길게 (8 단위 on, 4 단위 off) 잘 보이게
DASH_STYLE = (0, (8, 4))

STAT_INTERVAL = 200
MD_PATH = 'result_PCPET_maxTest_6by6.md'

EXPERIMENTS = [
    # {'lam': 2, 'total_ticks': 80000, 'title': 'λ=2'},
    {'lam': 3.5, 'total_ticks': 80000, 'title': 'λ=3.5'},
    {'lam': 3.7, 'total_ticks': 80000, 'title': 'λ=3.7'},
    # {'lam': 4, 'total_ticks': 14000, 'title': 'λ=4'},
]


# -------------------------------------------------------------------------
# b_max sweep 실험: 3개 부하별 ADT 그래프 (각 부하에 7개 b_max 점선)
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(45, 8))
    fig.suptitle(
        f"6x6 Grid — pfe_c_pre_echo_tick pfe_b_max sweep "
        f"(c={C_VALUE}, L={L}, seed={SEED})"
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# 6x6 Grid PFE_c_pre_echo_tick pfe_b_max sweep\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            lam = exp['lam']
            total_ticks = exp['total_ticks']
            x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

            md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
            md.write("| pfe_b_max | generated | delivered | undelivered | delivery_rate |\n")
            md.write("|-----------|-----------|-----------|-------------|---------------|\n")

            print(f"\n=== λ={lam} ===")
            for b_max, color in zip(B_MAX_VALUES, B_COLORS):
                random.seed(SEED)
                np.random.seed(SEED)
                params = {**BASE_PARAMS, 'pfe_b_max': b_max}
                label = f"b_max={b_max}"
                print(f"  Running {label}...")

                sim = Simulator(algorithm=ALGORITHM, params=params, seed=SEED, topology=TOPOLOGY_GRID)
                adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

                gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                rate = (dlv / gen * 100) if gen > 0 else 0.0
                print(f"    {label:14s} generated={gen:6d}  delivered={dlv:6d}  "
                      f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                md.write(f"| {b_max} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

                # ---- T_est / T_max + PFE 진단 ----
                t_est_series = getattr(sim, 't_est_series', None)
                t_max_series = getattr(sim, 't_max_series', None)
                if t_est_series and t_max_series:
                    n_chunks   = max(1, total_ticks // 1000)
                    chunk_size = max(1, len(t_est_series) // n_chunks)
                    t_est_chunks = [
                        float(np.mean(t_est_series[i:i + chunk_size]))
                        for i in range(0, chunk_size * n_chunks, chunk_size)
                    ]
                    t_max_chunks = [
                        float(np.mean(t_max_series[i:i + chunk_size]))
                        for i in range(0, chunk_size * n_chunks, chunk_size)
                    ]
                    print(f"      [T_est ] {' '.join(f'{m:6.2f}' for m in t_est_chunks)}")
                    print(f"      [T_max ] {' '.join(f'{m:6.2f}' for m in t_max_chunks)}")

                    fe_series = getattr(sim, 'pfe_full_echo_ratio_series', None)
                    tp_series = getattr(sim, 'pfe_total_point_series', None)
                    if fe_series and tp_series:
                        fe_chunks = [
                            float(np.mean(fe_series[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        tp_chunks = [
                            float(np.mean(tp_series[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        print(f"      [FE_rt ] {' '.join(f'{m:6.3f}' for m in fe_chunks)}")
                        print(f"      [Point ] {' '.join(f'{m:6.2f}' for m in tp_chunks)}")

                # 긴 점선 (8 on / 4 off) — viridis 톤
                ax.plot(x_axis, adt, label=label, color=color)

            md.write("\n")
            ax.set_title(exp['title'])
            ax.set_xlabel('Simulator Time')
            ax.set_ylabel('Average Delivery Time')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'result_PCPET_maxTest_6by6.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()

    print(f"\n로그: {MD_PATH}")


if __name__ == '__main__':
    run_all()
