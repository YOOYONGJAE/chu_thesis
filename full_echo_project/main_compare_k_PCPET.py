import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

SEED = 100

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정
# -------------------------------------------------------------------------
ETA = 0.9
L   = 3
C   = 0.3       # c 는 고정. 이 스크립트는 k 만 sweep.
B_MAX = 1.0     # pfe_b_max 도 고정

BASE_PARAMS = {'eta': ETA, 'L': L, 'c': C, 'pfe_b_max': B_MAX}

# 단일 알고리즘 — pfe_c_pre_echo_tick
ALGORITHM = 'pfe_c_pre_echo_tick'

# k sweep 설정 (10 개)
K_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 적녹색약 친화 sequential colormap (viridis) — k 가 클수록 밝은 톤
K_COLORS = plt.cm.viridis(np.linspace(0, 1, len(K_VALUES)))

# 점선 패턴 — 길게 (8 단위 on, 4 단위 off) 잘 보이게
DASH_STYLE = (0, (8, 4))

STAT_INTERVAL = 100
MD_PATH = 'result_compare_k_PCPET.md'

EXPERIMENTS = [
    {'lam': 1, 'total_ticks': 14000, 'title': 'λ=1'},
    {'lam': 2, 'total_ticks': 14000, 'title': 'λ=2'},
    {'lam': 3, 'total_ticks': 14000, 'title': 'λ=3'},
    {'lam': 3.5, 'total_ticks': 14000, 'title': 'λ=3.5'},
    {'lam': 3.7, 'total_ticks': 14000, 'title': 'λ=3.7'},
]


# -------------------------------------------------------------------------
# k sweep 실험: 3개 부하별 ADT 그래프 (각 부하에 10개 k 점선)
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(50, 8), squeeze=False)
    axes = axes.flatten()  # EXPERIMENTS 가 1 개여도 1D 배열로 유지
    fig.suptitle(
        f"6x6 Grid — pfe_c_pre_echo_tick k sweep "
        f"(c={C}, eta={ETA}, L={L}, b_max={B_MAX}, seed={SEED})"
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# 6x6 Grid PFE_c_pre_echo_tick k sweep\n\n')
        md.write(f'- Seed: {SEED}\n')
        md.write(f'- Fixed: eta={ETA}, L={L}, c={C}, pfe_b_max={B_MAX}\n')
        md.write(f'- Sweep: k in {K_VALUES}\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            lam = exp['lam']
            total_ticks = exp['total_ticks']
            x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

            md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
            md.write("| k | generated | delivered | undelivered | delivery_rate |\n")
            md.write("|---|-----------|-----------|-------------|---------------|\n")

            print(f"\n=== λ={lam} ===")
            for k, color in zip(K_VALUES, K_COLORS):
                random.seed(SEED)
                np.random.seed(SEED)
                params = {**BASE_PARAMS, 'k': k}
                label = f"k={k}"
                print(f"  Running {label}...")

                sim = Simulator(algorithm=ALGORITHM, params=params, seed=SEED, topology=TOPOLOGY_GRID)
                adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

                gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                rate = (dlv / gen * 100) if gen > 0 else 0.0
                print(f"    {label:10s} generated={gen:6d}  delivered={dlv:6d}  "
                      f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                md.write(f"| {k} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

                # ---- T_est / T_max 1000-tick 간격 평균 (시간 흐름 진단) ----
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

                    # ---- PFE 진단: Full Echo 발동 비율, 평균 누적 포인트 ----
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
    filename = 'result_compare_k_PCPET.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()

    print(f"\n로그: {MD_PATH}")


if __name__ == '__main__':
    run_all()
