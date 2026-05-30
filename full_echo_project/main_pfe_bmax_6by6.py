import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

SEED = 800

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정 (AQRERM 논문 기준)
# -------------------------------------------------------------------------
ETA = 0.9
K = 0.5             # AQRERM 논문 기준 — eta2 = k · R_x, k=0.5
L = 3
C = 0.5             # AQLRERM 식 큐 페널티 — 본 sweep 에선 고정

BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

# -------------------------------------------------------------------------
# b_max sweep 설정 — PFE_c 의 포인트 잔고 상한만 바꿔가며 비교
# -------------------------------------------------------------------------
BMAX_VALUES = [1.0, 1.2, 1.5, 1.8, 2.0] 

# 점차 진해지는 빨강 계열 — b_max 가 커질수록 진한 색
BMAX_COLORS = ['gold', 'orange', 'blue', 'red', 'black']

# 비교용 baseline 알고리즘 — 점선으로 표시
BASELINE_ALGOS = [
    ('aqrerm',   'AQRERM',           'navy'),
    # ('aqlrerm',  'AQLRERM_c=0.5',    'green'),
]

STAT_INTERVAL = 100
MD_PATH = 'result_pfe_bmax_6by6.md'

EXPERIMENTS = [
    # {'lam': 2.5, 'total_ticks': 10000, 'title': 'λ=2.5'},
    # {'lam': 3,   'total_ticks': 10000, 'title': 'λ=3'},
    {'lam': 3.5, 'total_ticks': 20000, 'title': 'λ=3.5'},
    {'lam': 4,   'total_ticks': 20000, 'title': 'λ=4'},
]


# -------------------------------------------------------------------------
# 진단 출력 헬퍼 — sim 객체에서 T_est/T_max + (PFE 면) FE_rt/Point 시계열 추출
# -------------------------------------------------------------------------
def print_diagnostics(sim, total_ticks, is_pfe):
    t_est_series = getattr(sim, 't_est_series', None)
    t_max_series = getattr(sim, 't_max_series', None)
    if not (t_est_series and t_max_series):
        return

    n_chunks   = max(1, total_ticks // 1000)
    chunk_size = max(1, len(t_est_series) // n_chunks)

    def chunk_mean(series):
        return [
            float(np.mean(series[i:i + chunk_size]))
            for i in range(0, chunk_size * n_chunks, chunk_size)
        ]

    t_est_chunks = chunk_mean(t_est_series)
    t_max_chunks = chunk_mean(t_max_series)
    print(f"      [T_est ] {' '.join(f'{m:6.2f}' for m in t_est_chunks)}   (1000-tick 평균)")
    print(f"      [T_max ] {' '.join(f'{m:6.2f}' for m in t_max_chunks)}   (1000-tick 평균)")

    if is_pfe:
        fe_series = getattr(sim, 'pfe_full_echo_ratio_series', None)
        tp_series = getattr(sim, 'pfe_total_point_series', None)
        if fe_series and tp_series:
            fe_chunks = chunk_mean(fe_series)
            tp_chunks = chunk_mean(tp_series)
            print(f"      [FE_rt ] {' '.join(f'{m:6.3f}' for m in fe_chunks)}   (Full Echo 발동 비율)")
            print(f"      [Point ] {' '.join(f'{m:6.2f}' for m in tp_chunks)}   (노드별 평균 포인트)")


# -------------------------------------------------------------------------
# 메인 실험: 5 개 부하 × (2 개 baseline + 5 개 b_max sweep) 그래프
# -------------------------------------------------------------------------
def run_bmax_sweep(md_file):
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(45, 8), squeeze=False)
    axes = axes.flatten()  # EXPERIMENTS 가 1 개여도 1D 배열로 유지
    fig.suptitle(
        f"6x6 Grid — PFE_c_AdE b_max sweep (c={C}, L={L})  "
        f"b_max ∈ {BMAX_VALUES}"
    )

    md_file.write(f"# PFE_c_AdE b_max sweep (6x6 Grid, c={C}, L={L})\n\n")
    md_file.write(f"b_max values: {BMAX_VALUES}\n\n")

    for ax, exp in zip(axes, EXPERIMENTS):
        lam = exp['lam']
        total_ticks = exp['total_ticks']
        x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

        md_file.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
        md_file.write("| algo | b_max | generated | delivered | undelivered | delivery_rate |\n")
        md_file.write("|------|-------|-----------|-----------|-------------|---------------|\n")

        print(f"\n=== λ={lam} ===")

        # ---- Baseline 알고리즘 (점선으로 표시) ----
        for algo, label, color in BASELINE_ALGOS:
            random.seed(SEED)
            np.random.seed(SEED)
            print(f"  Running {label}...")

            sim = Simulator(algorithm=algo, params=BASE_PARAMS, seed=SEED, topology=TOPOLOGY_GRID)
            adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    {label:22s} generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            md_file.write(f"| {label} | - | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

            print_diagnostics(sim, total_ticks, is_pfe=False)
            ax.plot(x_axis, adt, label=label, color=color, linestyle='--', alpha=0.7)

        # ---- PFE_c with b_max sweep (비활성화) ----
        # for b_max, color in zip(BMAX_VALUES, BMAX_COLORS):
        #     params = {**BASE_PARAMS, 'pfe_b_max': b_max}
        #     label = f"PFE_c b_max={b_max}"

        #     random.seed(SEED)
        #     np.random.seed(SEED)
        #     print(f"  Running {label}...")

        #     sim = Simulator(algorithm='pfe_c', params=params, seed=SEED, topology=TOPOLOGY_GRID)
        #     adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

        #     gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
        #     rate = (dlv / gen * 100) if gen > 0 else 0.0
        #     print(f"    {label:22s} generated={gen:6d}  delivered={dlv:6d}  "
        #           f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
        #     md_file.write(f"| PFE_c | {b_max} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

        #     print_diagnostics(sim, total_ticks, is_pfe=True)
        #     ax.plot(x_axis, adt, label=label, color=color)

        # ---- PFE_c_AdE with b_max sweep (실선) ----
        for b_max, color in zip(BMAX_VALUES, BMAX_COLORS):
            params = {**BASE_PARAMS, 'pfe_b_max': b_max}
            label = f"PFE_c_AdE b_max={b_max}"

            random.seed(SEED)
            np.random.seed(SEED)
            print(f"  Running {label}...")

            sim = Simulator(algorithm='pfe_c_ade', params=params, seed=SEED, topology=TOPOLOGY_GRID)
            adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    {label:22s} generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            md_file.write(f"| PFE_c_AdE | {b_max} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

            print_diagnostics(sim, total_ticks, is_pfe=True)
            ax.plot(x_axis, adt, label=label, color=color, linestyle='--')

        md_file.write("\n")
        ax.set_title(exp['title'])
        ax.set_xlabel('Simulator Time')
        ax.set_ylabel('Average Delivery Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "result_pfe_bmax_6by6.png"
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        run_bmax_sweep(md)
    print(f"로그: {MD_PATH}")
