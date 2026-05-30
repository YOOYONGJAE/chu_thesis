import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_nsfnet import NUM_NODES as NSFNET_NUM_NODES, ADJACENCY as NSFNET_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 설정
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22   # pfe_c_pre_echo_tick 의 큐 페널티 가중치 (다른 알고리즘은 이 키를 안 읽음)
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_NSFNET = {'num_nodes': NSFNET_NUM_NODES, 'adjacency': NSFNET_ADJACENCY}

# 비교 대상 알고리즘 (활성 리스트만 실제 실행, 나머지는 LABELS/COLORS 매핑만 유지)
ALGORITHMS = ['aqrerm', 'pfe_c_pre_echo_tick']
LABELS = {
    # 'q_routing':           'Q-routing',
    # 'aqfe':                'AQFE',
    'aqrerm':              'AQRERM',
    'pfe_c_pre_echo_tick': 'PFE_c_PreEcho_Tick',
}
# 적녹색약 친화 (Wong palette)
COLORS = {
    'q_routing':           '#0072B2',  # 파랑
    'aqfe':                '#E69F00',  # 주황
    'aqrerm':              '#009E73',  # 청록
    'pfe_c_pre_echo_tick': '#D55E00',  # 주홍 (vermillion)
}

# 10 개 시드 × 알고리즘 × 부하별 반복
SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]

STAT_INTERVAL = 100
MD_PATH = 'result_compare_algos_nsfnet.md'

EXPERIMENTS = [
    # {'lam': 1, 'total_ticks': 14000, 'title': 'λ=1'},
    {'lam': 2, 'total_ticks': 40000, 'title': 'λ=2'},
    {'lam': 3, 'total_ticks': 40000, 'title': 'λ=3'},
    {'lam': 3.5, 'total_ticks': 40000, 'title': 'λ=3.5'},
    {'lam': 3.7, 'total_ticks': 40000, 'title': 'λ=3.7'},
    {'lam': 3.8, 'total_ticks': 40000, 'title': 'λ=3.8'},
    # {'lam': 3.9, 'total_ticks': 40000, 'title': 'λ=3.9'},
    # {'lam': 4.0, 'total_ticks': 40000, 'title': 'λ=4.0'},
]


# -------------------------------------------------------------------------
# 단일 (algo, lam, seed) 실행 헬퍼
# -------------------------------------------------------------------------
def run_one(algo, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    sim = Simulator(algorithm=algo, params=BASE_PARAMS, seed=seed, topology=TOPOLOGY_NSFNET)
    adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
    return sim, adt


# -------------------------------------------------------------------------
# 메인: 각 부하별로 알고리즘들을 10 시드씩 돌려 median + IQR 시각화
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 10), squeeze=False)
    axes = axes.flatten()  # EXPERIMENTS 가 1 개여도 1D 배열로 유지
    active_labels = ' / '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"NSFNET — {active_labels} comparison "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)}, median + IQR band)"
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# NSFNET Algorithm comparison (seed sweep)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in ALGORITHMS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            lam = exp['lam']
            total_ticks = exp['total_ticks']
            x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

            md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
            print(f"\n========== λ={lam} ==========")

            for algo in ALGORITHMS:
                label = LABELS[algo]
                color = COLORS[algo]
                print(f"\n--- {label} ---")

                md.write(f"### {label}\n\n")
                md.write("| seed | generated | delivered | undelivered | delivery_rate |\n")
                md.write("|------|-----------|-----------|-------------|---------------|\n")

                adt_runs = []
                for seed in SEEDS:
                    print(f"  Running seed={seed}...")
                    sim, adt = run_one(algo, lam, total_ticks, seed)
                    gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                    rate = (dlv / gen * 100) if gen > 0 else 0.0
                    print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                          f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                    md.write(f"| {seed} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")
                    adt_runs.append(adt)

                # ---- 시드별 시리즈를 (n_seeds, n_windows) 배열로 쌓고 백분위수 계산 ----
                adt_arr = np.array(adt_runs)
                median = np.median(adt_arr, axis=0)
                q25 = np.percentile(adt_arr, 25, axis=0)
                q75 = np.percentile(adt_arr, 75, axis=0)

                # ---- median 굵은 실선 + IQR 음영 ----
                ax.plot(x_axis, median, label=label, color=color, linewidth=2.0)
                ax.fill_between(x_axis, q25, q75, color=color, alpha=0.2)

                # ---- steady-state 요약 (마지막 절반 평균 → 시드별로 → median/IQR) ----
                half = adt_arr.shape[1] // 2
                ss_per_seed = np.mean(adt_arr[:, half:], axis=1)
                ss_median = float(np.median(ss_per_seed))
                ss_q25 = float(np.percentile(ss_per_seed, 25))
                ss_q75 = float(np.percentile(ss_per_seed, 75))
                ss_min  = float(np.min(ss_per_seed))
                ss_max  = float(np.max(ss_per_seed))
                print(f"    [SS ADT] median={ss_median:6.2f}  "
                      f"IQR=[{ss_q25:6.2f}, {ss_q75:6.2f}]  "
                      f"range=[{ss_min:6.2f}, {ss_max:6.2f}]")
                md.write(f"\n**Steady-state ADT (last half mean per seed)** "
                         f"median={ss_median:.2f}  "
                         f"IQR=[{ss_q25:.2f}, {ss_q75:.2f}]  "
                         f"range=[{ss_min:.2f}, {ss_max:.2f}]\n\n")

            ax.set_title(exp['title'])
            ax.set_xlabel('Simulator Time')
            ax.set_ylabel('Average Delivery Time')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'result_compare_algos_nsfnet.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()

    print(f"\n로그: {MD_PATH}")


if __name__ == '__main__':
    run_all()
