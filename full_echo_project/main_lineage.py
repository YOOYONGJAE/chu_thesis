# =============================================================================
# [요약] 계보 비교 허브 — Q-routing / AQFE / AQRERM / AQPACE 4종 비교
# - 6x6 grid, 10 시드 sweep, λ 별 ADT median + IQR 그래프 + markdown 결과표
# - ALGORITHMS 주석 토글로 원하는 조합만 실행 가능
# - 구세대 변형 포함 버전은 legacy_algorithm_files/main_compare_PFE.py 참조
# - 산출물: result_compare_AQPACE.md / .png
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 설정
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22   # aqpace 의 큐 페널티 가중치 (다른 알고리즘은 이 키를 안 읽음)
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 비교 대상 알고리즘 (4종)
# - q_routing : Boyan & Littman 1994. y* 만 단일 업데이트, echo 없음
# - aqfe      : Adaptive Q-routing with Full Echo. 매 라우팅 전 이웃 echo
# - aqrerm    : Random Echo + Route Memory. 확률 p = T_est/T_max 로 부분 echo
# - aqpace    : 포인트 예산 게이트 + pre-echo + c·queue 페널티 (제안 기법)
# -------------------------------------------------------------------------
ALGORITHMS = [
    'q_routing',
    'aqfe',
    'aqrerm',
    'aqpace',
]
LABELS = {
    'q_routing': 'Q-routing',
    'aqfe':      'AQFE',
    'aqrerm':    'AQRERM',
    'aqpace':    'AQPACE',
}
# 적녹색약 친화 (Wong palette)
COLORS = {
    'q_routing': '#117733',  # 진녹 (baseline 최단순)
    'aqfe':      '#44AA99',  # teal (AQRERM 의 부모)
    'aqrerm':    '#CC79A7',  # 분홍보라 (baseline)
    'aqpace':    '#56B4E9',  # 하늘색 (제안 기법)
}

# 10 개 시드 × 알고리즘 × 부하별 반복
SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]

STAT_INTERVAL = 100
MD_PATH = 'result_compare_AQPACE.md'

EXPERIMENTS = [
    # {'lam': 2, 'total_ticks': 40000, 'title': 'λ=2'},
    # {'lam': 3, 'total_ticks': 40000, 'title': 'λ=3'},
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
    sim = Simulator(algorithm=algo, params=BASE_PARAMS, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
    return sim, adt


# -------------------------------------------------------------------------
# 메인: 각 부하별로 알고리즘들을 10 시드씩 돌려 median + IQR 시각화
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 15), squeeze=False)
    axes = axes.flatten()  # EXPERIMENTS 가 1 개여도 1D 배열로 유지
    active_labels = ' / '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"6x6 Grid — Algorithm comparison ({active_labels}) "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)}, median + IQR band)"
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# 6x6 Grid Algorithm comparison (seed sweep)\n\n')
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
                # 전달 패킷이 0인 통계 구간은 simulator 가 NaN 을 넣으므로,
                # 그 시드 자리만 제외하고 나머지 시드로 집계하는 nan 무시 함수를 사용
                adt_arr = np.array(adt_runs)
                median = np.nanmedian(adt_arr, axis=0)
                q25 = np.nanpercentile(adt_arr, 25, axis=0)
                q75 = np.nanpercentile(adt_arr, 75, axis=0)

                # ---- median 실선 + IQR 오차 막대 (일정 간격 세로선) ----
                # yerr 는 (아래 길이, 위 길이) 두 행 — median 기준 비대칭 IQR 범위
                # offset : 알고리즘마다 시작점을 어긋내 막대가 같은 x 에 겹치지 않게 함
                yerr = np.vstack([median - q25, q75 - median])
                spacing = max(1, len(x_axis) // 10)
                offset = ALGORITHMS.index(algo) * spacing // len(ALGORITHMS)
                ax.errorbar(x_axis, median, yerr=yerr,
                            errorevery=(offset, spacing),
                            capsize=3, elinewidth=1.2, capthick=1.2,
                            label=label, color=color, linewidth=2.0)

                # ---- steady-state 요약 (마지막 절반 평균 → 시드별로 → median/IQR) ----
                # 여기도 NaN 구간을 제외하고 집계 (nan 무시 계열)
                half = adt_arr.shape[1] // 2
                ss_per_seed = np.nanmean(adt_arr[:, half:], axis=1)
                ss_median = float(np.nanmedian(ss_per_seed))
                ss_q25 = float(np.nanpercentile(ss_per_seed, 25))
                ss_q75 = float(np.nanpercentile(ss_per_seed, 75))
                ss_min  = float(np.nanmin(ss_per_seed))
                ss_max  = float(np.nanmax(ss_per_seed))
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
    filename = 'result_compare_AQPACE.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    run_all()
