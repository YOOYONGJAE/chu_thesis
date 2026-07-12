# =============================================================================
# [요약] 범용 알고리즘 비교 허브 — 6x6 grid, 10 시드 sweep, median + IQR 그래프
# - 알고리즘 풀 전체 (Q-routing / AQFE / AQRERM 계열 / PFE 계열 / FE) 를 보유,
#   ALGORITHMS 리스트의 주석 토글로 원하는 조합만 골라 실행
# - 계보 비교, L 유무 ablation, c/pre 속성 이식 실험 등
#   result_compare 폴더의 대부분 비교 md/png 가 이 스크립트 산출물
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
C   = 0.22   # pfe_c_*_echo_tick / aqrerm_c 등에서 사용하는 큐 페널티 가중치
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# 비교 대상 알고리즘 풀 (각 한 줄 요약)
# 차원: 선택 정보 (stale Q / fresh t) · c 페널티 (있음/없음) · echo 전략 (PFE 포인트 게이트 / 확률적 / always full) · Route Memory (L=3 / L=0)
#
# === AQRERM family (확률적 echo) ===
# - aqrerm                   : baseline. stale Q 선택, 확률 p 로 부분 echo. c 없음, L=3
# - aqrerm_no_L              : aqrerm 와 동일하되 L=0 (Route Memory 완전 비활성)
# - aqrerm_c                 : stale Q + c·queue 선택, 확률적 echo (구 AQLRERM)
# - aqrerm_pre              : AQRERM pre-echo. 랜덤 echo 먼저 (이전 y* 보장) → fresh t 로 선택, c 없음
# - aqrerm_c_pre            : AQRERM_c pre-echo. 랜덤 echo 먼저 → fresh t + c·queue 로 선택
#
# === PFE family (포인트 게이트 풀에코) ===
# - aqpace_no_pre_no_queue            : select-first (stale Q) + PFE 포인트 게이트, c 없음
# - aqpace_no_pre          : select-first (stale Q + c·queue) + PFE 포인트 게이트
# - aqpace_no_queue        : echo-first (fresh t) + PFE 포인트 게이트, c 없음
# - aqpace      : echo-first (fresh t + c·queue) + PFE 포인트 게이트 (★ 전부 적용)
# - aqpace_no_L : aqpace 와 동일하되 L=0
#
# === Full Echo family (게이트 없음) ===
# - aqpace_no_point            : 매 라우팅 무조건 full echo, fresh t + c·queue 로 선택 (echo 비용 상한선 기준)
ALGORITHMS = [
    # 'q_routing',
    # 'aqfe',
    'aqrerm',
    # 'aqrerm_no_L',
    'aqrerm_pre',
    'aqrerm_c',
    'aqrerm_c_pre',
    # 'aqpace_no_pre_no_queue',
    # 'aqpace_no_pre',
    # 'aqpace_no_queue',
    'aqpace', # ★ 메인 포커스: 큐 항 (c · queue) 추가한 PFE 변형
    # 'aqpace_no_point',
    # 'aqpace_no_L',
]
LABELS = {
    'q_routing': 'Q-routing', 
    'aqfe': 'AQFE',
    'aqpace_no_pre_no_queue':            'AQPACE(-pre,-queue)',
    'aqpace_no_queue':        'AQPACE(-queue)',
    'aqrerm_c':                  'AQRERM_c',
    'aqrerm':                   'AQRERM',
    'aqrerm_no_L':              'AQRERM_no_L',
    'aqpace':      'AQPACE',
    'aqrerm_c_pre':             'AQRERM_c_pre_RERM',
    'aqrerm_pre':              'AQRERM_pre',
    'aqpace_no_point':            'AQPACE(-point)',
    'aqpace_no_pre':          'AQPACE(-pre)',
    'aqpace_no_L': 'AQPACE(-L)',
}
# 적녹색약 친화 (Wong palette)
COLORS = {
    'q_routing':                "#00FF73", 
    'aqfe':                     "#FFBCBC",
    'aqpace_no_pre_no_queue':            '#0072B2',  # 파랑
    'aqpace_no_queue':        "#DAA32D",  # 주황
    'aqrerm_c':                  "#0011FF",  # 검정
    'aqrerm':                   '#CC79A7',  # 분홍보라
    'aqrerm_no_L':              '#882255',  # 진한 자주 (AQRERM family, 항상 no L)
    'aqpace_no_pre':          '#D55E00',  # 주홍 (vermillion)
    'aqpace':      '#56B4E9',  # 하늘색
    'aqrerm_c_pre':             "#E4D611",  # 노랑
    'aqrerm_pre':              '#117733',  # 진녹 (AQRERM pre-echo, c 없음)
    'aqpace_no_point':            '#000000',  # 검정 (always-FE 강조)
    'aqpace_no_L': "#3700FF",  # 회색 (L=0, no Route Memory)
}

# 10 개 시드 × 5 알고리즘 × 부하별 반복
SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]

STAT_INTERVAL = 100
MD_PATH = 'result_compare_PFE.md'

EXPERIMENTS = [
    # {'lam': 1, 'total_ticks': 14000, 'title': 'λ=1'},
    # {'lam': 2, 'total_ticks': 40000, 'title': 'λ=2'},
    # {'lam': 2.5, 'total_ticks': 40000, 'title': 'λ=2.5'},
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
# 메인: 각 부하별로 5 개 알고리즘을 10 시드씩 돌려 median + IQR 시각화
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 15), squeeze=False)
    axes = axes.flatten()  # EXPERIMENTS 가 1 개여도 1D 배열로 유지
    active_labels = ' / '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"6x6 Grid — PFE 변형 comparison ({active_labels}) "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)}, median + IQR band)"
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# 6x6 Grid PFE 변형 comparison (seed sweep)\n\n')
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
    filename = 'result_compare_PFE.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()

    print(f"\n로그: {MD_PATH}")


if __name__ == '__main__':
    run_all()
