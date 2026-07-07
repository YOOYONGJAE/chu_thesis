# =============================================================================
# [요약] 링크 절단 회복력 비교 — 6x6 grid, 10 시드, tick 4000 에서 링크 절단
# - 시나리오 2개: top bridge (14,15) 절단 / bottom bridge (2,3) 절단
# - Pre-cut / Post-cut ADT 를 분리 집계 → 토폴로지 변화 적응력 측정
# - AQRERM vs AQPACE 정면 비교 + 양쪽의 L 유무 ablation 에 사용
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

# 비교 대상 알고리즘 (main_compare_PFE.py 와 동일 풀, 활성 리스트만 실행)
ALGORITHMS = [
    'q_routing',
    'aqfe',
    'aqrerm',
    # 'aqrerm_c',
    # 'aqrerm_c_pre',
    # 'pfe_echo_tick',
    # 'pfe_c_echo_tick',
    # 'pfe_pre_echo_tick',
    'aqpace',
    # 'fe_c_pre_echo',
    # 'aqpace_no_L',
    # 'aqrerm_no_L',
    # 'aqrerm_4000_no_L',
]
LABELS = {
    'q_routing':                'Q-routing',
    'aqfe':                     'AQFE',
    'pfe_echo_tick':            'PFE_echo_tick',
    'pfe_pre_echo_tick':        'PFE_pre_echo_tick',
    'aqrerm_c':                  'AQRERM_c',
    'aqrerm':                   'AQRERM',
    'pfe_c_echo_tick':          'PFE_c_echo_tick',
    'aqpace':      'AQPACE',
    'aqrerm_c_pre':             'AQRERM_c_pre_RERM',
    'fe_c_pre_echo':            'FE_c_pre_echo',
    'aqpace_no_L': 'AQPACE_noL',
    'aqrerm_no_L':              'AQRERM_no_L',
    'aqrerm_4000_no_L':         'AQRERM_4000_no_L',
}
# 적녹색약 친화 (Wong palette)
COLORS = {
    'q_routing':                '#117733',  # 진녹 (baseline 최단순)
    'aqfe':                     '#44AA99',  # teal (AQRERM 의 부모)
    'pfe_echo_tick':            '#0072B2',  # 파랑
    'pfe_pre_echo_tick':        '#E69F00',  # 주황
    'aqrerm_c':                  '#009E73',  # 청록
    'aqrerm':                   '#CC79A7',  # 분홍보라
    'pfe_c_echo_tick':          '#D55E00',  # 주홍 (vermillion)
    'aqpace':      '#56B4E9',  # 하늘색
    'aqrerm_c_pre':             '#F0E442',  # 노랑
    'fe_c_pre_echo':            '#000000',  # 검정 (always-FE 강조)
    'aqpace_no_L': '#999999',  # 회색 (L=0, no Route Memory)
    'aqrerm_no_L':              '#882255',  # 진한 자주 (AQRERM family, 항상 no L)
    'aqrerm_4000_no_L':         "#0400FF",  # 진한 파랑 (AQRERM family, 4000 이후 no L)
}

# 10 개 시드 × 알고리즘 × (절단 시나리오 × 부하) 별 반복
SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]

STAT_INTERVAL = 100
MD_PATH = 'result_compare_PFE_link_cut.md'

# -------------------------------------------------------------------------
# 링크 절단 시나리오 (main_link_cut.py 기반)
# -------------------------------------------------------------------------
CUT_TICK = 4000
TOTAL_TICKS = 20000

CUT_SCENARIOS = [
    {'name': '(14,15) [top bridge]',    'cuts': [(CUT_TICK, 14, 15)]},
    {'name': '(2,3) [bottom bridge]',   'cuts': [(CUT_TICK, 2, 3)]},
]

# 부하 (절단 시나리오용 — 보통 절단 효과를 잘 보려고 중부하로 잡음)
LAMBDAS = [1.5, 2]
# LAMBDAS = [2.5, 3]


# -------------------------------------------------------------------------
# 단일 (algo, lam, seed, cuts) 실행 헬퍼
# -------------------------------------------------------------------------
def run_one(algo, lam, seed, cuts):
    random.seed(seed)
    np.random.seed(seed)
    # memory_cut_tick : _l0 변형들이 cut 후 L=0 강제하는 데 사용 (해당 알고리즘만 읽음)
    params = {**BASE_PARAMS, 'memory_cut_tick': CUT_TICK}
    sim = Simulator(algorithm=algo, params=params, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(
        lam=lam,
        total_ticks=TOTAL_TICKS,
        stat_interval=STAT_INTERVAL,
        link_cuts=cuts,
    )
    return sim, adt


# -------------------------------------------------------------------------
# 메인: 각 (scenario, lambda) 셀에 알고리즘들을 시드 sweep 후 median + IQR 시각화
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(
        len(CUT_SCENARIOS), len(LAMBDAS),
        figsize=(45, 20),
        squeeze=False,
    )
    active_labels = ' / '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"6x6 Grid Link Cut — PFE 변형 comparison ({active_labels}) "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)}, cut@tick={CUT_TICK}, median + IQR band)"
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# 6x6 Grid Link Cut + PFE 변형 comparison (seed sweep)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in ALGORITHMS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- CUT_TICK: {CUT_TICK}, TOTAL_TICKS: {TOTAL_TICKS}\n\n')

        x_axis = np.arange(1, TOTAL_TICKS // STAT_INTERVAL + 1) * STAT_INTERVAL

        for row, scenario in enumerate(CUT_SCENARIOS):
            md.write(f"## Cut {scenario['name']}\n\n")

            for col, lam in enumerate(LAMBDAS):
                ax = axes[row, col]
                print(f"\n========== Cut {scenario['name']}  λ={lam} ==========")

                md.write(f"### λ={lam} ({TOTAL_TICKS} ticks)\n\n")

                for algo in ALGORITHMS:
                    label = LABELS[algo]
                    color = COLORS[algo]
                    print(f"\n--- {label} ---")

                    md.write(f"#### {label}\n\n")
                    md.write("| seed | generated | delivered | undelivered | delivery_rate |\n")
                    md.write("|------|-----------|-----------|-------------|---------------|\n")

                    adt_runs = []
                    for seed in SEEDS:
                        print(f"  Running seed={seed}...")
                        sim, adt = run_one(algo, lam, seed, scenario['cuts'])
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

                    # ---- 절단 전/후 SS ADT 요약 ----
                    cut_window = CUT_TICK // STAT_INTERVAL
                    pre  = adt_arr[:, :cut_window]
                    post = adt_arr[:, cut_window:]
                    ss_pre  = np.mean(pre, axis=1)  if pre.shape[1]  > 0 else np.array([0.0])
                    ss_post = np.mean(post, axis=1) if post.shape[1] > 0 else np.array([0.0])
                    print(f"    [SS pre ] median={float(np.median(ss_pre)):6.2f}  "
                          f"IQR=[{float(np.percentile(ss_pre, 25)):6.2f}, "
                          f"{float(np.percentile(ss_pre, 75)):6.2f}]")
                    print(f"    [SS post] median={float(np.median(ss_post)):6.2f}  "
                          f"IQR=[{float(np.percentile(ss_post, 25)):6.2f}, "
                          f"{float(np.percentile(ss_post, 75)):6.2f}]")
                    md.write(f"\n**Pre-cut ADT** median={float(np.median(ss_pre)):.2f}  "
                             f"IQR=[{float(np.percentile(ss_pre, 25)):.2f}, "
                             f"{float(np.percentile(ss_pre, 75)):.2f}]  \n")
                    md.write(f"**Post-cut ADT** median={float(np.median(ss_post)):.2f}  "
                             f"IQR=[{float(np.percentile(ss_post, 25)):.2f}, "
                             f"{float(np.percentile(ss_post, 75)):.2f}]\n\n")

                # ---- 절단 시점 수직선 + 라벨 ----
                ax.axvline(x=CUT_TICK, color='red', linestyle='--', linewidth=1.5, label='Link cut')
                ax.set_title(f"λ={lam} — Cut {scenario['name']}")
                ax.set_xlabel('Simulator Time')
                ax.set_ylabel('Average Delivery Time')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'result_compare_PFE_link_cut.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()

    print(f"\n로그: {MD_PATH}")


if __name__ == '__main__':
    run_all()
