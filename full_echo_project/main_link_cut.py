# =============================================================================
# [링크 절단 시나리오 스크립트 — 이번 연구 마감 범위 제외]
#
# 이 파일은 연구 진행 중 링크 장애 발생 시 Q-routing / AQFE / AQRERM / AQPRICE
# 4종 알고리즘이 얼마나 빠르게 대체 경로를 학습하는지 확인하기 위해 작성되었습니다.
#
# 이번 논문 마감 범위에는 해당 실험을 포함하지 않기로 결정하였으므로, 파일 하단의
# 실행 블록(if __name__ == '__main__')을 주석 처리하여 실수로 실행되지 않도록
# 비활성화해 두었습니다.
#
# 이후 연구에서 재활용이 필요하면 하단 실행 블록 주석을 해제하고,
# simulator.py 내 링크 절단 관련 주석 처리 구간도 함께 해제하십시오.
#
# [요약] 링크 절단 회복력 비교 — 4종 알고리즘, tick 4000 에서 링크 절단
# - 6x6 grid, 10 시드, 시나리오 2개: top bridge (14,15) / bottom bridge (2,3) 절단
# - Pre-cut / Post-cut ADT 를 분리 집계 → 토폴로지 변화 적응력 측정
# - 산출물: result_compare_AQPRICE_link_cut.md / .png
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트 (그래프 한글 깨짐 방지)
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 설정
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22   # aqprice 의 큐 페널티 가중치 (다른 알고리즘은 이 키를 안 읽음)
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# 비교 대상 알고리즘 (4종, main_compare_PFE.py 와 동일)
ALGORITHMS = [
    'q_routing',
    'aqfe',
    'aqrerm',
    'aqprice',
]
LABELS = {
    'q_routing': 'Q-routing',
    'aqfe':      'AQFE',
    'aqrerm':    'AQRERM',
    'aqprice':    'AQPRICE',
}
# 적녹색약 친화 (Wong palette)
COLORS = {
    'q_routing': '#117733',  # 진녹 (baseline 최단순)
    'aqfe':      '#44AA99',  # teal (AQRERM 의 부모)
    'aqrerm':    '#CC79A7',  # 분홍보라 (baseline)
    'aqprice':    '#56B4E9',  # 하늘색 (제안 기법)
}

# 10 개 시드 × 알고리즘 × (절단 시나리오 × 부하) 별 반복
SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]

STAT_INTERVAL = 100
MD_PATH = 'result_compare_AQPRICE_link_cut.md'

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
    sim = Simulator(algorithm=algo, params=BASE_PARAMS, seed=seed, topology=TOPOLOGY_GRID)
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
                    # 전달 패킷이 0인 통계 구간은 simulator 가 NaN 을 넣으므로,
                    # 그 시드 자리만 제외하고 나머지 시드로 집계하는 nan 무시 함수를 사용
                    adt_arr = np.array(adt_runs)
                    median = np.nanmedian(adt_arr, axis=0)
                    q25 = np.nanpercentile(adt_arr, 25, axis=0)
                    q75 = np.nanpercentile(adt_arr, 75, axis=0)

                    # ---- median 굵은 실선 + IQR 음영 ----
                    ax.plot(x_axis, median, label=label, color=color, linewidth=2.0)
                    ax.fill_between(x_axis, q25, q75, color=color, alpha=0.2)

                    # ---- 절단 전/후 SS ADT 요약 ----
                    cut_window = CUT_TICK // STAT_INTERVAL
                    pre  = adt_arr[:, :cut_window]
                    post = adt_arr[:, cut_window:]
                    ss_pre  = np.nanmean(pre, axis=1)  if pre.shape[1]  > 0 else np.array([0.0])
                    ss_post = np.nanmean(post, axis=1) if post.shape[1] > 0 else np.array([0.0])
                    print(f"    [SS pre ] median={float(np.nanmedian(ss_pre)):6.2f}  "
                          f"IQR=[{float(np.nanpercentile(ss_pre, 25)):6.2f}, "
                          f"{float(np.nanpercentile(ss_pre, 75)):6.2f}]")
                    print(f"    [SS post] median={float(np.nanmedian(ss_post)):6.2f}  "
                          f"IQR=[{float(np.nanpercentile(ss_post, 25)):6.2f}, "
                          f"{float(np.nanpercentile(ss_post, 75)):6.2f}]")
                    md.write(f"\n**Pre-cut ADT** median={float(np.nanmedian(ss_pre)):.2f}  "
                             f"IQR=[{float(np.nanpercentile(ss_pre, 25)):.2f}, "
                             f"{float(np.nanpercentile(ss_pre, 75)):.2f}]  \n")
                    md.write(f"**Post-cut ADT** median={float(np.nanmedian(ss_post)):.2f}  "
                             f"IQR=[{float(np.nanpercentile(ss_post, 25)):.2f}, "
                             f"{float(np.nanpercentile(ss_post, 75)):.2f}]\n\n")

                # ---- 절단 시점 수직선 + 라벨 ----
                ax.axvline(x=CUT_TICK, color='red', linestyle='--', linewidth=1.5, label='Link cut')
                ax.set_title(f"λ={lam} — Cut {scenario['name']}")
                ax.set_xlabel('Simulator Time')
                ax.set_ylabel('Average Delivery Time')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'result_compare_AQPRICE_link_cut.png'
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()

    print(f"\n로그: {MD_PATH}")


# =====================================================================
# [실행 블록 — 이번 연구 마감 범위 제외]
# 링크 절단 시나리오 실험을 비활성화합니다. 재활성화 시 아래 주석을 해제하고
# simulator.py 내 링크 절단 관련 주석 처리 구간도 함께 해제하십시오.
# =====================================================================
# if __name__ == '__main__':
#     run_all()
