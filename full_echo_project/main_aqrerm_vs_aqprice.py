# =============================================================================
# [요약] AQRERM vs AQPRICE 정량 지표 비교
# - 6x6 grid, 10 시드. 핵심 지표 4개 + 패킷 전달 통계
# - 지표:
# ① 전체 평균 ADT — 그래프 중앙값 실선을 실행 전체 구간에 걸쳐 시간 평균
# ② 전체 IQR 폭 — 그래프 IQR 띠(q75−q25)의 전체 구간 평균 폭 (작을수록 시드 간 일관)
# ③ 라우팅 결정당 이웃 조회 수 (이웃 조회 총횟수 / 라우팅 결정 총횟수, 성능 대 비용)
# ④ 경로 변경률 (next-hop 변경 총횟수 / 라우팅 결정 총횟수)
# - 구간을 나누지 않고 실행 전체를 요약 (정착 임계값 지정 없음)
# - 산출물: result_aqrerm_vs_aqprice.md / .png
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트 (그래프 한글 깨짐 방지)
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from simulator import Simulator
import diagnostics as diag
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_sub_test_grid_full import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_nsfnet import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_sub_test_pq import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (main_compare_AQPRICE 와 동일)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
# C   = 0
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}


def topo_name(n):
    """토폴로지 이름 자동 판별. 노드 수가 같은 격자 두 종은 링크 수로 가른다.
    (import 한 줄만 바꾸면 제목과 파일명이 함께 따라온다.)"""
    return diag.topo_name(n, GRID_ADJACENCY)

# 비교 대상 2 개
ALGORITHMS = ['aqrerm', 'aqprice']
LABELS = {
    'aqrerm': 'AQRERM',
    'aqprice': 'AQPRICE',
}
COLORS = {
    'aqrerm':              "#FF0000",  # 분홍보라 (baseline)
    'aqprice': '#56B4E9',  # 하늘색 (메인 후보)
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100
TOTAL_TICKS = 20000

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = diag.topo_tag(GRID_NUM_NODES, GRID_ADJACENCY)
# RESULT_TAG = 'LOW'   # 부하 표기 (수동): LOW / HIGH 등
# RESULT_TAG = 'HIGH'   # 부하 표기 (수동): LOW / HIGH 등
RESULT_TAG = '3LVL'   # 부하 표기 (수동): LOW / HIGH 등
_BASENAME  = f'result_aqrerm_vs_aqprice({TOPO_TAG},{RESULT_TAG},{TOTAL_TICKS})'
MD_PATH       = f'{_BASENAME}.md'
PNG_PATH      = f'{_BASENAME}.png'
LINK_PNG_PATH = f'{_BASENAME}_linkuse.png'

EXPERIMENTS = [
    # 4 단계 부하 — 저 / 중저 / 중고 / 고. 모든 실험 공통.
    # 고부하 상한을 3.8 로 둔 이유 : AQRERM 은 발산하고 AQPRICE 는 정착하는 경계 지점.
    # {'lam': 2.5, 'total_ticks': 20000},   # 저부하
    {'lam': 3.0, 'total_ticks': 20000},   # 중저부하
    # {'lam': 3.2, 'total_ticks': 20000},   # 중저부하
    {'lam': 3.5, 'total_ticks': 20000},   # 중고부하
    {'lam': 3.8, 'total_ticks': 20000},   # 고부하
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
# Metric 계산 함수들
# - 정착 임계값을 정당화할 필요가 없도록 구간을 나누지 않고 실행 전체를 요약
# - 표의 두 숫자는 ADT 그래프의 실선(중앙값)·띠(IQR)를 전체 구간 평균낸 값과 같다
# -------------------------------------------------------------------------
def compute_metrics(adt_arr):
    """전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱) + 전체 IQR 폭 (띠 평균 폭 // 단위 = 틱) 을 dict 로 반환."""
    med  = np.nanmedian(adt_arr, axis=0)
    band = (np.nanpercentile(adt_arr, 75, axis=0)
            - np.nanpercentile(adt_arr, 25, axis=0))
    return {
        'overall':     float(np.nanmean(med)),
        'overall_iqr': float(np.nanmean(band)),
    }


# -------------------------------------------------------------------------
# 한 부하 실험 (한 패널)
# -------------------------------------------------------------------------
def run_lambda(ax, lam, total_ticks, md):
    x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

    md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
    print(f"\n========== λ={lam} ==========")

    # ---- 알고리즘별 시뮬레이션 수집 ----
    results = {}
    for algo in ALGORITHMS:
        label = LABELS[algo]
        print(f"\n--- {label} ---")
        adt_runs = []
        echo_costs = []
        echo_totals, route_totals = [], []   # 나누기 전 총횟수 (시드별, 36개 노드 합)
        switch_rates = []
        rx_runs, echo_runs, switch_runs, link_dicts = [], [], [], []
        te_runs, tm_runs = [], []          # R_x 의 분자·분모 시계열
        gens, dlvs, unds, rates = [], [], [], []
        for seed in SEEDS:
            print(f"  Running seed={seed}...")
            sim, adt = run_one(algo, lam, total_ticks, seed)
            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            adt_runs.append(adt)
            gens.append(gen); dlvs.append(dlv); unds.append(und); rates.append(rate)
            ec = (sim.total_echo_queries / sim.total_route_calls
                  if sim.total_route_calls > 0 else 0.0)
            echo_costs.append(ec)
            echo_totals.append(sim.total_echo_queries)
            route_totals.append(sim.total_route_calls)
            sw =(sim.total_switches / sim.total_route_calls
                  if sim.total_route_calls > 0 else 0.0)
            switch_rates.append(sw)
            # 진단 시계열·링크 사용량 수집
            rx_runs.append(sim.r_ratio_series)
            te_runs.append(sim.t_est_series)
            tm_runs.append(sim.t_max_series)
            echo_runs.append(sim.echo_per_decision_series)
            switch_runs.append(sim.switch_rate_series)
            link_dicts.append(sim.link_usage)
        adt_arr = np.array(adt_runs)
        results[algo] = {
            'adt_arr':   adt_arr,
            'median':    np.nanmedian(adt_arr, axis=0),
            'q25':       np.nanpercentile(adt_arr, 25, axis=0),
            'q75':       np.nanpercentile(adt_arr, 75, axis=0),
            'echo_cost':   float(np.mean(echo_costs)),          # 진단 시계열 요약용 스칼라. 시드 10개 값의 평균
            'echo_total':  float(np.mean(echo_totals)),         # 이웃 조회 총횟수, 시드 10개 평균
            'route_total': float(np.mean(route_totals)),        # 라우팅 결정 총횟수, 시드 10개 평균
            'switch_rate': float(np.mean(switch_rates)),
            'rx_med':     diag.median_series(rx_runs),
            't_est_med':  diag.median_series(te_runs),
            't_max_med':  diag.median_series(tm_runs),
            'echo_med':   diag.median_series(echo_runs),
            'switch_med': diag.median_series(switch_runs),
            'link_sum':   diag.sum_link_usage(link_dicts),
            'gen_mean':  float(np.mean(gens)),
            'dlv_mean':  float(np.mean(dlvs)),
            'und_mean':  float(np.mean(unds)),
            'rate_mean': float(np.mean(rates)),
            'rate_max':  float(np.max(rates)),
            'rate_min':  float(np.min(rates)),
        }

    # ---- 알고리즘별 ADT 지표 (전체 평균 ADT, 전체 IQR 폭). 에코·변경률은 진단 시계열로 ----
    metrics = {a: compute_metrics(results[a]['adt_arr']) for a in ALGORITHMS}

    # ---- 시각화 : median 실선 + IQR 오차 막대 ----
    spacing = max(1, len(x_axis) // 10)
    for idx, algo in enumerate(ALGORITHMS):
        median = results[algo]['median']
        yerr = np.vstack([median - results[algo]['q25'],
                          results[algo]['q75'] - median])
        offset = idx * spacing // len(ALGORITHMS)
        ax.errorbar(x_axis, median, yerr=yerr,
                    errorevery=(offset, spacing),
                    capsize=3, elinewidth=1.2, capthick=1.2,
                    label=LABELS[algo], color=COLORS[algo], linewidth=2.0)

    ax.set_title(f"λ={lam}", fontsize=20)
    ax.set_xlabel('Simulator Time', fontsize=18)
    ax.set_ylabel('Average Delivery Time', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc='upper right', fontsize=17)
    ax.grid(True, alpha=0.3)

    aqrerm_m  = metrics['aqrerm']
    aqprice_m = metrics['aqprice']
    adt_diff = aqprice_m['overall'] - aqrerm_m['overall']

    # 차이 열은 모두 AQPRICE−AQRERM 절대 차이. 음수면 AQPRICE 가 그 지표에서 더 작다.
    md.write("| 지표 | AQRERM | AQPRICE | 차이 (AQPRICE−AQRERM) |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| 전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱) | {aqrerm_m['overall']:.2f} | {aqprice_m['overall']:.2f} | {adt_diff:+.2f} |\n")
    md.write(f"| 전체 IQR 폭 (띠 평균 폭, 작을수록 일관 // 단위 = 틱) | {aqrerm_m['overall_iqr']:.2f} | {aqprice_m['overall_iqr']:.2f} | {aqprice_m['overall_iqr'] - aqrerm_m['overall_iqr']:+.2f} |\n")
    # 진단 시계열 전체 평균 중 비교 지표(에코·변경률)를 위 표에도 함께. R_x 는 시계열에만.
    ar, pr = results['aqrerm'], results['aqprice']
    md.write(f"| 라우팅 결정당 이웃 조회 수 (전체 평균 // 단위 = 개/결정) | {ar['echo_cost']:.2f} | {pr['echo_cost']:.2f} | {pr['echo_cost'] - ar['echo_cost']:+.2f} |\n")
    # 위 비율의 분자·분모. 시드마다 20000 tick 동안 36개 노드 합을 구하고 시드 10개를 평균.
    md.write(f"| 이웃 조회 총횟수 (시드 10개 평균 // 단위 = 회) | {ar['echo_total']:,.0f} | {pr['echo_total']:,.0f} | {pr['echo_total'] - ar['echo_total']:+,.0f} |\n")
    md.write(f"| 라우팅 결정 총횟수 (시드 10개 평균 // 단위 = 회) | {ar['route_total']:,.0f} | {pr['route_total']:,.0f} | {pr['route_total'] - ar['route_total']:+,.0f} |\n")
    # 경로 변경률은 비율이라 백분율로, 차이는 두 백분율의 차이이므로 %p 로 적는다.
    md.write(f"| 경로 변경률 (전체 평균 // 단위 = %) | {ar['switch_rate'] * 100:.1f}% | {pr['switch_rate'] * 100:.1f}% | "
             f"{(pr['switch_rate'] - ar['switch_rate']) * 100:+.1f}%p |\n")
    # R_x 는 진단 시계열의 전체구간 평균 (아래 시계열 표의 '전체 평균' 과 같은 값)
    ar_rx, pr_rx = diag.series_overall(ar['rx_med']), diag.series_overall(pr['rx_med'])
    md.write(f"| R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율) | {ar_rx:.3f} | {pr_rx:.3f} | {pr_rx - ar_rx:+.3f} |\n")
    # R_x 의 분자·분모. 둘 다 전체구간 시간평균 (R_x 와 같은 요약 규칙).
    ar_te, pr_te = diag.series_overall(ar['t_est_med']), diag.series_overall(pr['t_est_med'])
    ar_tm, pr_tm = diag.series_overall(ar['t_max_med']),   diag.series_overall(pr['t_max_med'])
    md.write(f"| T_est (전체 평균, R_x 분자 // 단위 = 틱) | {ar_te:.2f} | {pr_te:.2f} | {pr_te - ar_te:+.2f} |\n")
    md.write(f"| T_max (전체 평균, R_x 분모 // 단위 = 틱) | {ar_tm:.2f} | {pr_tm:.2f} | {pr_tm - ar_tm:+.2f} |\n\n")

    # ---- MD 진단 시계열 (알고리즘별 구간 평균 + 전체 평균) ----
    for algo in ALGORITHMS:
        r = results[algo]
        diag.write_diag_series_md(md, LABELS[algo],
                                  r['rx_med'], r['echo_med'], r['switch_med'],
                                  r['echo_cost'], r['switch_rate'],
                                  t_est_med=r['t_est_med'], t_max_med=r['t_max_med'],
                                  stat_interval=STAT_INTERVAL)

    # ---- MD: 패킷 전달 통계 ----
    aqrerm_r  = results['aqrerm']
    aqprice_r = results['aqprice']
    md.write(f"#### 패킷 전달 통계 ({len(SEEDS)} 시드)\n\n")
    md.write("| 항목 | AQRERM | AQPRICE |\n")
    md.write("|---|---:|---:|\n")
    md.write(f"| 생성 (평균) | {aqrerm_r['gen_mean']:.0f} | {aqprice_r['gen_mean']:.0f} |\n")
    md.write(f"| 전달 (평균) | {aqrerm_r['dlv_mean']:.0f} | {aqprice_r['dlv_mean']:.0f} |\n")
    md.write(f"| 미전달 (평균) | {aqrerm_r['und_mean']:.0f} | {aqprice_r['und_mean']:.0f} |\n")
    md.write(f"| 성공률 평균 | {aqrerm_r['rate_mean']:.1f}% | {aqprice_r['rate_mean']:.1f}% |\n")
    md.write(f"| 성공률 최고 | {aqrerm_r['rate_max']:.1f}% | {aqprice_r['rate_max']:.1f}% |\n")
    md.write(f"| 성공률 최저 | {aqrerm_r['rate_min']:.1f}% | {aqprice_r['rate_min']:.1f}% |\n\n")

    # ---- MD 링크 사용량 수치 (알고리즘별, 통과량 순) ----
    for algo in ALGORITHMS:
        md.write(f"#### 링크 사용량 ({LABELS[algo]}, 통과량 순, 시드 합)\n\n")
        md.write(diag.link_usage_md(results[algo]['link_sum']))
        md.write("\n")

    # ---- 콘솔 출력 ----
    print(f"  {'지표':<24} | {'AQRERM':>12} | {'AQPRICE':>12} | {'차이':>12}")
    print(f"  {'전체 평균 ADT':<24} | {aqrerm_m['overall']:>12.2f} | {aqprice_m['overall']:>12.2f} | {adt_diff:>+11.2f}")
    print(f"  {'전체 IQR 폭':<24} | {aqrerm_m['overall_iqr']:>12.2f} | {aqprice_m['overall_iqr']:>12.2f} | "
          f"{aqprice_m['overall_iqr'] - aqrerm_m['overall_iqr']:>+11.2f} ")

    # 링크 다이어그램용: 알고리즘별 시드 합산 link_usage 반환
    return {a: results[a]['link_sum'] for a in ALGORITHMS}


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    # 패널 배치 — 부하 4개면 2x2. 패널 하나당 크기가 고정이라 실험 수가 달라져도 비율 유지.
    fig, axes = diag.make_panel_grid(len(EXPERIMENTS))
    active_labels = ' vs '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — {active_labels}  "
        f"(c={C}, seeds n={len(SEEDS)}, {TOTAL_TICKS}tick)",
        fontsize=22,
        y=0.99,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write(f'# Final comparison : {active_labels}\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in ALGORITHMS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 출력 구성\n\n')
        md.write('- 성능 비교표 : 전체 평균 ADT(그래프 중앙값 실선의 시간평균)와 전체 IQR 폭(그래프 띠의 평균 폭, 작을수록 일관). 구간 분할 없이 실행 전체를 요약.\n')
        md.write('- 진단 시계열 : R_x / 라우팅 결정당 이웃 조회 수 / 경로 변경률 구간 평균 + 끝의 전체 평균(에코·변경률 요약).\n')
        md.write('- 링크 사용량 : 별도 PNG 에 grid 링크 굵기·색(적을수록 하늘, 많을수록 빨강) + md 통과량 표.\n\n')

        diag.write_metric_guide_md(md)

        link_data = []   # [(lam, {algo: link_sum}), ...]
        for ax, exp in zip(axes, EXPERIMENTS):
            per_algo = run_lambda(ax, exp['lam'], exp['total_ticks'], md)
            link_data.append((exp['lam'], per_algo))

    # rect 위쪽을 0.96 으로 잘라 suptitle 자리를 남긴다 (제목이 패널 제목과 겹치지 않게)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")

    # ---- 링크 사용량 다이어그램 (그리드: 행=알고리즘, 열=부하). grid 만 ----
    if GRID_NUM_NODES == 36:
        nrows, ncols = len(ALGORITHMS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                                   squeeze=False)
        figL.suptitle(
            f"{topo_name(GRID_NUM_NODES)} — link usage  [color & width = usage level]",
            fontsize=20)
        vmin, vmax = diag.link_usage_range([per[a] for (lam, per) in link_data for a in per])
        for r, algo in enumerate(ALGORITHMS):
            for c, (lam, per_algo) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_algo[algo],
                                         GRID_NUM_NODES, f"{LABELS[algo]}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
