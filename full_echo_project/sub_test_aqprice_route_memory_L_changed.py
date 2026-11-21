# =============================================================================
# [요약] AQPRICE 라우트 메모리 길이 비교 — L=0 / 1 / 2 / 3
# - 같은 aqprice 를 라우트 메모리 길이만 다르게 비교 (요청노드 제외는 네 변형 모두 유지=기본)
#   L=0 : 라우트 메모리 미사용 (AQPRICE 정식 기본)
#   L=1/2/3 : 최근 방문 노드를 각각 1/2/3 개까지 다음 홉 후보에서 제외
# - node.py 는 aqprice_L 로 이 길이를 제어 (기본 0)
# - 6x6 grid, 10 시드, λ=3.8 만 (40000 tick). 지표표 + 그래프 생산
# - 4 지표: 전체 평균 ADT / 전체 IQR 폭 / 라우팅 결정당 이웃 조회 수 / 경로 변경률 (구간 분할 없음)
# - 산출물: result_aqprice_route_memory_L_changed.md / .png
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

# -------------------------------------------------------------------------
# 파라미터 (aqprice_L 만 변형별로 덮어씀)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

def topo_name(n):
    """토폴로지 이름 자동 판별. 노드 수가 같은 격자 두 종은 링크 수로 가른다.
    (import 한 줄만 바꾸면 제목과 파일명이 함께 따라온다.)"""
    return diag.topo_name(n, GRID_ADJACENCY)


# -------------------------------------------------------------------------
# 4 변형: 같은 aqprice, 라우트 메모리 길이만 다름 (요청노드 제외는 모두 기본 True 유지)
# -------------------------------------------------------------------------
VARIANTS = ['L0', 'L3']
REF_KEY  = 'L0'   # AQPRICE 정식 기본. 기준선·표기 기준
VARIANT_L = {'L0': 0, 'L3': 3}
# 그래프 범례와 md 표에 그대로 나가는 이름. 논문 표기와 일치시킨다.
#   L0 = AQPRICE 정식 (라우트 메모리 미사용)
#   L3 = 라우트 메모리를 복원한 조건
LABELS = {
    'L0': 'AQPRICE',
    'L3': 'AQPRICE+L',
}
LABELS_SHORT = dict(LABELS)
COLORS = {
    'L0': '#56B4E9',  # 하늘색 (기본)
    # 'L1': '#E69F00',  # 주황
    # 'L2': '#009E73',  # 초록
    'L3': "#FF0000",  # 주홍
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100


# λ=3.8 만, 20000 tick
EXPERIMENTS = [
    # 4 단계 부하 — 저 / 중저 / 중고 / 고. 모든 실험 공통.
    # 고부하 상한을 3.8 로 둔 이유 : AQRERM 은 발산하고 AQPRICE 는 정착하는 경계 지점.
    {'lam': 2.5, 'total_ticks': 20000},   # 저부하
    {'lam': 3.0, 'total_ticks': 20000},   # 중저부하
    {'lam': 3.5, 'total_ticks': 20000},   # 중고부하
    {'lam': 3.8, 'total_ticks': 20000},   # 고부하
]

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = diag.topo_tag(GRID_NUM_NODES, GRID_ADJACENCY)
RESULT_TAG = 'LOW'   # 부하 표기 (수동): LOW / HIGH 등
# RESULT_TAG = 'HIGH'   # 부하 표기 (수동): LOW / HIGH 등
_BASENAME  = f'result_aqprice_route_memory_L_changed({TOPO_TAG},{RESULT_TAG},{EXPERIMENTS[0]["total_ticks"]})'
MD_PATH       = f'{_BASENAME}.md'
PNG_PATH      = f'{_BASENAME}.png'
LINK_PNG_PATH = f'{_BASENAME}_linkuse.png'


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 항상 aqprice, aqprice_L 만 변형별로 교체
# (aqprice_exclude_requester 는 안 넘기므로 네 변형 모두 기본 True = 요청노드 제외 유지)
# -------------------------------------------------------------------------
def run_one(variant, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    params = dict(BASE_PARAMS)
    params['aqprice_L'] = VARIANT_L[variant]
    sim = Simulator(algorithm='aqprice', params=params, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
    return sim, adt


# -------------------------------------------------------------------------
# Metric 계산 함수 — 전체구간 ADT 요약 2 지표
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

    results = {}
    for variant in VARIANTS:
        print(f"\n--- {LABELS[variant]} ---")
        adt_runs = []
        echo_costs = []
        switch_rates = []
        rx_runs, echo_runs, switch_runs, link_dicts = [], [], [], []
        te_runs, tm_runs = [], []          # R_x 의 분자·분모 시계열
        gens, dlvs, unds, rates = [], [], [], []
        for seed in SEEDS:
            print(f"  Running seed={seed}...")
            sim, adt = run_one(variant, lam, total_ticks, seed)
            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            adt_runs.append(adt)
            gens.append(gen); dlvs.append(dlv); unds.append(und); rates.append(rate)
            ec = (sim.total_echo_queries / sim.total_route_calls
                  if sim.total_route_calls > 0 else 0.0)
            echo_costs.append(ec)
            sw = (sim.total_switches / sim.total_route_calls
                  if sim.total_route_calls > 0 else 0.0)
            switch_rates.append(sw)
            rx_runs.append(sim.r_ratio_series)
            te_runs.append(sim.t_est_series)
            tm_runs.append(sim.t_max_series)
            echo_runs.append(sim.echo_per_decision_series)
            switch_runs.append(sim.switch_rate_series)
            link_dicts.append(sim.link_usage)
        adt_arr = np.array(adt_runs)
        results[variant] = {
            'adt_arr':   adt_arr,
            'median':    np.nanmedian(adt_arr, axis=0),
            'q25':       np.nanpercentile(adt_arr, 25, axis=0),
            'q75':       np.nanpercentile(adt_arr, 75, axis=0),
            'echo_cost':   float(np.mean(echo_costs)),      # 시드 10개 값의 평균 (패킷 전달 통계와 같은 규칙)
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

    # ---- 변형별 ADT 지표 (전체 평균 ADT, 전체 IQR 폭). 에코·변경률은 진단 시계열로 ----
    metrics = {v: compute_metrics(results[v]['adt_arr']) for v in VARIANTS}

    # ---- 시각화 : median 실선 + IQR 오차 막대 ----
    spacing = max(1, len(x_axis) // 10)
    for idx, variant in enumerate(VARIANTS):
        median = results[variant]['median']
        yerr = np.vstack([median - results[variant]['q25'],
                          results[variant]['q75'] - median])
        offset = idx * spacing // len(VARIANTS)
        ax.errorbar(x_axis, median, yerr=yerr,
                    errorevery=(offset, spacing),
                    capsize=3, elinewidth=1.2, capthick=1.2,
                    label=LABELS[variant], color=COLORS[variant], linewidth=2.0)

    ax.set_title(f"λ={lam} ({total_ticks} ticks)", fontsize=20)
    ax.set_xlabel('Simulator Time', fontsize=18)
    ax.set_ylabel('Average Delivery Time', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc='upper right', fontsize=17)
    ax.grid(True, alpha=0.3)

    # ---- MD 표 (변형별 열, 원값. 4 지표) ----
    cols = " | ".join(LABELS_SHORT[v] for v in VARIANTS)
    align = "|---" + "|---:" * len(VARIANTS) + "|\n"

    def row(name, fmt):
        return f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS) + " |\n"

    # 변형이 정확히 두 개일 때는 기준(REF_KEY) 대비 차이 열을 덧붙여
    # 다른 2자 비교 스크립트(aqrerm_vs_aqprice, inline_compare)와 같은 형식으로 낸다.
    # 셋 이상이면 어느 쌍의 차이인지 모호해지므로 원값 표만 낸다.
    two_way = (len(VARIANTS) == 2)
    if two_way:
        alt = [v for v in VARIANTS if v != REF_KEY][0]   # 기준이 아닌 쪽
        m_ref, m_alt = metrics[REF_KEY], metrics[alt]
        r_ref, r_alt = results[REF_KEY], results[alt]
        adt_diff = m_alt['overall'] - m_ref['overall']

        def row_d(name, fmt, diff_txt):
            """원값 열들 뒤에 차이 열 하나를 붙인 행."""
            return (f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS)
                    + f" | {diff_txt} |\n")

        # 차이 열은 모두 (기준이 아닌 쪽 − 기준) 절대 차이. 음수면 비교 대상이 그 지표에서 더 작다.
        md.write(f"| 지표 | {cols} | 차이 ({LABELS_SHORT[alt]}−{LABELS_SHORT[REF_KEY]}) |\n")
        md.write("|---" + "|---:" * (len(VARIANTS) + 1) + "|\n")
        md.write(row_d("전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱)",
                       lambda v: f"{metrics[v]['overall']:.2f}",
                       f"{adt_diff:+.2f}"))
        md.write(row_d("전체 IQR 폭 (띠 평균 폭 // 단위 = 틱)",
                       lambda v: f"{metrics[v]['overall_iqr']:.2f}",
                       f"{m_alt['overall_iqr'] - m_ref['overall_iqr']:+.2f}"))
        md.write(row_d("라우팅 결정당 이웃 조회 수 (전체 평균 // 단위 = 개/결정)",
                       lambda v: f"{results[v]['echo_cost']:.2f}",
                       f"{r_alt['echo_cost'] - r_ref['echo_cost']:+.2f}"))
        # 경로 변경률은 비율이라 백분율로, 차이는 두 백분율의 차이이므로 %p 로 적는다.
        md.write(row_d("경로 변경률 (전체 평균 // 단위 = %)",
                       lambda v: f"{results[v]['switch_rate'] * 100:.1f}%",
                       f"{(r_alt['switch_rate'] - r_ref['switch_rate']) * 100:+.1f}%p"))
        # R_x 는 진단 시계열의 전체구간 평균 (아래 시계열 표 끝 행과 같은 값)
        rx_ref = diag.series_overall(r_ref['rx_med'])
        rx_alt = diag.series_overall(r_alt['rx_med'])
        md.write(row_d("R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율)",
                       lambda v: f"{diag.series_overall(results[v]['rx_med']):.3f}",
                       f"{rx_alt - rx_ref:+.3f}"))
        # R_x 의 분자·분모. 둘 다 전체구간 시간평균 (R_x 와 같은 요약 규칙).
        te_ref, te_alt = diag.series_overall(r_ref['t_est_med']), diag.series_overall(r_alt['t_est_med'])
        tm_ref, tm_alt = diag.series_overall(r_ref['t_max_med']),   diag.series_overall(r_alt['t_max_med'])
        md.write(row_d("T_est (전체 평균, R_x 분자 // 단위 = 틱)",
                       lambda v: f"{diag.series_overall(results[v]['t_est_med']):.2f}",
                       f"{te_alt - te_ref:+.2f}"))
        md.write(row_d("T_max (전체 평균, R_x 분모 // 단위 = 틱)",
                       lambda v: f"{diag.series_overall(results[v]['t_max_med']):.2f}",
                       f"{tm_alt - tm_ref:+.2f}"))
    else:
        md.write(f"| 지표 | {cols} |\n")
        md.write(align)
        md.write(row("전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱)", lambda v: f"{metrics[v]['overall']:.2f}"))
        md.write(row("전체 IQR 폭 (띠 평균 폭 // 단위 = 틱)",              lambda v: f"{metrics[v]['overall_iqr']:.2f}"))
        md.write(row("라우팅 결정당 이웃 조회 수 (전체 평균 // 단위 = 개/결정)", lambda v: f"{results[v]['echo_cost']:.2f}"))
        md.write(row("경로 변경률 (전체 평균 // 단위 = %)",        lambda v: f"{results[v]['switch_rate'] * 100:.1f}%"))
        md.write(row("R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율)",
                     lambda v: f"{diag.series_overall(results[v]['rx_med']):.3f}"))
        md.write(row("T_est (전체 평균, R_x 분자 // 단위 = 틱)",
                     lambda v: f"{diag.series_overall(results[v]['t_est_med']):.2f}"))
        md.write(row("T_max (전체 평균, R_x 분모 // 단위 = 틱)",
                     lambda v: f"{diag.series_overall(results[v]['t_max_med']):.2f}"))
    md.write("\n")

    # ---- MD 진단 시계열 (변형별 구간 평균 + 전체 평균) ----
    for variant in VARIANTS:
        r = results[variant]
        diag.write_diag_series_md(md, LABELS[variant],
                                  r['rx_med'], r['echo_med'], r['switch_med'],
                                  r['echo_cost'], r['switch_rate'],
                                  t_est_med=r['t_est_med'], t_max_med=r['t_max_med'],
                                  stat_interval=STAT_INTERVAL)

    # ---- MD 패킷 전달 통계 ----
    md.write(f"#### 패킷 전달 통계 ({len(SEEDS)} 시드)\n\n")
    md.write(f"| 항목 | {cols} |\n")
    md.write(align)
    md.write(row("생성 (평균)",   lambda v: f"{results[v]['gen_mean']:.0f}"))
    md.write(row("전달 (평균)",   lambda v: f"{results[v]['dlv_mean']:.0f}"))
    md.write(row("미전달 (평균)", lambda v: f"{results[v]['und_mean']:.0f}"))
    md.write(row("성공률 평균",   lambda v: f"{results[v]['rate_mean']:.1f}%"))
    md.write(row("성공률 최고",   lambda v: f"{results[v]['rate_max']:.1f}%"))
    md.write(row("성공률 최저",   lambda v: f"{results[v]['rate_min']:.1f}%"))
    md.write("\n")

    # ---- MD 링크 사용량 수치 (변형별, 통과량 순) ----
    for variant in VARIANTS:
        md.write(f"#### 링크 사용량 ({LABELS[variant]}, 통과량 순, 시드 합)\n\n")
        md.write(diag.link_usage_md(results[variant]['link_sum']))
        md.write("\n")

    # ---- 콘솔 출력 ----
    head = " | ".join(f"{LABELS_SHORT[v]:>12}" for v in VARIANTS)
    print(f"  {'지표':<24} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>12}" for v in VARIANTS)
        print(f"  {name:<24} | {vals}")

    prow("전체 평균 ADT", lambda v: f"{metrics[v]['overall']:.2f}")
    prow("전체 IQR 폭",   lambda v: f"{metrics[v]['overall_iqr']:.2f}")
    if two_way:
        print(f"  {'→ ADT 차이(기준=' + LABELS_SHORT[REF_KEY] + ')':<24} | {adt_diff:>+11.2f}")

    # 링크 다이어그램용: 변형별 시드 합산 link_usage 반환
    return {v: results[v]['link_sum'] for v in VARIANTS}


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    # 패널 배치 — 부하 4개면 2x2. 패널 하나당 크기가 고정이라 실험 수가 달라져도 비율 유지.
    fig, axes = diag.make_panel_grid(len(EXPERIMENTS))
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — AQPRICE Route Memory on/off (AQPRICE / AQPRICE+L)  "
        f"(c={C}, seeds n={len(SEEDS)})",
        fontsize=22,
        y=0.99,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE 라우트 메모리 길이 비교 \n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS} (aqprice_L 만 변형별로 0/1/2/3, 요청노드 제외는 모두 유지)\n')
        md.write(f'- STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 출력 구성\n\n')
        md.write(f'- 각 변형은 라우트 메모리 길이(L)만 다르다. L=0 은 라우트 메모리 미사용(AQPRICE 정식 기본), '
                 f'L>0 은 최근 방문 노드를 L 개까지 다음 홉 후보에서 제외. 요청노드 제외는 모두 유지.\n')
        md.write(f'- 이번 실행의 변형별 aqprice_L: {VARIANT_L}\n')
        md.write('- 성능 비교표 : 전체 평균 ADT(그래프 중앙값 실선의 시간평균)와 전체 IQR 폭(그래프 띠의 평균 폭). 구간 분할 없이 실행 전체를 요약.\n')
        if len(VARIANTS) == 2:
            _alt = [v for v in VARIANTS if v != REF_KEY][0]
            md.write(f'- 차이 열 : 기준은 {LABELS_SHORT[REF_KEY]}, 비교 대상은 {LABELS_SHORT[_alt]}. '
                     f'모두 {LABELS_SHORT[_alt]} − {LABELS_SHORT[REF_KEY]} 절대 차이이며, '
                     f'음수면 {LABELS_SHORT[_alt]} 가 그 지표에서 더 작다.\n')
        md.write('- 진단 시계열 : R_x / 라우팅 결정당 이웃 조회 수 / 경로 변경률 구간 평균 + 끝의 전체 평균(에코·변경률 요약).\n')
        md.write('- 링크 사용량 : 별도 PNG 에 grid 링크 굵기·색 + md 통과량 표.\n\n')

        diag.write_metric_guide_md(md)

        link_data = []
        for ax, exp in zip(axes, EXPERIMENTS):
            per_var = run_lambda(ax, exp['lam'], exp['total_ticks'], md)
            link_data.append((exp['lam'], per_var))

    # rect 위쪽을 0.96 으로 잘라 suptitle 자리를 남긴다 (제목이 패널 제목과 겹치지 않게)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")

    # ---- 링크 사용량 다이어그램 (그리드: 행=변형, 열=부하). grid 만 ----
    if GRID_NUM_NODES == 36:
        nrows, ncols = len(VARIANTS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        figL.suptitle(
            f"{topo_name(GRID_NUM_NODES)} — link usage  [color & width = usage level]",
            fontsize=20)
        vmin, vmax = diag.link_usage_range([per[v] for (lam, per) in link_data for v in per])
        for r, variant in enumerate(VARIANTS):
            for c, (lam, per_var) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_var[variant],
                                         GRID_NUM_NODES, f"{LABELS_SHORT[variant]}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
