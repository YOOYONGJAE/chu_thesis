# =============================================================================
# [요약] 큐 페널티 c 효과 비교 — 4 변형
#   aqrerm_c    : AQRERM + 큐 페널티 c (aqrerm_c>0)
#   aqrerm      : 순수 AQRERM (c 없음)
#   aqprice_noc : AQPRICE 에서 c 제거 (c=0)
#   aqprice     : AQPRICE (c 있음, 정식)
# - 배경: PQ(15노드) 토폴로지 고부하에서 c 가 있는 AQPRICE 가 AQRERM 보다 오히려 발산하는
#         현상을 관찰. c(이웃 큐 혼잡 반영)가 고부하에서 역효과인지 4 변형으로 확인.
# - node.py 의 aqrerm_c(AQRERM 큐 페널티) / c(AQPRICE 큐 페널티) 로 제어
# - 4 지표: 전체 평균 ADT / 전체 IQR 폭 / 결정당 이웃 정보 조회 수 / 경로 변경률 (구간 분할 없음)
# - 산출물: result_c_effect_compare.md / .png
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
# 토폴로지 선택 — import 와 TOPO_NAME 을 같이 토글 (제목이 실제 토폴로지에 맞게 나옴)
from topology_pq import NUM_NODES as PQ_NUM_NODES, ADJACENCY as PQ_ADJACENCY
TOPO_NAME = 'PQ 15노드'
# from topology_grid import NUM_NODES as PQ_NUM_NODES, ADJACENCY as PQ_ADJACENCY
# TOPO_NAME = '6x6 Grid'
# from topology_nsfnet import NUM_NODES as PQ_NUM_NODES, ADJACENCY as PQ_ADJACENCY
# TOPO_NAME = 'NSFNET 14노드'

# -------------------------------------------------------------------------
# 파라미터 (c 관련만 변형별로 덮어씀)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22   # 큐 페널티 계수 (aqprice 의 c, aqrerm_c 에 동일 값 사용)
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_PQ = {'num_nodes': PQ_NUM_NODES, 'adjacency': PQ_ADJACENCY}

# -------------------------------------------------------------------------
# 4 변형: (algorithm, 덮어쓸 파라미터)
#   c 를 켜고 끄며 AQRERM / AQPRICE 각각에서 큐 페널티 효과를 분리
# -------------------------------------------------------------------------
VARIANTS = ['aqrerm_c', 'aqrerm', 'aqprice_noc', 'aqprice']
VARIANT_CFG = {
    'aqrerm_c':    {'algorithm': 'aqrerm',  'params': {'aqrerm_c': C}},   # AQRERM + c
    'aqrerm':      {'algorithm': 'aqrerm',  'params': {'aqrerm_c': 0.0}}, # 순수 AQRERM
    'aqprice_noc': {'algorithm': 'aqprice', 'params': {'c': 0.0}},        # AQPRICE - c
    'aqprice':     {'algorithm': 'aqprice', 'params': {'c': C}},          # AQPRICE (정식)
}
LABELS = {
    'aqrerm_c':    'AQRERM + c',
    'aqrerm':      'AQRERM',
    'aqprice_noc': 'AQPRICE (c removed)',
    'aqprice':     'AQPRICE',
}
LABELS_SHORT = {
    'aqrerm_c':    'AQRERM+c',
    'aqrerm':      'AQRERM',
    'aqprice_noc': 'AQPRICE-c',
    'aqprice':     'AQPRICE',
}
COLORS = {
    'aqrerm_c':    "#5EFF00",  # 분홍보라
    'aqrerm':      '#D55E00',  # 주홍
    'aqprice_noc': '#009E73',  # 초록
    'aqprice':     '#56B4E9',  # 하늘색 (정식)
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100

# 고부하 위주 (발산 현상이 나타나는 구간)
EXPERIMENTS = [
    # 4 단계 부하 — 저 / 중저 / 중고 / 고. 모든 실험 공통.
    # 고부하 상한을 3.8 로 둔 이유 : AQRERM 은 발산하고 AQPRICE 는 정착하는 경계 지점.
    {'lam': 2.5, 'total_ticks': 20000},   # 저부하
    {'lam': 3.0, 'total_ticks': 20000},   # 중저부하
    {'lam': 3.5, 'total_ticks': 20000},   # 중고부하
    {'lam': 3.8, 'total_ticks': 20000},   # 고부하
]

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = {36: 'GRID', 14: 'NSFNET', 15: 'PQ'}.get(PQ_NUM_NODES, f'{PQ_NUM_NODES}N')
RESULT_TAG = 'HIGH'   # 부하 표기 (수동): LOW / HIGH 등
_BASENAME  = f'result_c_effect_compare({TOPO_TAG},{RESULT_TAG},{EXPERIMENTS[0]["total_ticks"]})'
MD_PATH       = f'{_BASENAME}.md'
PNG_PATH      = f'{_BASENAME}.png'
LINK_PNG_PATH = f'{_BASENAME}_linkuse.png'


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 변형별 algorithm + 파라미터 override
# -------------------------------------------------------------------------
def run_one(variant, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    cfg = VARIANT_CFG[variant]
    params = dict(BASE_PARAMS)
    params.update(cfg['params'])
    sim = Simulator(algorithm=cfg['algorithm'], params=params, seed=seed, topology=TOPOLOGY_PQ)
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
            'echo_cost':   float(np.nanmedian(echo_costs)),
            'switch_rate': float(np.nanmedian(switch_rates)),
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

    ax.set_title(f"λ={lam} ({TOPO_NAME})", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- MD 표 (변형별 열, 원값. 4 지표) ----
    cols = " | ".join(LABELS_SHORT[v] for v in VARIANTS)
    align = "|---" + "|---:" * len(VARIANTS) + "|\n"

    def row(name, fmt):
        return f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS) + " |\n"

    md.write(f"| 지표 | {cols} |\n")
    md.write(align)
    md.write(row("전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱)", lambda v: f"{metrics[v]['overall']:.2f}"))
    md.write(row("전체 IQR 폭 (띠 평균 폭 // 단위 = 틱)",              lambda v: f"{metrics[v]['overall_iqr']:.2f}"))
    md.write(row("결정당 이웃 정보 조회 수 (전체 평균 // 단위 = 개/결정)", lambda v: f"{results[v]['echo_cost']:.2f}"))
    # 경로 변경률은 비율이라 백분율로 적는다.
    md.write(row("경로 변경률 (전체 평균 // 단위 = %)",        lambda v: f"{results[v]['switch_rate'] * 100:.1f}%"))
    # R_x 는 진단 시계열의 전체구간 평균 (아래 시계열 표의 '전체 평균' 과 같은 값)
    md.write(row("R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율)",
                 lambda v: f"{diag.series_overall(results[v]['rx_med']):.3f}"))
    # R_x 의 분자·분모. 둘 다 전체구간 시간평균 (R_x 와 같은 요약 규칙).
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
                                  t_est_med=r['t_est_med'], t_max_med=r['t_max_med'])

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
    head = " | ".join(f"{LABELS_SHORT[v]:>10}" for v in VARIANTS)
    print(f"  {'지표':<22} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>10}" for v in VARIANTS)
        print(f"  {name:<22} | {vals}")

    prow("전체 평균 ADT", lambda v: f"{metrics[v]['overall']:.2f}")
    prow("전체 IQR 폭",   lambda v: f"{metrics[v]['overall_iqr']:.2f}")

    # 링크 다이어그램용: 변형별 시드 합산 link_usage 반환
    return {v: results[v]['link_sum'] for v in VARIANTS}


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    # 패널 배치 — 부하 4개면 2x2. 패널 하나당 크기가 고정이라 실험 수가 달라져도 비율 유지.
    fig, axes = diag.make_panel_grid(len(EXPERIMENTS))
    fig.suptitle(
        f"{TOPO_NAME} — Queue-penalty c effect compare (AQRERM+c / AQRERM / AQPRICE-c / AQPRICE)  "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)})",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write(f'# 큐 페널티 c 효과 비교 ({TOPO_NAME}, 4 변형)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS} (c={C})\n')
        md.write(f'- 변형 설정:\n')
        for v in VARIANTS:
            cfg = VARIANT_CFG[v]
            md.write(f"  - {v} : algorithm={cfg['algorithm']}, override={cfg['params']}\n")
        md.write(f'- STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 출력 구성\n\n')
        md.write('- 배경 : PQ 토폴로지 고부하에서 c(이웃 큐 혼잡 반영)가 있는 AQPRICE 가 AQRERM 보다 발산하는 현상 관찰.\n')
        md.write('  c 를 AQRERM 에 붙이고(aqrerm_c) AQPRICE 에서 빼서(c=0) c 의 순효과를 4 변형으로 분리.\n')
        md.write('- 성능 비교표 : 전체 평균 ADT(그래프 중앙값 실선의 시간평균)와 전체 IQR 폭(그래프 띠의 평균 폭). 구간 분할 없이 실행 전체를 요약.\n')
        md.write('- 진단 시계열 : R_x / 결정당 이웃 정보 조회 수 / 경로 변경률 구간 평균 + 끝의 전체 평균(에코·변경률 요약).\n')
        md.write('- 링크 사용량 : 별도 PNG 에 grid 링크 굵기·색(grid 만) + md 통과량 표.\n\n')

        diag.write_metric_guide_md(md)

        link_data = []
        for ax, exp in zip(axes, EXPERIMENTS):
            per_var = run_lambda(ax, exp['lam'], exp['total_ticks'], md)
            link_data.append((exp['lam'], per_var))

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")

    # ---- 링크 사용량 다이어그램 (그리드: 행=변형, 열=부하). grid(36) 에서만 ----
    if PQ_NUM_NODES == 36:
        nrows, ncols = len(VARIANTS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        figL.suptitle(f"{TOPO_NAME} — link usage  [color & width = usage level]", fontsize=15)
        vmin, vmax = diag.link_usage_range([per[v] for (lam, per) in link_data for v in per])
        for r, variant in enumerate(VARIANTS):
            for c, (lam, per_var) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_var[variant],
                                         PQ_NUM_NODES, f"{variant}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid(36) 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
