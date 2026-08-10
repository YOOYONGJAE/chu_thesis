# =============================================================================
# [요약] AQPRICE 포인트 게이트 유무 비교 — 예산·가격 판정만 제거
# - 같은 aqprice 를 두 설정으로 비교 (Inline·c·라우트메모리 미사용은 모두 동일):
#     AQPRICE : 정식 (total_point >= 가격 일 때만 Full Echo)
#     -Point  : 판정 없이 매 결정 Inline Full Echo (aqprice_point_gate=False)
# - node.py 는 aqprice_point_gate 로 이 판정을 제어 (기본 True)
# - 포인트 예산이 성능에 독립적으로 기여하는지를 단일 요인으로 확인
# - 나머지 제거 조건은 각각 아래 스크립트에 같은 시드·부하·BASE_PARAMS 로 이미 있다:
#     -Inline → main_aqprice_inline_compare.py 의 postecho
#     -c      → main_c_effect_compare.py 의 AQPRICE-c
#     +L      → main_aqprice_route_memory_L_changed.py 의 L=3
#   다섯 조건을 한 실행에서 다시 확인하려면 main_aqprice_ablation.py 를 쓴다.
# - 6x6 grid, 10 시드, 부하 4 단계
# - 산출물: result_aqprice_point_compare.md / .png / _linkuse.png
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
# from topology_nsfnet import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (변형별로 키 하나씩만 덮어씀)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3      # AQRERM 용 키. aqprice 는 aqprice_L 을 따로 읽으므로 여기 값에 영향받지 않는다.
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}


def topo_name(n):
    """노드 수로 토폴로지 이름 자동 판별 (import 만 바꾸면 제목도 자동 반영)."""
    return {36: '6x6 Grid', 14: 'NSFNET', 15: 'PQ'}.get(n, f'{n}-node')


# -------------------------------------------------------------------------
# 5 변형 — 정식을 기준으로 요소 하나씩만 제거(또는 복원)
# 일부만 돌리려면 VARIANTS 에서 빼면 된다 (REF_KEY 는 항상 남길 것).
# -------------------------------------------------------------------------
VARIANTS = ['aqprice', 'nopoint']
REF_KEY  = 'aqprice'          # 정식. 차이 표의 기준

VARIANT_CFG = {
    'aqprice':  {},                                   # 정식 (기본값 그대로)
    'nopoint':  {'aqprice_point_gate': False},        # 포인트·가격 판정 제거
    'noinline': {'aqprice_inline': False},            # 에코 정보를 현재 선택에 안 씀
    'noc':      {'c': 0.0},                           # 큐 페널티 제거
    'withL':    {'aqprice_L': 3},                     # 라우트 메모리 복원
}
LABELS = {
    'aqprice':  'AQPRICE (정식)',
    'nopoint':  'AQPRICE-Point (포인트 게이트 제거)',
    'noinline': 'AQPRICE-Inline (선택 먼저)',
    'noc':      'AQPRICE-c (큐 페널티 제거)',
    'withL':    'AQPRICE+L (라우트 메모리 L=3)',
}
LABELS_SHORT = {
    'aqprice':  'AQPRICE',
    'nopoint':  '-Point',
    'noinline': '-Inline',
    'noc':      '-c',
    'withL':    '+L',
}
# 적녹 구분에 의존하지 않는 색 (Okabe-Ito) + 선 모양 이중 인코딩
COLORS = {
    'aqprice':  '#56B4E9',   # 하늘색 (정식)
    'nopoint':  '#D55E00',   # 주홍
    'noinline': '#0072B2',   # 진파랑
    'noc':      '#E69F00',   # 주황
    'withL':    '#CC79A7',   # 분홍보라
}
LINESTYLES = {
    'aqprice':  '-',
    'nopoint':  '--',
    'noinline': '-.',
    'noc':      ':',
    'withL':    (0, (5, 1, 1, 1)),
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100

EXPERIMENTS = [
    # 4 단계 부하 — 저 / 중저 / 중고 / 고. 다른 비교 실험과 동일하게 맞춘다.
    # 포인트 게이트는 부하마다 상태가 달라(열림 / 차단 / 차단 / 열림) 네 부하가 모두 필요하다.
    {'lam': 2.5, 'total_ticks': 20000},   # 저부하   — 게이트 열림
    {'lam': 3.0, 'total_ticks': 20000},   # 중저부하 — 가격이 상한 초과, 게이트 차단
    {'lam': 3.5, 'total_ticks': 20000},   # 중고부하 — 게이트 차단
    {'lam': 3.8, 'total_ticks': 20000},   # 고부하   — 게이트 다시 열림
]

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = {36: 'GRID', 14: 'NSFNET', 15: 'PQ'}.get(GRID_NUM_NODES, f'{GRID_NUM_NODES}N')
RESULT_TAG = '4LVL'   # 부하 표기 (수동)
_BASENAME  = f'result_aqprice_point_compare({TOPO_TAG},{RESULT_TAG},{EXPERIMENTS[0]["total_ticks"]})'
MD_PATH       = f'{_BASENAME}.md'
PNG_PATH      = f'{_BASENAME}.png'
LINK_PNG_PATH = f'{_BASENAME}_linkuse.png'


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 항상 aqprice, 변형별 키 하나만 덮어씀
# -------------------------------------------------------------------------
def run_one(variant, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    params = dict(BASE_PARAMS)
    params.update(VARIANT_CFG[variant])
    sim = Simulator(algorithm='aqprice', params=params, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
    return sim, adt


# -------------------------------------------------------------------------
# Metric 계산 — 전체구간 ADT 요약 2 지표
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
                    label=LABELS[variant], color=COLORS[variant],
                    linestyle=LINESTYLES[variant], linewidth=2.0)

    ax.set_title(f"λ={lam} ({total_ticks} ticks)", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- 지표 접근 헬퍼 (원값 표와 차이 표가 같은 정의를 공유하도록 한 곳에 모음) ----
    def v_adt(v):    return metrics[v]['overall']
    def v_iqr(v):    return metrics[v]['overall_iqr']
    def v_echo(v):   return results[v]['echo_cost']
    def v_switch(v): return results[v]['switch_rate'] * 100.0     # 백분율
    def v_rx(v):     return diag.series_overall(results[v]['rx_med'])
    def v_te(v):     return diag.series_overall(results[v]['t_est_med'])
    def v_tm(v):     return diag.series_overall(results[v]['t_max_med'])

    cols  = " | ".join(LABELS_SHORT[v] for v in VARIANTS)
    align = "|---" + "|---:" * len(VARIANTS) + "|\n"

    def row(name, fmt):
        return f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS) + " |\n"

    # ---- MD 표 1 : 변형별 원값 ----
    md.write(f"| 지표 | {cols} |\n")
    md.write(align)
    md.write(row("전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱)", lambda v: f"{v_adt(v):.2f}"))
    md.write(row("전체 IQR 폭 (띠 평균 폭 // 단위 = 틱)",              lambda v: f"{v_iqr(v):.2f}"))
    md.write(row("라우팅 결정당 이웃 조회 수 (전체 평균 // 단위 = 개/결정)", lambda v: f"{v_echo(v):.2f}"))
    md.write(row("경로 변경률 (전체 평균 // 단위 = %)",                lambda v: f"{v_switch(v):.1f}%"))
    md.write(row("R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율)",   lambda v: f"{v_rx(v):.3f}"))
    md.write(row("T_est (전체 평균, R_x 분자 // 단위 = 틱)",           lambda v: f"{v_te(v):.2f}"))
    md.write(row("T_max (전체 평균, R_x 분모 // 단위 = 틱)",           lambda v: f"{v_tm(v):.2f}"))
    md.write("\n")

    # ---- MD 표 2 : 정식 대비 차이 (제거 조건들만) ----
    # 모두 (제거 조건 − 정식) 절대 차이. 양수면 그 요소를 뺐을 때 값이 커진 것이다.
    # ADT·IQR·경로 변경률은 작을수록 좋으므로, 양수면 그 요소가 기여했다는 뜻이 된다.
    alts = [v for v in VARIANTS if v != REF_KEY]
    if alts:
        dcols  = " | ".join(f"{LABELS_SHORT[v]} − {LABELS_SHORT[REF_KEY]}" for v in alts)
        dalign = "|---" + "|---:" * len(alts) + "|\n"

        def drow(name, fmt):
            return f"| {name} | " + " | ".join(fmt(v) for v in alts) + " |\n"

        md.write(f"#### 정식 대비 차이\n\n")
        md.write(f"| 지표 | {dcols} |\n")
        md.write(dalign)
        md.write(drow("전체 평균 ADT (틱)",
                      lambda v: f"{v_adt(v) - v_adt(REF_KEY):+.2f}"))
        md.write(drow("전체 IQR 폭 (틱)",
                      lambda v: f"{v_iqr(v) - v_iqr(REF_KEY):+.2f}"))
        md.write(drow("라우팅 결정당 이웃 조회 수 (개/결정)",
                      lambda v: f"{v_echo(v) - v_echo(REF_KEY):+.2f}"))
        # 경로 변경률은 비율이라 백분율로, 차이는 두 백분율의 차이이므로 %p 로 적는다.
        md.write(drow("경로 변경률 (%p)",
                      lambda v: f"{v_switch(v) - v_switch(REF_KEY):+.1f}%p"))
        md.write(drow("R_x",
                      lambda v: f"{v_rx(v) - v_rx(REF_KEY):+.3f}"))
        md.write(drow("T_est (틱)",
                      lambda v: f"{v_te(v) - v_te(REF_KEY):+.2f}"))
        md.write(drow("T_max (틱)",
                      lambda v: f"{v_tm(v) - v_tm(REF_KEY):+.2f}"))
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
    head = " | ".join(f"{LABELS_SHORT[v]:>10}" for v in VARIANTS)
    print(f"  {'지표':<26} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>10}" for v in VARIANTS)
        print(f"  {name:<26} | {vals}")

    prow("전체 평균 ADT",              lambda v: f"{v_adt(v):.2f}")
    prow("전체 IQR 폭",                lambda v: f"{v_iqr(v):.2f}")
    prow("라우팅 결정당 이웃 조회 수", lambda v: f"{v_echo(v):.2f}")
    print(f"  {'→ ADT 차이(정식 대비)':<26} | "
          + " | ".join(f"{(v_adt(v) - v_adt(REF_KEY)):>+10.2f}" for v in VARIANTS))

    # 링크 다이어그램용: 변형별 시드 합산 link_usage 반환
    return {v: results[v]['link_sum'] for v in VARIANTS}


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    # 패널 배치 — 부하 4개면 2x2. 패널 하나당 크기가 고정이라 실험 수가 달라져도 비율 유지.
    fig, axes = diag.make_panel_grid(len(EXPERIMENTS))
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — AQPRICE 포인트 게이트 유무 비교  "
        f"(c={C}, seeds n={len(SEEDS)})",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE 포인트 게이트 유무 비교\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- 변형별 덮어쓴 키: '
                 + ', '.join(f'{LABELS_SHORT[v]}={VARIANT_CFG[v] or "기본값"}' for v in VARIANTS)
                 + '\n')
        md.write(f'- STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 출력 구성\n\n')
        md.write('- 두 조건 모두 알고리즘은 aqprice 이고 aqprice_point_gate 하나만 다르다. '
                 'Inline Echo, 큐 페널티, 라우트 메모리 미사용, 요청 노드 제외는 모두 동일하므로 '
                 '단일 요인 비교가 된다.\n')
        md.write('- -Point 는 포인트·가격 판정을 건너뛰어 매 결정 Inline Full Echo 를 수행한다. '
                 '적응형 Echo 학습률(eta2 = R_x·k)은 그대로 쓰이므로, 계보상 AQFE 에 Inline 과 '
                 '큐 페널티를 얹은 형태에 가깝다.\n')
        md.write('- 나머지 제거 조건(-Inline / -c / +L)은 각각 inline_compare, c_effect_compare, '
                 'route_memory_L_changed 에 같은 조건으로 이미 있다.\n')
        md.write('- 정식 대비 차이 표는 모두 (제거 조건 − 정식) 절대 차이다. '
                 'ADT·IQR 폭·경로 변경률은 작을수록 좋으므로, 양수면 그 요소를 뺐을 때 나빠졌다는 뜻이다.\n')
        md.write('- 진단 시계열 : R_x / 라우팅 결정당 이웃 조회 수 / 경로 변경률 구간 평균 + 끝의 전체 평균.\n')
        md.write('- 링크 사용량 : 별도 PNG 에 grid 링크 굵기·색 + md 통과량 표.\n\n')

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

    # ---- 링크 사용량 다이어그램 (그리드: 행=변형, 열=부하). grid 만 ----
    if GRID_NUM_NODES == 36:
        nrows, ncols = len(VARIANTS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        figL.suptitle(
            f"{topo_name(GRID_NUM_NODES)} — link usage  [color & width = usage level]",
            fontsize=15)
        vmin, vmax = diag.link_usage_range([per[v] for (lam, per) in link_data for v in per])
        for r, variant in enumerate(VARIANTS):
            for c, (lam, per_var) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_var[variant],
                                         GRID_NUM_NODES,
                                         f"{LABELS_SHORT[variant]}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
