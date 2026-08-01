# =============================================================================
# [요약] AQPRICE Inline Echo 유무 비교 — 에코 결과를 현재 선택에 쓰는지 여부
# - 같은 aqprice 를 두 설정으로 비교 (에코·포인트·c·라우트메모리 다 동일, 순서만 다름):
#     inline  : 에코 먼저 → 최신 t_n + c·큐 로 현재 패킷 선택 (AQPRICE 정식)
#     postecho: 기존 Q + c·큐캐시로 먼저 선택 → 그 뒤 에코 (AQFE/AQRERM 처럼 선택 먼저)
# - node.py 는 aqprice_inline 로 이 순서를 제어 (기본 True=inline)
# - Inline 이 실제로 성능을 바꾸는지(=AQPRICE 의 novelty 검증) 순서만 바꿔 분리 확인
# - 6x6 grid, 10 시드. 4 지표 + 패킷 전달 통계 (구간 분할 없이 실행 전체 요약)
# - 산출물: result_aqprice_inline_compare.md / .png
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


def topo_name(n):
    """노드 수로 토폴로지 이름 자동 판별 (import 만 바꾸면 제목도 자동 반영)."""
    return {36: '6x6 Grid', 14: 'NSFNET', 15: 'PQ'}.get(n, f'{n}-node')


# -------------------------------------------------------------------------
# 파라미터 (main_aqrerm_vs_aqprice 와 동일. aqprice_inline 만 변형별로 세팅)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 비교 대상: 같은 aqprice, 에코-선택 순서만 다름 (나머지 요소는 모두 동일)
#   postecho : 선택 먼저(기존 Q + 큐캐시) → 에코 (aqprice_inline=False)
#   inline   : 에코 먼저 → 최신값으로 선택 (aqprice_inline=True, 현재 기본)
# -------------------------------------------------------------------------
VARIANTS = ['postecho', 'inline']   # 표/그래프 순서: baseline(post) → main(inline=현재)
BASE_KEY = 'postecho'               # 개선 계산의 기준(baseline)
MAIN_KEY = 'inline'                 # 현재 AQPRICE 기본
VARIANT_INLINE = {'postecho': False, 'inline': True}
LABELS = {
    'postecho': 'AQPRICE (post-echo, select first)',
    'inline':   'AQPRICE (inline, current)',
}
LABELS_SHORT = {
    'postecho': 'post-echo',
    'inline':   'inline(현재)',
}
COLORS = {
    'postecho': '#D55E00',  # 주홍
    'inline':   '#56B4E9',  # 하늘색 (현재 기본)
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100
TOTAL_TICKS = 20000

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = {36: 'GRID', 14: 'NSFNET', 15: 'PQ'}.get(GRID_NUM_NODES, f'{GRID_NUM_NODES}N')
RESULT_TAG = 'LOW'   # 부하 표기 (수동): LOW / HIGH 등
# RESULT_TAG = 'HIGH'   # 부하 표기 (수동): LOW / HIGH 등
_BASENAME  = f'result_aqprice_inline_compare({TOPO_TAG},{RESULT_TAG},{TOTAL_TICKS})'
MD_PATH       = f'{_BASENAME}.md'
PNG_PATH      = f'{_BASENAME}.png'
LINK_PNG_PATH = f'{_BASENAME}_linkuse.png'

EXPERIMENTS = [
    # 저, 중부하
    {'lam': 2.5, 'total_ticks': 20000},
    {'lam': 3.0, 'total_ticks': 20000},
    {'lam': 3.5, 'total_ticks': 20000},

    # 고부하
    # {'lam': 3.7, 'total_ticks': 20000},
    # {'lam': 3.9, 'total_ticks': 20000},
    # {'lam': 4.1, 'total_ticks': 20000},
]


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 항상 aqprice, aqprice_inline 만 교체
# (aqprice_L / aqprice_exclude_requester 는 안 넘기므로 정식 기본값 유지)
# -------------------------------------------------------------------------
def run_one(variant, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    params = dict(BASE_PARAMS)
    params['aqprice_inline'] = VARIANT_INLINE[variant]
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


def pct_improvement(baseline, new):
    """baseline 대비 new 가 얼마나 줄었는지 % (양수면 개선)."""
    if baseline == 0:
        return 0.0
    return (1.0 - new / baseline) * 100.0


# -------------------------------------------------------------------------
# 한 부하 실험 (한 패널)
# -------------------------------------------------------------------------
def run_lambda(ax, lam, total_ticks, md):
    x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

    md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
    print(f"\n========== λ={lam} ==========")

    results = {}
    for variant in VARIANTS:
        label = LABELS[variant]
        print(f"\n--- {label} ---")
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
            # 진단 시계열·링크 사용량 수집
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
            'echo_cost':   float(np.nanmedian(echo_costs)),   # 진단 시계열 요약(전체 평균)용 스칼라
            'switch_rate': float(np.nanmedian(switch_rates)),
            'rx_med':     diag.median_series(rx_runs),         # 윈도우별 시드 간 중앙값
            't_est_med':  diag.median_series(te_runs),
            't_max_med':  diag.median_series(tm_runs),
            'echo_med':   diag.median_series(echo_runs),
            'switch_med': diag.median_series(switch_runs),
            'link_sum':   diag.sum_link_usage(link_dicts),     # 시드 합산 링크 사용량
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

    ax.set_title(f"λ={lam}", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    # ---- 개선 = baseline(post-echo) 대비 main(inline). ADT 전체구간 요약 2지표만 표로 ----
    base_m = metrics[BASE_KEY]
    main_m = metrics[MAIN_KEY]
    adt_imp = pct_improvement(base_m['overall'], main_m['overall'])

    base_col = 'post-echo'
    main_col = 'inline (현재)'

    # ADT 는 % 개선(값이 커서 %가 유효), IQR·에코·변경률은 값이 작아 % 가 과장되므로 절대 차이(inline−post).
    md.write(f"| 지표 | {base_col} | {main_col} | 개선(ADT) / 차이(그 외) |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| 전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱) | {base_m['overall']:.2f} | {main_m['overall']:.2f} | {adt_imp:+.1f}% |\n")
    md.write(f"| 전체 IQR 폭 (띠 평균 폭, 작을수록 일관 // 단위 = 틱) | {base_m['overall_iqr']:.2f} | {main_m['overall_iqr']:.2f} | {main_m['overall_iqr'] - base_m['overall_iqr']:+.2f} |\n")
    # 진단 시계열 전체 평균 중 비교 지표(에코·변경률)를 위 표에도 함께. R_x 는 시계열에만.
    br, mr = results[BASE_KEY], results[MAIN_KEY]
    md.write(f"| 결정당 에코 이웃 수 (전체 평균 // 단위 = 개/결정) | {br['echo_cost']:.2f} | {mr['echo_cost']:.2f} | {mr['echo_cost'] - br['echo_cost']:+.2f} |\n")
    # 경로 변경률은 비율이라 백분율로, 차이는 두 백분율의 차이이므로 %p 로 적는다.
    md.write(f"| 경로 변경률 (전체 평균 // 단위 = %) | {br['switch_rate'] * 100:.1f}% | {mr['switch_rate'] * 100:.1f}% | "
             f"{(mr['switch_rate'] - br['switch_rate']) * 100:+.1f}%p |\n")
    # R_x 는 진단 시계열의 전체구간 평균 (아래 시계열 표의 '전체 평균' 과 같은 값)
    br_rx, mr_rx = diag.series_overall(br['rx_med']), diag.series_overall(mr['rx_med'])
    md.write(f"| R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율) | {br_rx:.3f} | {mr_rx:.3f} | {mr_rx - br_rx:+.3f} |\n")
    # R_x 의 분자·분모. T_est 는 전체구간 평균, T_max 는 단조 증가라 최종값.
    br_te, mr_te = diag.series_overall(br['t_est_med']), diag.series_overall(mr['t_est_med'])
    br_tm, mr_tm = diag.series_final(br['t_max_med']),   diag.series_final(mr['t_max_med'])
    md.write(f"| T_est (전체 평균, R_x 분자 // 단위 = 틱) | {br_te:.2f} | {mr_te:.2f} | {mr_te - br_te:+.2f} |\n")
    md.write(f"| T_max (최종값, R_x 분모 // 단위 = 틱) | {br_tm:.2f} | {mr_tm:.2f} | {mr_tm - br_tm:+.2f} |\n\n")

    # ---- 진단 시계열 (변형별 구간 평균 + 전체 평균). 에코·변경률 요약이 여기 들어감 ----
    for variant in VARIANTS:
        r = results[variant]
        diag.write_diag_series_md(md, LABELS[variant],
                                  r['rx_med'], r['echo_med'], r['switch_med'],
                                  r['echo_cost'], r['switch_rate'],
                                  t_est_med=r['t_est_med'], t_max_med=r['t_max_med'])

    # ---- 패킷 전달 통계 ----
    base_r = results[BASE_KEY]
    main_r = results[MAIN_KEY]
    md.write(f"#### 패킷 전달 통계 ({len(SEEDS)} 시드)\n\n")
    md.write(f"| 항목 | {base_col} | {main_col} |\n")
    md.write("|---|---:|---:|\n")
    md.write(f"| 생성 (평균) | {base_r['gen_mean']:.0f} | {main_r['gen_mean']:.0f} |\n")
    md.write(f"| 전달 (평균) | {base_r['dlv_mean']:.0f} | {main_r['dlv_mean']:.0f} |\n")
    md.write(f"| 미전달 (평균) | {base_r['und_mean']:.0f} | {main_r['und_mean']:.0f} |\n")
    md.write(f"| 성공률 평균 | {base_r['rate_mean']:.1f}% | {main_r['rate_mean']:.1f}% |\n")
    md.write(f"| 성공률 최고 | {base_r['rate_max']:.1f}% | {main_r['rate_max']:.1f}% |\n")
    md.write(f"| 성공률 최저 | {base_r['rate_min']:.1f}% | {main_r['rate_min']:.1f}% |\n\n")

    print(f"  [ADT] {base_col:>14}: 전체={base_m['overall']:.2f} IQR폭={base_m['overall_iqr']:.2f}")
    print(f"  [ADT] {main_col:>14}: 전체={main_m['overall']:.2f} IQR폭={main_m['overall_iqr']:.2f}")

    # ---- 링크 사용량 수치 (변형별, 통과량 순) → md ----
    for variant in VARIANTS:
        md.write(f"#### 링크 사용량 ({LABELS[variant]}, 통과량 순, 시드 합)\n\n")
        md.write(diag.link_usage_md(results[variant]['link_sum']))
        md.write("\n")

    # 링크 다이어그램용: 변형별 시드 합산 link_usage 반환
    return {v: results[v]['link_sum'] for v in VARIANTS}


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(20 * len(EXPERIMENTS), 12), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — AQPRICE Inline Echo on/off (post-echo / inline)  "
        f"(c={C}, seeds n={len(SEEDS)}, {TOTAL_TICKS}tick)",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE Inline Echo 유무 비교 (에코 결과를 현재 선택에 쓰는지)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- 변형별 aqprice_inline: postecho=False, inline=True(현재 기본)\n')
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 출력 구성\n\n')
        md.write('- 비교 대상 : 이번 Full Echo 로 얻은 최신 정보(t_n, 큐)를 현재 패킷의 next-hop 선택에 쓸지 여부.\n')
        md.write('  inline(현재)은 에코 먼저 → 최신값으로 선택. postecho 는 기존 Q + 이전 큐캐시로 먼저 선택 → 그 뒤 에코(AQFE/AQRERM 식).\n')
        md.write('  포인트·c·라우트메모리·요청노드 제외는 두 변형 모두 동일 → 차이는 오직 에코-선택 순서에서 온다.\n\n')
        md.write('- 성능 비교표 : 전체 평균 ADT(그래프 중앙값 실선의 시간평균)와 전체 IQR 폭(그래프 띠의 평균 폭, 작을수록 일관). 구간 분할 없이 실행 전체를 요약.\n')
        md.write('- 진단 시계열 : R_x(=T_est/T_max) / 결정당 에코 이웃 수 / 경로 변경률 을 구간 평균으로. '
                 '끝의 전체 평균이 에코·변경률의 요약값(총합 기준).\n')
        md.write('- 링크 사용량 : 별도 PNG 에 grid 링크 굵기(점선) + 통과량 정렬 리스트.\n\n')

        md.write('> 개선 열은 baseline(post-echo) 대비 main(inline, 현재 기본) 기준. '
                 '양수면 Inline(최신 정보를 현재 선택에 반영)이 그 지표를 개선.\n\n')

        diag.write_metric_guide_md(md)

        link_data = []   # [(lam, {variant: link_sum}), ...]
        for ax, exp in zip(axes, EXPERIMENTS):
            per_var = run_lambda(ax, exp['lam'], exp['total_ticks'], md)
            link_data.append((exp['lam'], per_var))

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")

    # ---- 링크 사용량 다이어그램 (그리드: 행=변형, 열=부하) → 변형·부하 간 비교 ----
    # (수치는 md 에 이미 변형별로 기록. 여기 그림은 다이어그램만. 그림 텍스트는 영어.)
    if GRID_NUM_NODES == 36:
        nrows, ncols = len(VARIANTS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                                   squeeze=False)
        figL.suptitle(
            f"{topo_name(GRID_NUM_NODES)} — link usage  [color & width = usage level]",
            fontsize=15)
        vmin, vmax = diag.link_usage_range([per[v] for (lam, per) in link_data for v in per])
        for r, variant in enumerate(VARIANTS):
            for c, (lam, per_var) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_var[variant],
                                         GRID_NUM_NODES, f"{variant}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
