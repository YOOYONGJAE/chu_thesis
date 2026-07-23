# =============================================================================
# [요약] AQPRICE 요청 노드 제외 ablation — 이웃이 최소 Q 를 낼 때 질문자를 뺄지 비교
# - 같은 aqprice 를 두 설정으로 비교:
#     exclude : 이웃이 최소 Q 계산에서 질문한 노드(x)를 제외 (현재 기본, AQRERM 라우트 메모리에서 온 조건)
#     include : 질문한 노드도 포함 (Q-routing/AQFE 와 동일, 순수 min)
# - node.py 는 aqprice_exclude_requester 로 이 조건을 제어 (기본 True=제외)
# - 라우트 메모리는 둘 다 끔(L=0, AQPRICE 정식 구성) → 요청 노드 제외 효과만 분리
# - 6x6 grid, 10 시드, λ=2/3.5/3.8 (20000 tick). 지표표 + 그래프 생산
# - 지표: 정착 ADT / 정착 ADT 중간 50% 범위 / 누적 ADT / 최악 ADT /
#         정착 후 상위 5% 지연 / 랜덤시드 간 변동성 / 결정당 에코 이웃 수 / 기준선 도달 시간
# - 기준선 = exclude(현재 기본) 정착 ADT 중앙값보다 20% 높은 값 (× 1.2, 양 변형 공통)
# - 산출물: result_compare_AQPRICE_requester_exclude.md / .png
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트 (그래프 한글 깨짐 방지)
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (main_aqrerm_vs_aqprice 와 동일. aqprice_exclude_requester 만 변형별로 세팅)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 비교 대상: 같은 aqprice, 요청 노드 제외 여부만 다름 (둘 다 라우트 메모리 없음)
#   include : 이웃이 최소 Q 낼 때 질문자도 포함 (aqprice_exclude_requester=False)
#   exclude : 이웃이 최소 Q 낼 때 질문자 제외 (aqprice_exclude_requester=True, 현재 기본)
# -------------------------------------------------------------------------
VARIANTS = ['include', 'exclude']   # 표/그래프 순서: baseline(포함) → main(제외=현재)
BASE_KEY = 'include'                # 개선 계산의 기준(baseline)
MAIN_KEY = 'exclude'                # 현재 AQPRICE 기본, 기준선도 이쪽 정착값 기준
VARIANT_EXCLUDE = {'include': False, 'exclude': True}
LABELS = {
    'include': 'AQPRICE (요청노드 포함, exclude=False)',
    'exclude': 'AQPRICE (요청노드 제외, exclude=True)',
}
COLORS = {
    'include': '#D55E00',  # 주홍
    'exclude': '#56B4E9',  # 하늘색 (현재 기본)
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100
TOTAL_TICKS = 20000

CONVERGENCE_THRESHOLD_RATIO = 1.2

MD_PATH  = 'result_compare_AQPRICE_requester_exclude.md'
PNG_PATH = 'result_compare_AQPRICE_requester_exclude.png'

EXPERIMENTS = [
    {'lam': 2, 'total_ticks': TOTAL_TICKS, 'title': 'λ=2.0'},
    {'lam': 3.5, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.5'},
    {'lam': 3.8, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.8'},
]


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 알고리즘은 항상 aqprice, 요청노드 제외 토글만 교체
# (aqprice_L 은 안 넘기므로 라우트 메모리는 두 변형 모두 기본 0 = 미사용)
# -------------------------------------------------------------------------
def run_one(variant, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    params = dict(BASE_PARAMS)
    params['aqprice_exclude_requester'] = VARIANT_EXCLUDE[variant]
    sim = Simulator(algorithm='aqprice', params=params, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
    return sim, adt


# -------------------------------------------------------------------------
# Metric 계산 함수들 (main_aqrerm_vs_aqprice 와 동일)
# -------------------------------------------------------------------------
def compute_target_reach_time(median_series, threshold, x_axis, min_fraction=0.95):
    """median ADT 시계열이 기준선(threshold) 아래로 '안정적으로' 유지되기 시작한 tick 반환.
    조건: 그 시점 이후의 윈도우 중 min_fraction (기본 95%) 이상이 threshold 이하. 못 도달하면 None."""
    median_series = np.asarray(median_series)
    n = len(median_series)
    for i in range(n):
        rest = median_series[i:]
        if np.mean(rest <= threshold) >= min_fraction:
            return x_axis[i]
    return None


def compute_auc(adt_arr):
    """시드별 ADT 시계열의 합 → 시드 간 median 반환 (누적 ADT)."""
    per_seed_auc = np.nansum(adt_arr, axis=1)
    return float(np.nanmedian(per_seed_auc)), per_seed_auc


def compute_worst_spike(adt_arr):
    """시드별 max ADT → 시드 간 median 반환 (최악 ADT, 구간 최댓값)."""
    per_seed_max = np.nanmax(adt_arr, axis=1)
    return float(np.nanmedian(per_seed_max)), per_seed_max


def compute_late_p95(adt_arr):
    """각 시드의 뒤쪽 절반 구간 95퍼센타일 → 시드 간 median 반환 (정착 후 상위 5% 지연)."""
    half = adt_arr.shape[1] // 2
    per_seed_p95 = np.nanpercentile(adt_arr[:, half:], 95, axis=1)
    return float(np.nanmedian(per_seed_p95)), per_seed_p95


def compute_ss_metrics(adt_arr):
    """정착 ADT (뒤쪽 절반 평균) per seed → median / mean / std / CV / IQR / range."""
    half = adt_arr.shape[1] // 2
    ss_per_seed = np.nanmean(adt_arr[:, half:], axis=1)
    return {
        'per_seed': ss_per_seed,
        'median': float(np.nanmedian(ss_per_seed)),
        'mean':   float(np.nanmean(ss_per_seed)),
        'std':    float(np.nanstd(ss_per_seed)),
        'cv':     float(np.nanstd(ss_per_seed) / np.nanmean(ss_per_seed)) if np.nanmean(ss_per_seed) > 0 else 0.0,
        'q25':    float(np.nanpercentile(ss_per_seed, 25)),
        'q75':    float(np.nanpercentile(ss_per_seed, 75)),
        'min':    float(np.nanmin(ss_per_seed)),
        'max':    float(np.nanmax(ss_per_seed)),
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
        adt_arr = np.array(adt_runs)
        results[variant] = {
            'adt_arr':   adt_arr,
            'median':    np.nanmedian(adt_arr, axis=0),
            'q25':       np.nanpercentile(adt_arr, 25, axis=0),
            'q75':       np.nanpercentile(adt_arr, 75, axis=0),
            'echo_cost': float(np.nanmedian(echo_costs)),
            'gen_mean':  float(np.mean(gens)),
            'dlv_mean':  float(np.mean(dlvs)),
            'und_mean':  float(np.mean(unds)),
            'rate_mean': float(np.mean(rates)),
            'rate_max':  float(np.max(rates)),
            'rate_min':  float(np.min(rates)),
        }

    # ---- 기준선 = MAIN(exclude) 정착 ADT 중앙값 × 1.2 ----
    main_ss = compute_ss_metrics(results[MAIN_KEY]['adt_arr'])
    threshold = main_ss['median'] * CONVERGENCE_THRESHOLD_RATIO

    metrics = {}
    for variant in VARIANTS:
        adt_arr = results[variant]['adt_arr']
        median_series = results[variant]['median']
        ss = compute_ss_metrics(adt_arr)
        auc_median, _ = compute_auc(adt_arr)
        worst_median, _ = compute_worst_spike(adt_arr)
        p95_median, _ = compute_late_p95(adt_arr)
        conv_tick = compute_target_reach_time(median_series, threshold, x_axis)
        metrics[variant] = {
            'ss':        ss,
            'auc':       auc_median,
            'worst':     worst_median,
            'p95':       p95_median,
            'conv_tick': conv_tick,
            'echo_cost': results[variant]['echo_cost'],
        }

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

    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.2,
               label=f'reference = exclude steady ADT x {CONVERGENCE_THRESHOLD_RATIO} ({threshold:.2f})')
    for variant in VARIANTS:
        conv_tick = metrics[variant]['conv_tick']
        if conv_tick is not None:
            ax.axvline(x=conv_tick, color=COLORS[variant], linestyle=':',
                       linewidth=1.5, alpha=0.7)
            ax.annotate(f'{LABELS[variant]}\nreach: {conv_tick}',
                        xy=(conv_tick, threshold),
                        xytext=(conv_tick + total_ticks * 0.01, threshold * 1.5),
                        fontsize=12, color=COLORS[variant])

    ax.set_title(f"λ={lam}", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    # ---- 지표 계산 (개선 = baseline(요청노드 포함) 대비 main(요청노드 제외)) ----
    base_m = metrics[BASE_KEY]
    main_m = metrics[MAIN_KEY]
    if base_m['conv_tick'] is not None and main_m['conv_tick'] is not None:
        speedup = base_m['conv_tick'] / main_m['conv_tick']
        conv_disp = f"{speedup:.1f}x faster"
    else:
        conv_disp = "N/A"

    ss_imp    = pct_improvement(base_m['ss']['median'], main_m['ss']['median'])
    auc_imp   = pct_improvement(base_m['auc'],          main_m['auc'])
    worst_imp = pct_improvement(base_m['worst'],        main_m['worst'])
    p95_imp   = pct_improvement(base_m['p95'],          main_m['p95'])
    echo_imp  = pct_improvement(base_m['echo_cost'],    main_m['echo_cost'])
    cv_factor = base_m['ss']['cv'] / main_m['ss']['cv'] if main_m['ss']['cv'] > 0 else float('inf')

    base_col = '요청노드 포함'
    main_col = '요청노드 제외 (현재)'

    md.write(f"| 지표 | {base_col} | {main_col} | 개선 |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| [변형별] 정착 ADT 중앙값 (SS ADT median) | {base_m['ss']['median']:.2f} | {main_m['ss']['median']:.2f} | {ss_imp:+.1f}% |\n")
    md.write(f"| [변형별] 정착 ADT 중간 50% 범위 (IQR) | [{base_m['ss']['q25']:.2f}, {base_m['ss']['q75']:.2f}] | "
             f"[{main_m['ss']['q25']:.2f}, {main_m['ss']['q75']:.2f}] | - |\n")
    md.write(f"| [변형별] 누적 ADT (AUC) | {base_m['auc']:.0f} | {main_m['auc']:.0f} | {auc_imp:+.1f}% |\n")
    md.write(f"| [변형별] 최악 ADT (worst spike, 구간 최댓값) | {base_m['worst']:.2f} | {main_m['worst']:.2f} | {worst_imp:+.1f}% |\n")
    md.write(f"| [변형별] 정착 후 상위 5% 지연 (P95) | {base_m['p95']:.2f} | {main_m['p95']:.2f} | {p95_imp:+.1f}% |\n")
    md.write(f"| [변형별] 랜덤시드 간 변동성 (CV, 작을수록 일관적) | {base_m['ss']['cv']:.3f} | {main_m['ss']['cv']:.3f} | "
             f"{cv_factor:.1f}× |\n")
    md.write(f"| [변형별] 결정당 에코 이웃 수 (echo cost) | {base_m['echo_cost']:.2f} | {main_m['echo_cost']:.2f} | {echo_imp:+.1f}% |\n")
    md.write(f"| [기준] 기준선 도달 시간 (convergence time, tick) | {base_m['conv_tick']} | {main_m['conv_tick']} | {conv_disp} |\n")
    md.write(f"* 기준선 : 요청노드 제외(현재) 정착 ADT 중앙값보다 20% 높은 값 (× {CONVERGENCE_THRESHOLD_RATIO})\n\n")

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

    print(f"\n  [기준선] = {threshold:.2f}")
    print(f"  {'지표':<26} | {base_col:>14} | {main_col:>16} | {'개선':>15}")
    print(f"  {'-'*26}-+-{'-'*14}-+-{'-'*16}-+-{'-'*15}")
    print("  [변형별]")
    print(f"  {'정착 ADT 중앙값':<26} | {base_m['ss']['median']:>14.2f} | {main_m['ss']['median']:>16.2f} | {ss_imp:>+14.1f}%")
    print(f"  {'누적 ADT':<26} | {base_m['auc']:>14.0f} | {main_m['auc']:>16.0f} | {auc_imp:>+14.1f}%")
    print(f"  {'최악 ADT (구간 최댓값)':<26} | {base_m['worst']:>14.2f} | {main_m['worst']:>16.2f} | {worst_imp:>+14.1f}%")
    print(f"  {'정착 후 상위 5% 지연':<26} | {base_m['p95']:>14.2f} | {main_m['p95']:>16.2f} | {p95_imp:>+14.1f}%")
    print(f"  {'랜덤시드 간 변동성':<26} | {base_m['ss']['cv']:>14.3f} | {main_m['ss']['cv']:>16.3f} | {cv_factor:>13.1f}×")
    print(f"  {'결정당 에코 이웃 수':<26} | {base_m['echo_cost']:>14.2f} | {main_m['echo_cost']:>16.2f} | {echo_imp:>+14.1f}%")
    print("  [기준]")
    print(f"  {'기준선 도달 시간 (tick)':<26} | {str(base_m['conv_tick']):>14} | {str(main_m['conv_tick']):>16} | {conv_disp:>15}")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 12), squeeze=False)
    axes = axes.flatten()
    active_labels = ' vs '.join(LABELS[v] for v in VARIANTS)
    fig.suptitle(
        f"6x6 Grid — AQPRICE requester-exclude ablation : {active_labels}  "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)})",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE 요청 노드 제외 ablation (이웃 최소 Q 계산에서 질문자 제외 유무)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS} (라우트 메모리는 둘 다 없음 = 기본 L 0)\n')
        md.write(f'- 변형별 aqprice_exclude_requester: include=False, exclude=True(현재 기본)\n')
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 지표 설명\n\n')
        md.write('- [변형별] 지표는 각 변형(포함 / 제외)이 자기 자신의 곡선으로 독립 계산한다. '
                 '[기준] 지표(기준선 도달 시간)만 요청노드 제외(현재 기본) 정착 ADT 중앙값보다 20% 높은 값(× 1.2)을 공통 판정선으로 쓴다.\n\n')
        md.write('- 비교 대상 : 이웃이 목적지까지 최소 Q 를 계산해 돌려줄 때, 방금 질문한 노드(x)를 후보에서 뺄지 여부.\n')
        md.write('  exclude(현재 기본)는 x 를 빼고(AQRERM 라우트 메모리에서 온 조건), include 는 x 도 포함(Q-routing/AQFE 와 동일).\n')
        md.write('  라우트 메모리(L)는 두 변형 모두 없음 → 이 표의 차이는 오직 요청 노드 제외 여부에서 온다.\n\n')
        md.write('- [변형별] 정착 ADT 중앙값 (SS ADT median) : 학습이 끝난 뒤쪽 절반 구간 평균 전달시간의 시드 간 중앙값. 작을수록 좋음.\n')
        md.write('- [변형별] 정착 ADT 중간 50% 범위 (IQR) : 시드별 정착 ADT 의 25~75% 구간. 시드 간 퍼짐.\n')
        md.write('- [변형별] 누적 ADT (AUC) : 실행 전체 구간 ADT 합의 시드 간 중앙값. 과도기와 정착 수준을 함께 반영.\n')
        md.write('- [변형별] 최악 ADT (worst spike) : 실행 중 가장 높았던 구간 ADT(최댓값)의 시드 간 중앙값.\n')
        md.write('- [변형별] 정착 후 상위 5% 지연 (P95) : 정착 구간 ADT 의 95퍼센타일. 단발 이상치를 뺀 통상적 고지연 수준.\n')
        md.write('- [변형별] 랜덤시드 간 변동성 (CV) : 정착 ADT 의 시드 간 (표준편차 ÷ 평균). 작을수록 일관적.\n')
        md.write('- [변형별] 결정당 에코 이웃 수 (echo cost) : 라우팅 결정 1회당 이웃 조회 평균 횟수. '
                 '요청 노드 제외는 조회 수는 안 바꾸므로 두 변형이 거의 같아야 정상.\n')
        md.write('- [기준] 기준선 도달 시간 (convergence time) : ADT 가 공통 기준선을 안정적으로 하회한 tick. 작을수록 빨리 도달.\n\n')
        md.write('- 패킷 전달 통계 : 생성/전달/미전달은 시드 평균 패킷 수. '
                 '성공률은 전달/생성(시드 평균), 최고/최저는 시드 중 최대/최소 성공률.\n\n')

        md.write('> 개선 열은 baseline(요청노드 포함) 대비 main(요청노드 제외, 현재 기본) 기준. '
                 '양수면 요청 노드 제외가 그 지표를 개선.\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            run_lambda(ax, exp['lam'], exp['total_ticks'], md)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")


if __name__ == '__main__':
    run_all()
