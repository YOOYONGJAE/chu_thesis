# =============================================================================
# [요약] 발표용 최종 지표 비교 — AQRERM vs AQPRICE 정량 지표 요약표
# - 6x6 grid, 10 시드, λ=2/3.5 (20000 tick). 그래프가 아니라 지표표 생산이 목적
# - 지표:
# ① AQPRICE 기준선 도달 시간 (ADT 가 기준선을 안정적으로 하회한 tick)
# ② 정착 ADT 중앙값 (뒤쪽 절반 평균의 시드 간 중앙값)
# ③ 정착 ADT 중간 50% 범위 (시드 간 25~75% 구간)
# ④ 누적 ADT (시드별 시계열 합의 시드 간 중앙값)
# ⑤ 최악 ADT (구간 최댓값의 시드 간 중앙값)
# ⑥ 정착 후 상위 5% 지연 (뒤쪽 절반 95퍼센타일의 시드 간 중앙값)
# ⑦ 랜덤시드 간 변동성 (정착 ADT 의 시드 간 표준편차 / 평균)
# ⑧ 결정당 에코 이웃 수 (이웃 조회 총횟수 / 라우팅 결정 총횟수)
# - 기준선 = AQPRICE 정착 ADT 중앙값 × 1.2 (양 알고리즘에 공통 적용)
# - 산출물: result_compare_AQPRICE_final.md / .png
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (main_compare_AQPRICE 와 동일)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# 비교 대상 2 개
ALGORITHMS = ['aqrerm', 'aqprice']
LABELS = {
    'aqrerm':              'AQRERM',
    'aqprice': 'AQPRICE',
}
COLORS = {
    'aqrerm':              "#FF0000",  # 분홍보라 (baseline)
    'aqprice': '#56B4E9',  # 하늘색 (메인 후보)
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100
TOTAL_TICKS = 20000

# threshold 비율 (수렴 시간 정의에 사용)
CONVERGENCE_THRESHOLD_RATIO = 1.2

MD_PATH  = 'result_compare_AQPRICE_final.md'
PNG_PATH = 'result_compare_AQPRICE_final.png'

EXPERIMENTS = [
    {'lam': 2, 'total_ticks': TOTAL_TICKS, 'title': 'λ=2.0'},
    {'lam': 3.5, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.5'},
    {'lam': 3.8, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.8'},
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
# -------------------------------------------------------------------------
def compute_target_reach_time(median_series, threshold, x_axis, min_fraction=0.95):
    """median ADT 시계열이 기준선(threshold) 아래로 '안정적으로' 유지되기 시작한 tick 반환.
    조건: 그 시점 이후의 윈도우 중 min_fraction (기본 95%) 이상이 threshold 이하.
    못 도달하면 None.

    단순히 '처음으로 threshold 아래로 내려간 tick' 을 쓰면 첫 윈도우의 인위적 낮은
    ADT 값 (가까운 거리 패킷 1~2 개만 배달되어 평균이 작음) 에 속을 수 있어서
    '이후 95% 가 안정적으로 유지' 조건으로 보완.
    """
    median_series = np.asarray(median_series)
    n = len(median_series)
    for i in range(n):
        rest = median_series[i:]
        if np.mean(rest <= threshold) >= min_fraction:
            return x_axis[i]
    return None


def compute_auc(adt_arr):
    """시드별 ADT 시계열의 합 → 시드 간 median 반환."""
    # 전달 0 구간(NaN)은 제외하고 합/집계 (nan 무시 계열)
    per_seed_auc = np.nansum(adt_arr, axis=1)
    return float(np.nanmedian(per_seed_auc)), per_seed_auc


def compute_worst_spike(adt_arr):
    """시드별 max ADT → 시드 간 median 반환 (최악 ADT, 구간 최댓값)."""
    per_seed_max = np.nanmax(adt_arr, axis=1)
    return float(np.nanmedian(per_seed_max)), per_seed_max


def compute_late_p95(adt_arr):
    """각 시드의 뒤쪽 절반 구간 95퍼센타일 → 시드 간 median 반환 (정착 후 상위 5% 지연).
    최댓값 하나에 휘둘리는 최악 ADT 와 달리, 단발 이상치를 걸러낸 통상적 고지연 수준."""
    half = adt_arr.shape[1] // 2
    per_seed_p95 = np.nanpercentile(adt_arr[:, half:], 95, axis=1)
    return float(np.nanmedian(per_seed_p95)), per_seed_p95


def compute_ss_metrics(adt_arr):
    """SS ADT (뒷쪽 절반 평균) per seed → median / mean / std / CV / IQR / range."""
    # NaN 구간을 제외하고 집계 (nan 무시 계열)
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

    # ---- 알고리즘별 시뮬레이션 수집 ----
    results = {}
    for algo in ALGORITHMS:
        label = LABELS[algo]
        print(f"\n--- {label} ---")
        adt_runs = []
        echo_costs = []
        for seed in SEEDS:
            print(f"  Running seed={seed}...")
            sim, adt = run_one(algo, lam, total_ticks, seed)
            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            adt_runs.append(adt)
            # 결정당 에코 이웃 수 (이 시드 실행 전체 누적)
            ec = (sim.total_echo_queries / sim.total_route_calls
                  if sim.total_route_calls > 0 else 0.0)
            echo_costs.append(ec)
        adt_arr = np.array(adt_runs)
        results[algo] = {
            'adt_arr':   adt_arr,
            'median':    np.nanmedian(adt_arr, axis=0),
            'q25':       np.nanpercentile(adt_arr, 25, axis=0),
            'q75':       np.nanpercentile(adt_arr, 75, axis=0),
            'echo_cost': float(np.nanmedian(echo_costs)),   # 시드 간 중앙값
        }

    # ---- threshold = AQPRICE 의 SS median × 1.2 ----
    aqprice_ss = compute_ss_metrics(results['aqprice']['adt_arr'])
    threshold = aqprice_ss['median'] * CONVERGENCE_THRESHOLD_RATIO

    # ---- 알고리즘별 지표 계산 ----
    metrics = {}
    for algo in ALGORITHMS:
        adt_arr = results[algo]['adt_arr']
        median_series = results[algo]['median']
        ss = compute_ss_metrics(adt_arr)
        auc_median, _ = compute_auc(adt_arr)
        worst_median, _ = compute_worst_spike(adt_arr)
        p95_median, _ = compute_late_p95(adt_arr)
        conv_tick = compute_target_reach_time(median_series, threshold, x_axis)
        metrics[algo] = {
            'ss':         ss,
            'auc':        auc_median,
            'worst':      worst_median,
            'p95':        p95_median,
            'conv_tick':  conv_tick,
            'echo_cost':  results[algo]['echo_cost'],
        }

    # ---- 시각화 : median 실선 + IQR 오차 막대 (일정 간격 세로선) ----
    # spacing : 전체 윈도우 수를 10등분 → 알고리즘당 막대 약 10개
    #           (200 윈도우 기준 20 윈도우 = 2000 tick 간격)
    # offset  : 알고리즘마다 시작점을 spacing/알고리즘수 만큼 어긋내서
    #           서로 다른 알고리즘의 막대가 같은 x 위치에 겹치지 않게 함
    spacing = max(1, len(x_axis) // 10)
    for idx, algo in enumerate(ALGORITHMS):
        median = results[algo]['median']
        # yerr 는 (아래 길이, 위 길이) 두 행 — median 기준 비대칭 IQR 범위
        yerr = np.vstack([median - results[algo]['q25'],
                          results[algo]['q75'] - median])
        offset = idx * spacing // len(ALGORITHMS)
        ax.errorbar(x_axis, median, yerr=yerr,
                    errorevery=(offset, spacing),
                    capsize=3, elinewidth=1.2, capthick=1.2,
                    label=LABELS[algo], color=COLORS[algo], linewidth=2.0)

    # ---- 기준선 수평선 (그래프 텍스트는 Hangul tofu 방지 위해 ASCII 유지) ----
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.2,
               label=f'reference = AQPRICE steady ADT x {CONVERGENCE_THRESHOLD_RATIO} ({threshold:.2f})')

    # ---- 알고리즘별 기준선 도달 시점 vertical line ----
    for algo in ALGORITHMS:
        conv_tick = metrics[algo]['conv_tick']
        if conv_tick is not None:
            ax.axvline(x=conv_tick, color=COLORS[algo], linestyle=':',
                       linewidth=1.5, alpha=0.7)
            ax.annotate(f'{LABELS[algo]}\nreach: {conv_tick}',
                        xy=(conv_tick, threshold),
                        xytext=(conv_tick + total_ticks * 0.01, threshold * 1.5),
                        fontsize=12, color=COLORS[algo])

    # ---- 텍스트 박스 (핵심 지표. 그래프 안이라 ASCII 라벨 유지) ----
    aqrerm_m = metrics['aqrerm']
    aqprice_m  = metrics['aqprice']

    # 기준선 도달 시간 비교 (None 처리) — 표/콘솔용 개선 문자열
    if aqrerm_m['conv_tick'] is not None and aqprice_m['conv_tick'] is not None:
        speedup = aqrerm_m['conv_tick'] / aqprice_m['conv_tick']
        conv_disp = f"{speedup:.1f}x faster"
    else:
        conv_disp = "N/A"

    ss_imp    = pct_improvement(aqrerm_m['ss']['median'],   aqprice_m['ss']['median'])
    auc_imp   = pct_improvement(aqrerm_m['auc'],            aqprice_m['auc'])
    worst_imp = pct_improvement(aqrerm_m['worst'],          aqprice_m['worst'])
    p95_imp   = pct_improvement(aqrerm_m['p95'],            aqprice_m['p95'])
    echo_imp  = pct_improvement(aqrerm_m['echo_cost'],      aqprice_m['echo_cost'])
    cv_factor = aqrerm_m['ss']['cv'] / aqprice_m['ss']['cv'] if aqprice_m['ss']['cv'] > 0 else float('inf')

    # 수치는 그래프에 그리지 않고 MD 파일로만 출력 (아래 MD 로그 블록).
    # 범례를 (수치 박스가 있던) 우상단으로 옮기고 글씨를 키움.
    ax.set_title(f"λ={lam}", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    # ---- MD 로그 ----
    md.write(f"### 기준선 : AQPRICE 정착 ADT 중앙값 × {CONVERGENCE_THRESHOLD_RATIO} = **{threshold:.2f}**\n\n")
    md.write("| 지표 | AQRERM | AQPRICE | 개선 |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| AQPRICE 기준선 도달 시간 (tick) | {aqrerm_m['conv_tick']} | {aqprice_m['conv_tick']} | {conv_disp} |\n")
    md.write(f"| 정착 ADT 중앙값 | {aqrerm_m['ss']['median']:.2f} | {aqprice_m['ss']['median']:.2f} | {ss_imp:+.1f}% |\n")
    md.write(f"| 정착 ADT 중간 50% 범위 | [{aqrerm_m['ss']['q25']:.2f}, {aqrerm_m['ss']['q75']:.2f}] | "
             f"[{aqprice_m['ss']['q25']:.2f}, {aqprice_m['ss']['q75']:.2f}] | - |\n")
    md.write(f"| 누적 ADT | {aqrerm_m['auc']:.0f} | {aqprice_m['auc']:.0f} | {auc_imp:+.1f}% |\n")
    md.write(f"| 최악 ADT (구간 최댓값) | {aqrerm_m['worst']:.2f} | {aqprice_m['worst']:.2f} | {worst_imp:+.1f}% |\n")
    md.write(f"| 정착 후 상위 5% 지연 | {aqrerm_m['p95']:.2f} | {aqprice_m['p95']:.2f} | {p95_imp:+.1f}% |\n")
    md.write(f"| 랜덤시드 간 변동성 (작을수록 일관적) | {aqrerm_m['ss']['cv']:.3f} | {aqprice_m['ss']['cv']:.3f} | "
             f"{cv_factor:.1f}× |\n")
    md.write(f"| 결정당 에코 이웃 수 | {aqrerm_m['echo_cost']:.2f} | {aqprice_m['echo_cost']:.2f} | {echo_imp:+.1f}% |\n\n")

    # ---- 콘솔 출력 ----
    print(f"\n  [기준선] = {threshold:.2f}")
    print(f"  {'지표':<26} | {'AQRERM':>12} | {'AQPRICE':>12} | {'개선':>15}")
    print(f"  {'-'*26}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")
    print(f"  {'기준선 도달 시간 (tick)':<26} | {str(aqrerm_m['conv_tick']):>12} | {str(aqprice_m['conv_tick']):>12} | {conv_disp:>15}")
    print(f"  {'정착 ADT 중앙값':<26} | {aqrerm_m['ss']['median']:>12.2f} | {aqprice_m['ss']['median']:>12.2f} | {ss_imp:>+14.1f}%")
    print(f"  {'누적 ADT':<26} | {aqrerm_m['auc']:>12.0f} | {aqprice_m['auc']:>12.0f} | {auc_imp:>+14.1f}%")
    print(f"  {'최악 ADT (구간 최댓값)':<26} | {aqrerm_m['worst']:>12.2f} | {aqprice_m['worst']:>12.2f} | {worst_imp:>+14.1f}%")
    print(f"  {'정착 후 상위 5% 지연':<26} | {aqrerm_m['p95']:>12.2f} | {aqprice_m['p95']:>12.2f} | {p95_imp:>+14.1f}%")
    print(f"  {'랜덤시드 간 변동성':<26} | {aqrerm_m['ss']['cv']:>12.3f} | {aqprice_m['ss']['cv']:>12.3f} | {cv_factor:>13.1f}×")
    print(f"  {'결정당 에코 이웃 수':<26} | {aqrerm_m['echo_cost']:>12.2f} | {aqprice_m['echo_cost']:>12.2f} | {echo_imp:>+14.1f}%")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 12), squeeze=False)
    axes = axes.flatten()
    active_labels = ' vs '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"6x6 Grid — Final comparison : {active_labels}  "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)})",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write(f'# Final comparison : {active_labels}\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in ALGORITHMS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n')
        md.write(f'- Convergence threshold = AQPRICE SS median × {CONVERGENCE_THRESHOLD_RATIO}\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            run_lambda(ax, exp['lam'], exp['total_ticks'], md)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")


if __name__ == '__main__':
    run_all()
