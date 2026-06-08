"""
발표용 최종 비교 — aqrerm vs pfe_c_pre_echo_tick.

5 가지 metric :
  1. 수렴 시간 (convergence time) : ADT 가 처음으로 threshold 아래로 내려간 tick
  2. Steady-state ADT median        : 뒷쪽 절반 평균, 시드 간 median
  3. 누적 AUC                       : 전체 ADT 시계열의 합 (시드 간 median)
  4. Worst-case spike               : 시드별 max ADT, 시드 간 median
  5. CV (시드 일관성)               : SS ADT 의 std / mean

threshold 는 PCPET 의 SS median × 1.2 (양 알고리즘에 공통 적용).

main_compare_PFE.py 와 별도 파일이며 기존 파일은 건드리지 않음.
"""
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (main_compare_PFE 와 동일)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# 비교 대상 2 개
ALGORITHMS = ['aqrerm', 'pfe_c_pre_echo_tick']
LABELS = {
    'aqrerm':              'AQRERM',
    'pfe_c_pre_echo_tick': 'PFE_c_pre_echo_tick',
}
COLORS = {
    'aqrerm':              '#CC79A7',  # 분홍보라 (baseline)
    'pfe_c_pre_echo_tick': '#56B4E9',  # 하늘색 (메인 후보)
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100
TOTAL_TICKS = 20000

# threshold 비율 (수렴 시간 정의에 사용)
CONVERGENCE_THRESHOLD_RATIO = 1.2

MD_PATH  = 'result_compare_PFE_final.md'
PNG_PATH = 'result_compare_PFE_final.png'

EXPERIMENTS = [
    {'lam': 2, 'total_ticks': TOTAL_TICKS, 'title': 'λ=2.0'},
    {'lam': 3.5, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.5'},
    # {'lam': 3.8, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.8'},
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
def compute_convergence_time(median_series, threshold, x_axis, min_fraction=0.95):
    """median ADT 시계열이 threshold 아래로 '안정적으로' 유지되기 시작한 tick 반환.
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
    per_seed_auc = np.sum(adt_arr, axis=1)
    return float(np.median(per_seed_auc)), per_seed_auc


def compute_worst_spike(adt_arr):
    """시드별 max ADT → 시드 간 median 반환."""
    per_seed_max = np.max(adt_arr, axis=1)
    return float(np.median(per_seed_max)), per_seed_max


def compute_ss_metrics(adt_arr):
    """SS ADT (뒷쪽 절반 평균) per seed → median / mean / std / CV / IQR / range."""
    half = adt_arr.shape[1] // 2
    ss_per_seed = np.mean(adt_arr[:, half:], axis=1)
    return {
        'per_seed': ss_per_seed,
        'median': float(np.median(ss_per_seed)),
        'mean':   float(np.mean(ss_per_seed)),
        'std':    float(np.std(ss_per_seed)),
        'cv':     float(np.std(ss_per_seed) / np.mean(ss_per_seed)) if np.mean(ss_per_seed) > 0 else 0.0,
        'q25':    float(np.percentile(ss_per_seed, 25)),
        'q75':    float(np.percentile(ss_per_seed, 75)),
        'min':    float(np.min(ss_per_seed)),
        'max':    float(np.max(ss_per_seed)),
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
        for seed in SEEDS:
            print(f"  Running seed={seed}...")
            sim, adt = run_one(algo, lam, total_ticks, seed)
            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            adt_runs.append(adt)
        adt_arr = np.array(adt_runs)
        results[algo] = {
            'adt_arr':  adt_arr,
            'median':   np.median(adt_arr, axis=0),
            'q25':      np.percentile(adt_arr, 25, axis=0),
            'q75':      np.percentile(adt_arr, 75, axis=0),
        }

    # ---- threshold = PCPET 의 SS median × 1.2 ----
    pcpet_ss = compute_ss_metrics(results['pfe_c_pre_echo_tick']['adt_arr'])
    threshold = pcpet_ss['median'] * CONVERGENCE_THRESHOLD_RATIO

    # ---- 알고리즘별 5 가지 metric 계산 ----
    metrics = {}
    for algo in ALGORITHMS:
        adt_arr = results[algo]['adt_arr']
        median_series = results[algo]['median']
        ss = compute_ss_metrics(adt_arr)
        auc_median, _ = compute_auc(adt_arr)
        worst_median, _ = compute_worst_spike(adt_arr)
        conv_tick = compute_convergence_time(median_series, threshold, x_axis)
        metrics[algo] = {
            'ss':         ss,
            'auc':        auc_median,
            'worst':      worst_median,
            'conv_tick':  conv_tick,
        }

    # ---- 시각화 : median 실선 + IQR 음영 ----
    for algo in ALGORITHMS:
        ax.plot(x_axis, results[algo]['median'],
                label=LABELS[algo], color=COLORS[algo], linewidth=2.0)
        ax.fill_between(x_axis, results[algo]['q25'], results[algo]['q75'],
                        color=COLORS[algo], alpha=0.2)

    # ---- threshold 수평선 ----
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.2,
               label=f'threshold = PCPET SS × {CONVERGENCE_THRESHOLD_RATIO} ({threshold:.2f})')

    # ---- 알고리즘별 수렴 시점 vertical line ----
    for algo in ALGORITHMS:
        conv_tick = metrics[algo]['conv_tick']
        if conv_tick is not None:
            ax.axvline(x=conv_tick, color=COLORS[algo], linestyle=':',
                       linewidth=1.5, alpha=0.7)
            ax.annotate(f'{LABELS[algo]}\nconv: {conv_tick}',
                        xy=(conv_tick, threshold),
                        xytext=(conv_tick + total_ticks * 0.01, threshold * 1.5),
                        fontsize=8, color=COLORS[algo])

    # ---- 텍스트 박스 (핵심 metric 5 개) ----
    aqrerm_m = metrics['aqrerm']
    pcpet_m  = metrics['pfe_c_pre_echo_tick']

    # 수렴 시간 비교 (None 처리)
    if aqrerm_m['conv_tick'] is not None and pcpet_m['conv_tick'] is not None:
        speedup = aqrerm_m['conv_tick'] / pcpet_m['conv_tick']
        conv_str = f"{aqrerm_m['conv_tick']} -> {pcpet_m['conv_tick']} ({speedup:.1f}x faster)"
    else:
        conv_str = "(some seeds not reached)"

    ss_imp    = pct_improvement(aqrerm_m['ss']['median'],   pcpet_m['ss']['median'])
    auc_imp   = pct_improvement(aqrerm_m['auc'],            pcpet_m['auc'])
    worst_imp = pct_improvement(aqrerm_m['worst'],          pcpet_m['worst'])
    cv_factor = aqrerm_m['ss']['cv'] / pcpet_m['ss']['cv'] if pcpet_m['ss']['cv'] > 0 else float('inf')

    text = (
        f"PCPET vs AQRERM (lam={lam})\n"
        f"---------------------\n"
        f"Conv time : {conv_str}\n"
        f"SS ADT    : {aqrerm_m['ss']['median']:.2f} -> {pcpet_m['ss']['median']:.2f}  ({ss_imp:+.1f}%)\n"
        f"AUC       : {aqrerm_m['auc']:.0f} -> {pcpet_m['auc']:.0f}  ({auc_imp:+.1f}%)\n"
        f"Worst max : {aqrerm_m['worst']:.2f} -> {pcpet_m['worst']:.2f}  ({worst_imp:+.1f}%)\n"
        f"CV        : {aqrerm_m['ss']['cv']:.3f} -> {pcpet_m['ss']['cv']:.3f}  ({cv_factor:.1f}x more consistent)"
    )
    ax.text(0.98, 0.97, text,
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    ax.set_title(f"λ={lam}")
    ax.set_xlabel('Simulator Time')
    ax.set_ylabel('Average Delivery Time')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- MD 로그 ----
    md.write(f"### Threshold : PCPET SS median × {CONVERGENCE_THRESHOLD_RATIO} = **{threshold:.2f}**\n\n")
    md.write("| metric | AQRERM | PCPET | 개선 |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| 수렴 시간 (tick) | {aqrerm_m['conv_tick']} | {pcpet_m['conv_tick']} | "
             f"{conv_str.split('(')[-1].rstrip(')') if '×' in conv_str else 'N/A'} |\n")
    md.write(f"| SS ADT median | {aqrerm_m['ss']['median']:.2f} | {pcpet_m['ss']['median']:.2f} | {ss_imp:+.1f}% |\n")
    md.write(f"| SS ADT IQR | [{aqrerm_m['ss']['q25']:.2f}, {aqrerm_m['ss']['q75']:.2f}] | "
             f"[{pcpet_m['ss']['q25']:.2f}, {pcpet_m['ss']['q75']:.2f}] | - |\n")
    md.write(f"| AUC median | {aqrerm_m['auc']:.0f} | {pcpet_m['auc']:.0f} | {auc_imp:+.1f}% |\n")
    md.write(f"| Worst spike median | {aqrerm_m['worst']:.2f} | {pcpet_m['worst']:.2f} | {worst_imp:+.1f}% |\n")
    md.write(f"| CV (시드 일관성) | {aqrerm_m['ss']['cv']:.3f} | {pcpet_m['ss']['cv']:.3f} | "
             f"{cv_factor:.1f}× 일관 |\n\n")

    # ---- 콘솔 출력 ----
    print(f"\n  [Threshold] = {threshold:.2f}")
    print(f"  {'metric':<22} | {'AQRERM':>12} | {'PCPET':>12} | {'개선':>15}")
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")
    print(f"  {'수렴 시간 (tick)':<22} | {str(aqrerm_m['conv_tick']):>12} | {str(pcpet_m['conv_tick']):>12} | "
          f"{conv_str.split('(')[-1].rstrip(')') if '×' in conv_str else 'N/A':>15}")
    print(f"  {'SS ADT median':<22} | {aqrerm_m['ss']['median']:>12.2f} | {pcpet_m['ss']['median']:>12.2f} | {ss_imp:>+14.1f}%")
    print(f"  {'AUC median':<22} | {aqrerm_m['auc']:>12.0f} | {pcpet_m['auc']:>12.0f} | {auc_imp:>+14.1f}%")
    print(f"  {'Worst spike median':<22} | {aqrerm_m['worst']:>12.2f} | {pcpet_m['worst']:>12.2f} | {worst_imp:>+14.1f}%")
    print(f"  {'CV (시드 일관성)':<22} | {aqrerm_m['ss']['cv']:>12.3f} | {pcpet_m['ss']['cv']:>12.3f} | {cv_factor:>13.1f}× 일관")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 12), squeeze=False)
    axes = axes.flatten()
    active_labels = ' vs '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"6x6 Grid — Final comparison : {active_labels}  "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)}, 5 metric)",
        fontsize=14,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write(f'# Final comparison : {active_labels}\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in ALGORITHMS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n')
        md.write(f'- Convergence threshold = PCPET SS median × {CONVERGENCE_THRESHOLD_RATIO}\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            run_lambda(ax, exp['lam'], exp['total_ticks'], md)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")


if __name__ == '__main__':
    run_all()
