# =============================================================================
# [요약] AQPRICE 되돌아가기 방지 방식 비교 — 3 변형
#   inc : L=0, 요청노드 포함 (aqprice_exclude_requester=False) → 방지 없음
#   exc : L=0, 요청노드 제외 (aqprice_exclude_requester=True)  → 추정값 단계 방지 (현재 기본)
#   L1  : L=1, 요청노드 제외 (aqprice_L=1, exclude=True)        → 라우트 메모리로 직전 노드 방지
# - node.py 의 aqprice_L / aqprice_exclude_requester 두 토글을 변형별로 조합
# - 6x6 grid, 10 시드, λ=2/3.5/3.8 (20000 tick). 지표표 + 그래프 생산
# - 기준선 = exc(현재 기본) 정착 ADT 중앙값보다 20% 높은 값 (× 1.2, 세 변형 공통)
# - 산출물: result_compare_AQPRICE_backtrack.md / .png
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
# 파라미터 (aqprice_L / aqprice_exclude_requester 는 변형별로 덮어씀)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 3 변형 정의: (aqprice_L, aqprice_exclude_requester)
# -------------------------------------------------------------------------
VARIANTS = ['inc', 'exc', 'L1']
REF_KEY  = 'exc'   # 현재 기본. 기준선·표기 기준
VARIANT_CFG = {
    'inc': {'aqprice_L': 0, 'aqprice_exclude_requester': False},
    'exc': {'aqprice_L': 0, 'aqprice_exclude_requester': True},
    'L1':  {'aqprice_L': 1, 'aqprice_exclude_requester': True},
}
LABELS = {
    'inc': 'AQPRICE (L=0, 요청노드 포함)',
    'exc': 'AQPRICE (L=0, 요청노드 제외 / 현재)',
    'L1':  'AQPRICE (L=1, 요청노드 제외)',
}
LABELS_SHORT = {
    'inc': 'L0 포함',
    'exc': 'L0 제외(현재)',
    'L1':  'L1 제외',
}
COLORS = {
    'inc': '#D55E00',  # 주홍
    'exc': '#56B4E9',  # 하늘색 (현재 기본)
    'L1':  '#E69F00',  # 주황
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100
TOTAL_TICKS = 20000

CONVERGENCE_THRESHOLD_RATIO = 1.2

MD_PATH  = 'result_compare_AQPRICE_backtrack.md'
PNG_PATH = 'result_compare_AQPRICE_backtrack.png'

EXPERIMENTS = [
    {'lam': 2, 'total_ticks': TOTAL_TICKS, 'title': 'λ=2.0'},
    {'lam': 3.5, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.5'},
    {'lam': 3.8, 'total_ticks': TOTAL_TICKS, 'title': 'λ=3.8'},
]


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 항상 aqprice, 두 토글만 변형별로 교체
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
# Metric 계산 함수들 (main_aqrerm_vs_aqprice 와 동일)
# -------------------------------------------------------------------------
def compute_target_reach_time(median_series, threshold, x_axis, min_fraction=0.95):
    median_series = np.asarray(median_series)
    n = len(median_series)
    for i in range(n):
        rest = median_series[i:]
        if np.mean(rest <= threshold) >= min_fraction:
            return x_axis[i]
    return None


def compute_auc(adt_arr):
    per_seed_auc = np.nansum(adt_arr, axis=1)
    return float(np.nanmedian(per_seed_auc)), per_seed_auc


def compute_worst_spike(adt_arr):
    per_seed_max = np.nanmax(adt_arr, axis=1)
    return float(np.nanmedian(per_seed_max)), per_seed_max


def compute_late_p95(adt_arr):
    half = adt_arr.shape[1] // 2
    per_seed_p95 = np.nanpercentile(adt_arr[:, half:], 95, axis=1)
    return float(np.nanmedian(per_seed_p95)), per_seed_p95


def compute_ss_metrics(adt_arr):
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

    # ---- 기준선 = REF(exc) 정착 ADT 중앙값 × 1.2 ----
    ref_ss = compute_ss_metrics(results[REF_KEY]['adt_arr'])
    threshold = ref_ss['median'] * CONVERGENCE_THRESHOLD_RATIO

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
               label=f'reference = exc steady ADT x {CONVERGENCE_THRESHOLD_RATIO} ({threshold:.2f})')
    for variant in VARIANTS:
        conv_tick = metrics[variant]['conv_tick']
        if conv_tick is not None:
            ax.axvline(x=conv_tick, color=COLORS[variant], linestyle=':',
                       linewidth=1.5, alpha=0.7)

    ax.set_title(f"λ={lam}", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- MD 표 (변형별 열, 원값) ----
    cols = " | ".join(LABELS_SHORT[v] for v in VARIANTS)
    align = "|---" + "|---:" * len(VARIANTS) + "|\n"

    def row(name, fmt):
        return f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS) + " |\n"

    md.write(f"| 지표 | {cols} |\n")
    md.write(align)
    md.write(row("[변형별] 정착 ADT 중앙값 (SS ADT median)",
                 lambda v: f"{metrics[v]['ss']['median']:.2f}"))
    md.write(row("[변형별] 정착 ADT 중간 50% 범위 (IQR)",
                 lambda v: f"[{metrics[v]['ss']['q25']:.2f}, {metrics[v]['ss']['q75']:.2f}]"))
    md.write(row("[변형별] 누적 ADT (AUC)",
                 lambda v: f"{metrics[v]['auc']:.0f}"))
    md.write(row("[변형별] 최악 ADT (worst spike, 구간 최댓값)",
                 lambda v: f"{metrics[v]['worst']:.2f}"))
    md.write(row("[변형별] 정착 후 상위 5% 지연 (P95)",
                 lambda v: f"{metrics[v]['p95']:.2f}"))
    md.write(row("[변형별] 랜덤시드 간 변동성 (CV, 작을수록 일관적)",
                 lambda v: f"{metrics[v]['ss']['cv']:.3f}"))
    md.write(row("[변형별] 결정당 에코 이웃 수 (echo cost)",
                 lambda v: f"{metrics[v]['echo_cost']:.2f}"))
    md.write(row("[기준] 기준선 도달 시간 (convergence time, tick)",
                 lambda v: f"{metrics[v]['conv_tick']}"))
    md.write(f"* 기준선 : L0 제외(현재) 정착 ADT 중앙값보다 20% 높은 값 (× {CONVERGENCE_THRESHOLD_RATIO})\n\n")

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

    # ---- 콘솔 출력 ----
    print(f"\n  [기준선] = {threshold:.2f}")
    head = " | ".join(f"{LABELS_SHORT[v]:>14}" for v in VARIANTS)
    print(f"  {'지표':<26} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>14}" for v in VARIANTS)
        print(f"  {name:<26} | {vals}")

    prow("정착 ADT 중앙값",      lambda v: f"{metrics[v]['ss']['median']:.2f}")
    prow("누적 ADT",             lambda v: f"{metrics[v]['auc']:.0f}")
    prow("최악 ADT (구간 최댓값)", lambda v: f"{metrics[v]['worst']:.2f}")
    prow("정착 후 상위 5% 지연",  lambda v: f"{metrics[v]['p95']:.2f}")
    prow("랜덤시드 간 변동성",    lambda v: f"{metrics[v]['ss']['cv']:.3f}")
    prow("결정당 에코 이웃 수",   lambda v: f"{metrics[v]['echo_cost']:.2f}")
    prow("기준선 도달 시간",      lambda v: f"{metrics[v]['conv_tick']}")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 12), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        f"6x6 Grid — AQPRICE backtrack-guard compare (L0 포함 / L0 제외 / L1)  "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)})",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE 되돌아가기 방지 방식 비교 (없음 / 요청노드 제외 / 라우트 메모리 L=1)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- 변형 설정 (aqprice_L, aqprice_exclude_requester):\n')
        for v in VARIANTS:
            cfg = VARIANT_CFG[v]
            md.write(f"  - {v} : L={cfg['aqprice_L']}, 요청노드 제외={cfg['aqprice_exclude_requester']}\n")
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 지표 설명\n\n')
        md.write('- 세 변형은 "패킷이 방금 온 곳으로 되돌아가는 것" 을 막는 방식이 다르다.\n')
        md.write('  inc(L0 포함) : 아무 방지 없음. exc(L0 제외/현재) : 이웃이 최소 Q 낼 때 질문자를 뺌(추정값 단계).\n')
        md.write('  L1 : 라우트 메모리로 직전 노드를 다음 홉 후보에서 뺌(선택 단계). L1 은 요청노드 제외도 유지.\n\n')
        md.write('- 표 값은 모두 시드 집계 원값(개선 열 없음). 기준은 exc(현재)로, 기준선 도달 시간만 exc 정착 ADT × 1.2 를 공통 판정선으로 쓴다.\n')
        md.write('- 정착 ADT 중앙값 : 뒤쪽 절반 평균의 시드 간 중앙값(작을수록 좋음). '
                 '누적 ADT : 전체 구간 합의 중앙값. 최악 ADT : 구간 최댓값의 중앙값. '
                 'P95 : 정착 구간 95퍼센타일. CV : 정착 ADT 의 시드 간 표준편차/평균(작을수록 일관적). '
                 '에코 이웃 수 : 결정당 이웃 조회 평균. 기준선 도달 시간 : 공통 기준선을 안정적으로 하회한 tick.\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            run_lambda(ax, exp['lam'], exp['total_ticks'], md)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")


if __name__ == '__main__':
    run_all()
