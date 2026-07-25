# =============================================================================
# [요약] AQPRICE 라우트 메모리 길이 비교 — L=0 / 1 / 2 / 3
# - 같은 aqprice 를 라우트 메모리 길이만 다르게 비교 (요청노드 제외는 네 변형 모두 유지=기본)
#   L=0 : 라우트 메모리 미사용 (AQPRICE 정식 기본)
#   L=1/2/3 : 최근 방문 노드를 각각 1/2/3 개까지 다음 홉 후보에서 제외
# - node.py 는 aqprice_L 로 이 길이를 제어 (기본 0)
# - 6x6 grid, 10 시드, λ=3.8 만 (40000 tick). 지표표 + 그래프 생산
# - 5 지표: 앞/뒤 절반 평균 ADT / 최악 ADT / CV / 결정당 에코 이웃 수 (정착 구간 = 뒤 절반)
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
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (aqprice_L 만 변형별로 덮어씀)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 4 변형: 같은 aqprice, 라우트 메모리 길이만 다름 (요청노드 제외는 모두 기본 True 유지)
# -------------------------------------------------------------------------
VARIANTS = ['L0', 'L3']
REF_KEY  = 'L0'   # AQPRICE 정식 기본. 기준선·표기 기준
VARIANT_L = {'L0': 0, 'L3': 3}
LABELS = {
    'L0': 'AQPRICE (L=0, 라우트 메모리 미사용)',
    'L3': 'AQPRICE (L=3)',
}
LABELS_SHORT = {'L0': 'L=0(기본)', 'L3': 'L=3'}
COLORS = {
    'L0': '#56B4E9',  # 하늘색 (기본)
    # 'L1': '#E69F00',  # 주황
    # 'L2': '#009E73',  # 초록
    'L3': "#FF0000",  # 주홍
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100


MD_PATH  = 'result_aqprice_route_memory_L_changed.md'
PNG_PATH = 'result_aqprice_route_memory_L_changed.png'

# λ=3.8 만, 20000 tick
EXPERIMENTS = [
    {'lam': 3.8, 'total_ticks': 20000},
    {'lam': 4.0, 'total_ticks': 20000},
    {'lam': 4.4, 'total_ticks': 20000},
]


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
# Metric 계산 함수 — 5 지표 (앞/뒤 절반 평균 ADT / 최악 ADT / CV)
# - 정착 구간은 실행 시간을 정확히 반으로 나눈 뒤 절반으로 정의 (임의 tick 아님)
# -------------------------------------------------------------------------
def compute_metrics(adt_arr):
    """앞/뒤 절반 평균 ADT, 최악 ADT, CV 를 계산해 dict 로 반환."""
    half = adt_arr.shape[1] // 2
    front_ps = np.nanmean(adt_arr[:, :half], axis=1)
    back_ps  = np.nanmean(adt_arr[:, half:], axis=1)
    per_seed_max = np.nanmax(adt_arr, axis=1)
    back_mean = np.nanmean(back_ps)
    cv = float(np.nanstd(back_ps) / back_mean) if back_mean > 0 else 0.0
    return {
        'front': float(np.nanmedian(front_ps)),
        'back':  float(np.nanmedian(back_ps)),
        'worst': float(np.nanmedian(per_seed_max)),
        'cv':    cv,
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

    # ---- 변형별 지표 계산 (5 지표) ----
    metrics = {}
    for variant in VARIANTS:
        m = compute_metrics(results[variant]['adt_arr'])
        m['echo_cost'] = results[variant]['echo_cost']
        metrics[variant] = m

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

    ax.set_title(f"λ={lam} ({total_ticks} ticks)", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- MD 표 (변형별 열, 원값. 5 지표) ----
    cols = " | ".join(LABELS_SHORT[v] for v in VARIANTS)
    align = "|---" + "|---:" * len(VARIANTS) + "|\n"

    def row(name, fmt):
        return f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS) + " |\n"

    md.write(f"| 지표 | {cols} |\n")
    md.write(align)
    md.write(row("앞 절반 평균 ADT (학습 과도기)", lambda v: f"{metrics[v]['front']:.2f}"))
    md.write(row("뒤 절반 평균 ADT (정착 후)",   lambda v: f"{metrics[v]['back']:.2f}"))
    md.write(row("최악 ADT (구간 최댓값)",       lambda v: f"{metrics[v]['worst']:.2f}"))
    md.write(row("랜덤시드 간 변동성 (CV)",      lambda v: f"{metrics[v]['cv']:.3f}"))
    md.write(row("결정당 에코 이웃 수",          lambda v: f"{metrics[v]['echo_cost']:.2f}"))
    md.write("\n")

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
    head = " | ".join(f"{LABELS_SHORT[v]:>12}" for v in VARIANTS)
    print(f"  {'지표':<24} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>12}" for v in VARIANTS)
        print(f"  {name:<24} | {vals}")

    prow("앞 절반 평균 ADT", lambda v: f"{metrics[v]['front']:.2f}")
    prow("뒤 절반 평균 ADT", lambda v: f"{metrics[v]['back']:.2f}")
    prow("최악 ADT (구간 최댓값)", lambda v: f"{metrics[v]['worst']:.2f}")
    prow("랜덤시드 간 변동성", lambda v: f"{metrics[v]['cv']:.3f}")
    prow("결정당 에코 이웃 수", lambda v: f"{metrics[v]['echo_cost']:.2f}")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    # 패널 하나당 크기를 항상 동일하게 (20폭 × 패널수). 패널이 1개든 여러개든 각 패널 비율 유지.
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(20 * len(EXPERIMENTS), 12), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(
        f"6x6 Grid — AQPRICE Route Memory 길이 비교 (L=0/1/2/3)  "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)})",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE 라우트 메모리 길이 비교 (L=0 / 1 / 2 / 3)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS} (aqprice_L 만 변형별로 0/1/2/3, 요청노드 제외는 모두 유지)\n')
        md.write(f'- STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 지표 설명 (5 지표)\n\n')
        md.write('- 네 변형은 라우트 메모리 길이(L)만 다르다. L=0 은 라우트 메모리 미사용(AQPRICE 정식 기본), '
                 'L=1/2/3 은 최근 방문 노드를 각각 1/2/3 개까지 다음 홉 후보에서 제외한다. 요청노드 제외는 네 변형 모두 유지.\n')
        md.write('- 표 값은 모두 시드 집계 원값(개선 열 없음). 정착 구간은 실행 시간의 뒤 절반으로 정의 (임의 tick 아님).\n')
        md.write('- 앞 절반 평균 ADT (학습 과도기) / 뒤 절반 평균 ADT (정착 후) : 각 시드 절반 구간 평균 ADT 의 시드 간 중앙값.\n')
        md.write('- 최악 ADT : 각 시드 구간 최댓값의 중앙값. CV : 뒤 절반 평균 ADT 의 시드 간 표준편차/평균(작을수록 일관적). '
                 '에코 이웃 수 : 결정당 이웃 조회 평균.\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            run_lambda(ax, exp['lam'], exp['total_ticks'], md)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")


if __name__ == '__main__':
    run_all()
