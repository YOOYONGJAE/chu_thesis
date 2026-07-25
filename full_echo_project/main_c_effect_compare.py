# =============================================================================
# [요약] 큐 페널티 c 효과 비교 — 4 변형
#   aqrerm_c    : AQRERM + 큐 페널티 c (aqrerm_c>0)
#   aqrerm      : 순수 AQRERM (c 없음)
#   aqprice_noc : AQPRICE 에서 c 제거 (c=0)
#   aqprice     : AQPRICE (c 있음, 정식)
# - 배경: PQ(15노드) 토폴로지 고부하에서 c 가 있는 AQPRICE 가 AQRERM 보다 오히려 발산하는
#         현상을 관찰. c(이웃 큐 혼잡 반영)가 고부하에서 역효과인지 4 변형으로 확인.
# - node.py 의 aqrerm_c(AQRERM 큐 페널티) / c(AQPRICE 큐 페널티) 로 제어
# - 5 지표: 앞/뒤 절반 평균 ADT / 최악 ADT / CV / 결정당 에코 이웃 수 (정착 구간 = 뒤 절반)
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
    'aqprice_noc': 'AQPRICE (c 제거)',
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

MD_PATH  = 'result_c_effect_compare.md'
PNG_PATH = 'result_c_effect_compare.png'

# 고부하 위주 (발산 현상이 나타나는 구간)
EXPERIMENTS = [
    {'lam': 3.0, 'total_ticks': 20000},
    {'lam': 3.5, 'total_ticks': 20000},
    {'lam': 3.8, 'total_ticks': 20000},
    # {'lam': 5.0, 'total_ticks': 20000},
]


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

    ax.set_title(f"λ={lam} ({TOPO_NAME})", fontsize=16)
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
    head = " | ".join(f"{LABELS_SHORT[v]:>10}" for v in VARIANTS)
    print(f"  {'지표':<22} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>10}" for v in VARIANTS)
        print(f"  {name:<22} | {vals}")

    prow("앞 절반 평균 ADT", lambda v: f"{metrics[v]['front']:.2f}")
    prow("뒤 절반 평균 ADT", lambda v: f"{metrics[v]['back']:.2f}")
    prow("최악 ADT", lambda v: f"{metrics[v]['worst']:.2f}")
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
        f"{TOPO_NAME} — 큐 페널티 c 효과 비교 (AQRERM+c / AQRERM / AQPRICE-c / AQPRICE)  "
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

        md.write('## 지표 설명 (5 지표)\n\n')
        md.write('- 배경 : PQ 토폴로지 고부하에서 c(이웃 큐 혼잡 반영)가 있는 AQPRICE 가 AQRERM 보다 발산하는 현상 관찰.\n')
        md.write('  c 를 AQRERM 에 붙이고(aqrerm_c) AQPRICE 에서 빼서(c=0) c 의 순효과를 4 변형으로 분리.\n')
        md.write('- 정착 구간은 실행 시간의 뒤 절반으로 정의 (임의 tick 아님).\n')
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
