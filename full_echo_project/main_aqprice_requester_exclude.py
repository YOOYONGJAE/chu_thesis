# =============================================================================
# [요약] AQPRICE 요청 노드 제외 ablation — 이웃이 최소 Q 를 낼 때 질문자를 뺄지 비교
# - 같은 aqprice 를 두 설정으로 비교:
#     exclude : 이웃이 최소 Q 계산에서 질문한 노드(x)를 제외 (현재 기본, AQRERM 라우트 메모리에서 온 조건)
#     include : 질문한 노드도 포함 (Q-routing/AQFE 와 동일, 순수 min)
# - node.py 는 aqprice_exclude_requester 로 이 조건을 제어 (기본 True=제외)
# - 라우트 메모리는 둘 다 끔(L=0, AQPRICE 정식 구성) → 요청 노드 제외 효과만 분리
# - 6x6 grid, 10 시드. 5 지표 + 패킷 전달 통계
# - 지표: 앞 절반 평균 ADT / 뒤 절반 평균 ADT / 최악 ADT / 랜덤시드 간 변동성(CV) / 결정당 에코 이웃 수
# - 정착 구간은 '실행 시간의 뒤 절반' 으로 정의 (임의 tick 지정 아님)
# - 산출물: result_aqprice_requester_exclude.md / .png
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

def topo_name(n):
    """노드 수로 토폴로지 이름 자동 판별 (import 만 바꾸면 제목도 자동 반영)."""
    return {36: '6x6 Grid', 14: 'NSFNET', 15: 'PQ'}.get(n, f'{n}-node')


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


MD_PATH  = 'result_aqprice_requester_exclude.md'
PNG_PATH = 'result_aqprice_requester_exclude.png'

EXPERIMENTS = [
    {'lam': 2, 'total_ticks': TOTAL_TICKS},
    {'lam': 3.5, 'total_ticks': TOTAL_TICKS},
    {'lam': 3.8, 'total_ticks': TOTAL_TICKS},
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
# Metric 계산 함수들 — 5 지표 (앞/뒤 절반 평균 ADT / 최악 ADT / CV)
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

    ax.set_title(f"λ={lam}", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    # ---- 개선 = baseline(요청노드 포함) 대비 main(요청노드 제외) ----
    base_m = metrics[BASE_KEY]
    main_m = metrics[MAIN_KEY]
    front_imp = pct_improvement(base_m['front'],     main_m['front'])
    back_imp  = pct_improvement(base_m['back'],      main_m['back'])
    worst_imp = pct_improvement(base_m['worst'],     main_m['worst'])
    cv_imp    = pct_improvement(base_m['cv'],        main_m['cv'])
    echo_imp  = pct_improvement(base_m['echo_cost'], main_m['echo_cost'])

    base_col = '요청노드 포함'
    main_col = '요청노드 제외 (현재)'

    md.write(f"| 지표 | {base_col} | {main_col} | 개선 |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| 앞 절반 평균 ADT (학습 과도기) | {base_m['front']:.2f} | {main_m['front']:.2f} | {front_imp:+.1f}% |\n")
    md.write(f"| 뒤 절반 평균 ADT (정착 후) | {base_m['back']:.2f} | {main_m['back']:.2f} | {back_imp:+.1f}% |\n")
    md.write(f"| 최악 ADT (구간 최댓값) | {base_m['worst']:.2f} | {main_m['worst']:.2f} | {worst_imp:+.1f}% |\n")
    md.write(f"| 랜덤시드 간 변동성 (CV, 작을수록 일관적) | {base_m['cv']:.3f} | {main_m['cv']:.3f} | {cv_imp:+.1f}% |\n")
    md.write(f"| 결정당 에코 이웃 수 | {base_m['echo_cost']:.2f} | {main_m['echo_cost']:.2f} | {echo_imp:+.1f}% |\n\n")

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

    print(f"  {'지표':<24} | {base_col:>14} | {main_col:>16} | {'개선':>12}")
    print(f"  {'-'*24}-+-{'-'*14}-+-{'-'*16}-+-{'-'*12}")
    print(f"  {'앞 절반 평균 ADT':<24} | {base_m['front']:>14.2f} | {main_m['front']:>16.2f} | {front_imp:>+11.1f}%")
    print(f"  {'뒤 절반 평균 ADT':<24} | {base_m['back']:>14.2f} | {main_m['back']:>16.2f} | {back_imp:>+11.1f}%")
    print(f"  {'최악 ADT (구간 최댓값)':<24} | {base_m['worst']:>14.2f} | {main_m['worst']:>16.2f} | {worst_imp:>+11.1f}%")
    print(f"  {'랜덤시드 간 변동성':<24} | {base_m['cv']:>14.3f} | {main_m['cv']:>16.3f} | {cv_imp:>+11.1f}%")
    print(f"  {'결정당 에코 이웃 수':<24} | {base_m['echo_cost']:>14.2f} | {main_m['echo_cost']:>16.2f} | {echo_imp:>+11.1f}%")


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(60, 12), squeeze=False)
    axes = axes.flatten()
    active_labels = ' vs '.join(LABELS[v] for v in VARIANTS)
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — AQPRICE requester-exclude ablation : {active_labels}  "
        f"(c={C}, seeds n={len(SEEDS)}, {TOTAL_TICKS}tick)",
        fontsize=17,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE 요청 노드 제외 ablation (이웃 최소 Q 계산에서 질문자 제외 유무)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS} (라우트 메모리는 둘 다 없음 = 기본 L 0)\n')
        md.write(f'- 변형별 aqprice_exclude_requester: include=False, exclude=True(현재 기본)\n')
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 지표 설명 (5 지표)\n\n')
        md.write('- 비교 대상 : 이웃이 목적지까지 최소 Q 를 계산해 돌려줄 때, 방금 질문한 노드(x)를 후보에서 뺄지 여부.\n')
        md.write('  exclude(현재 기본)는 x 를 빼고(AQRERM 라우트 메모리에서 온 조건), include 는 x 도 포함(Q-routing/AQFE 와 동일).\n')
        md.write('  라우트 메모리(L)는 두 변형 모두 없음 → 이 표의 차이는 오직 요청 노드 제외 여부에서 온다.\n\n')
        md.write('- 정착 구간은 실행 시간을 정확히 반으로 나눈 뒤 절반으로 정의 (임의 tick 아님).\n')
        md.write('- 앞 절반 평균 ADT (학습 과도기) : 각 시드 앞 절반 구간 평균 ADT 의 시드 간 중앙값.\n')
        md.write('- 뒤 절반 평균 ADT (정착 후) : 각 시드 뒤 절반 구간 평균 ADT 의 시드 간 중앙값.\n')
        md.write('- 최악 ADT (구간 최댓값) : 각 시드 최댓값의 시드 간 중앙값.\n')
        md.write('- 랜덤시드 간 변동성 (CV) : 뒤 절반 평균 ADT 의 시드 간 (표준편차 ÷ 평균). 작을수록 일관적.\n')
        md.write('- 결정당 에코 이웃 수 : 라우팅 결정 1회당 이웃 조회 평균 횟수. '
                 '요청 노드 제외는 조회 수는 안 바꾸므로 두 변형이 거의 같아야 정상.\n')
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
