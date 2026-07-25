# =============================================================================
# [요약] AQRERM vs AQPRICE 정량 지표 비교
# - 6x6 grid, 10 시드. 핵심 지표 3개 + 패킷 전달 통계
# - 지표:
# ① 앞 절반 평균 ADT (학습 과도기 성능) — 각 시드 앞 절반 평균의 시드 간 중앙값
# ② 뒤 절반 평균 ADT (정착 후 성능) — 각 시드 뒤 절반 평균의 시드 간 중앙값
# ③ 결정당 에코 이웃 수 (이웃 조회 총횟수 / 라우팅 결정 총횟수, 성능 대 비용)
# - 정착 구간은 '실행 시간의 뒤 절반' 으로 정의 (임의 tick 지정 아님)
# - 산출물: result_aqrerm_vs_aqprice.md / .png
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
# 파라미터 (main_compare_AQPRICE 와 동일)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
# C   = 0.22
C   = 0
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

MD_PATH  = 'result_aqrerm_vs_aqprice.md'
PNG_PATH = 'result_aqrerm_vs_aqprice.png'

EXPERIMENTS = [
    # {'lam': 2, 'total_ticks': TOTAL_TICKS},
    {'lam': 3, 'total_ticks': TOTAL_TICKS},
    # {'lam': 3.5, 'total_ticks': TOTAL_TICKS},
    {'lam': 3.8, 'total_ticks': TOTAL_TICKS},
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
# - 정착 구간의 임의성 논란을 피하기 위해 실행 시간을 정확히 반으로 나눔
# - 앞 절반 평균 ADT = 학습 과도기 성능, 뒤 절반 평균 ADT = 정착 후 성능
# -------------------------------------------------------------------------
def compute_half_adt(adt_arr):
    """실행을 앞/뒤 절반으로 나눠 각 시드의 평균 ADT 를 낸 뒤 시드 간 중앙값 반환.
    (front, back) 두 값. 앞 절반=학습 과도기, 뒤 절반=정착 후."""
    half = adt_arr.shape[1] // 2
    front_per_seed = np.nanmean(adt_arr[:, :half], axis=1)
    back_per_seed  = np.nanmean(adt_arr[:, half:], axis=1)
    return float(np.nanmedian(front_per_seed)), float(np.nanmedian(back_per_seed))


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
        gens, dlvs, unds, rates = [], [], [], []
        for seed in SEEDS:
            print(f"  Running seed={seed}...")
            sim, adt = run_one(algo, lam, total_ticks, seed)
            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            adt_runs.append(adt)
            gens.append(gen); dlvs.append(dlv); unds.append(und); rates.append(rate)
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
            # 전달 통계 — 붕괴를 감추지 않도록 평균 사용 + 성공률 최고/최저
            'gen_mean':  float(np.mean(gens)),
            'dlv_mean':  float(np.mean(dlvs)),
            'und_mean':  float(np.mean(unds)),
            'rate_mean': float(np.mean(rates)),
            'rate_max':  float(np.max(rates)),
            'rate_min':  float(np.min(rates)),
        }

    # ---- 알고리즘별 지표 계산 (앞/뒤 절반 평균 ADT + 에코 비용) ----
    metrics = {}
    for algo in ALGORITHMS:
        front, back = compute_half_adt(results[algo]['adt_arr'])
        metrics[algo] = {
            'front':      front,           # 앞 절반 평균 ADT (학습 과도기)
            'back':       back,            # 뒤 절반 평균 ADT (정착 후)
            'echo_cost':  results[algo]['echo_cost'],
        }

    # ---- 시각화 : median 실선 + IQR 오차 막대 ----
    spacing = max(1, len(x_axis) // 10)
    for idx, algo in enumerate(ALGORITHMS):
        median = results[algo]['median']
        yerr = np.vstack([median - results[algo]['q25'],
                          results[algo]['q75'] - median])
        offset = idx * spacing // len(ALGORITHMS)
        ax.errorbar(x_axis, median, yerr=yerr,
                    errorevery=(offset, spacing),
                    capsize=3, elinewidth=1.2, capthick=1.2,
                    label=LABELS[algo], color=COLORS[algo], linewidth=2.0)

    ax.set_title(f"λ={lam}", fontsize=16)
    ax.set_xlabel('Simulator Time', fontsize=14)
    ax.set_ylabel('Average Delivery Time', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.3)

    aqrerm_m  = metrics['aqrerm']
    aqprice_m = metrics['aqprice']
    front_imp = pct_improvement(aqrerm_m['front'],     aqprice_m['front'])
    back_imp  = pct_improvement(aqrerm_m['back'],      aqprice_m['back'])
    echo_imp  = pct_improvement(aqrerm_m['echo_cost'], aqprice_m['echo_cost'])

    # ---- MD 로그 (핵심 지표 3개) ----
    md.write("| 지표 | AQRERM | AQPRICE | 개선 |\n")
    md.write("|---|---:|---:|---:|\n")
    md.write(f"| 앞 절반 평균 ADT (학습 과도기) | {aqrerm_m['front']:.2f} | {aqprice_m['front']:.2f} | {front_imp:+.1f}% |\n")
    md.write(f"| 뒤 절반 평균 ADT (정착 후) | {aqrerm_m['back']:.2f} | {aqprice_m['back']:.2f} | {back_imp:+.1f}% |\n")
    md.write(f"| 결정당 에코 이웃 수 | {aqrerm_m['echo_cost']:.2f} | {aqprice_m['echo_cost']:.2f} | {echo_imp:+.1f}% |\n\n")

    # ---- MD: 패킷 전달 통계 ----
    aqrerm_r  = results['aqrerm']
    aqprice_r = results['aqprice']
    md.write(f"#### 패킷 전달 통계 ({len(SEEDS)} 시드)\n\n")
    md.write("| 항목 | AQRERM | AQPRICE |\n")
    md.write("|---|---:|---:|\n")
    md.write(f"| 생성 (평균) | {aqrerm_r['gen_mean']:.0f} | {aqprice_r['gen_mean']:.0f} |\n")
    md.write(f"| 전달 (평균) | {aqrerm_r['dlv_mean']:.0f} | {aqprice_r['dlv_mean']:.0f} |\n")
    md.write(f"| 미전달 (평균) | {aqrerm_r['und_mean']:.0f} | {aqprice_r['und_mean']:.0f} |\n")
    md.write(f"| 성공률 평균 | {aqrerm_r['rate_mean']:.1f}% | {aqprice_r['rate_mean']:.1f}% |\n")
    md.write(f"| 성공률 최고 | {aqrerm_r['rate_max']:.1f}% | {aqprice_r['rate_max']:.1f}% |\n")
    md.write(f"| 성공률 최저 | {aqrerm_r['rate_min']:.1f}% | {aqprice_r['rate_min']:.1f}% |\n\n")

    # ---- 콘솔 출력 ----
    print(f"  {'지표':<24} | {'AQRERM':>12} | {'AQPRICE':>12} | {'개선':>12}")
    print(f"  {'-'*24}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'앞 절반 평균 ADT':<24} | {aqrerm_m['front']:>12.2f} | {aqprice_m['front']:>12.2f} | {front_imp:>+11.1f}%")
    print(f"  {'뒤 절반 평균 ADT':<24} | {aqrerm_m['back']:>12.2f} | {aqprice_m['back']:>12.2f} | {back_imp:>+11.1f}%")
    print(f"  {'결정당 에코 이웃 수':<24} | {aqrerm_m['echo_cost']:>12.2f} | {aqprice_m['echo_cost']:>12.2f} | {echo_imp:>+11.1f}%")


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
        md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}, STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 지표 설명\n\n')
        md.write('- 정착 구간은 실행 시간을 정확히 반으로 나눈 뒤 절반으로 정의한다 (임의 tick 지정 아님).\n')
        md.write('- 앞 절반 평균 ADT (학습 과도기) : 각 시드의 앞 절반 구간 평균 전달시간의 시드 간 중앙값. 작을수록 빨리 좋아짐.\n')
        md.write('- 뒤 절반 평균 ADT (정착 후) : 각 시드의 뒤 절반 구간 평균 전달시간의 시드 간 중앙값. 작을수록 정착 성능이 좋음.\n')
        md.write('- 결정당 에코 이웃 수 : 라우팅 결정 1회당 이웃에게 조회한 평균 횟수. 통신 오버헤드(성능 대 비용).\n')
        md.write('- 패킷 전달 통계 : 생성/전달/미전달은 시드 평균 패킷 수. '
                 '성공률은 전달/생성(시드 평균), 최고/최저는 시드 중 최대/최소 성공률(붕괴 확인용).\n\n')

        for ax, exp in zip(axes, EXPERIMENTS):
            run_lambda(ax, exp['lam'], exp['total_ticks'], md)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")


if __name__ == '__main__':
    run_all()
