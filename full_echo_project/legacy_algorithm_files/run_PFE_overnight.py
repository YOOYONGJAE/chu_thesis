"""
임시 overnight runner — main_compare_PFE 와 동일 로직으로 3 개 실험을 순차 실행.

순서:
  seq1 : aqrerm_pre / aqrerm_c / aqrerm_c_pre / aqpace   @ λ ∈ {2, 2.5, 3}
  seq2 : pfe_echo_tick / pfe_c_echo_tick / pfe_pre_echo_tick / aqpace   @ λ ∈ {2, 2.5, 3}
  seq3 : pfe_echo_tick / pfe_c_echo_tick / pfe_pre_echo_tick / aqpace   @ λ ∈ {3.5, 3.7, 3.8}

각 실험마다 PNG + MD = 2 파일 → 총 6 파일 생성.
파일명: result_compare_PFE_<seq명>.png / .md
"""
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 공통 파라미터 (main_compare_PFE 와 동일)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]
STAT_INTERVAL = 100
TOTAL_TICKS = 40000

# -------------------------------------------------------------------------
# LABELS / COLORS (main_compare_PFE 와 동일 풀)
# -------------------------------------------------------------------------
LABELS = {
    'pfe_echo_tick':            'PFE_echo_tick',
    'pfe_pre_echo_tick':        'PFE_pre_echo_tick',
    'aqrerm_c':                 'AQRERM_c',
    'aqrerm':                   'AQRERM',
    'aqrerm_no_L':              'AQRERM_no_L',
    'pfe_c_echo_tick':          'PFE_c_echo_tick',
    'aqpace':      'AQPACE',
    'aqrerm_c_pre':             'AQRERM_c_pre_RERM',
    'aqrerm_pre':               'AQRERM_pre',
    'fe_c_pre_echo':            'FE_c_pre_echo',
    'aqpace_no_L': 'AQPACE_noL',
}
COLORS = {
    'pfe_echo_tick':            '#0072B2',
    'pfe_pre_echo_tick':        '#DAA32D',
    'aqrerm_c':                 '#009E73',  # 청록 (main_compare_PFE 의 변경 전 표준값)
    'aqrerm':                   '#CC79A7',
    'aqrerm_no_L':              '#882255',
    'pfe_c_echo_tick':          '#D55E00',
    'aqpace':      '#56B4E9',
    'aqrerm_c_pre':             '#F0E442',
    'aqrerm_pre':               '#117733',
    'fe_c_pre_echo':            '#000000',
    'aqpace_no_L': '#3700FF',
}

# -------------------------------------------------------------------------
# 3 개 sequential 실험 정의
# -------------------------------------------------------------------------
EXPERIMENT_SEQUENCES = [
    {
        'name': 'seq1_aqrerm_family_low',
        'algorithms': ['aqrerm_pre', 'aqrerm_c', 'aqrerm_c_pre', 'aqpace'],
        'lambdas': [2, 2.5, 3],
    },
    {
        'name': 'seq2_pfe_family_low',
        'algorithms': ['pfe_echo_tick', 'pfe_c_echo_tick', 'pfe_pre_echo_tick', 'aqpace'],
        'lambdas': [2, 2.5, 3],
    },
    {
        'name': 'seq3_pfe_family_high',
        'algorithms': ['pfe_echo_tick', 'pfe_c_echo_tick', 'pfe_pre_echo_tick', 'aqpace'],
        'lambdas': [3.5, 3.7, 3.8],
    },
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
# 한 실험 (config 1 개) 실행 — PNG + MD 저장
# -------------------------------------------------------------------------
def run_experiment(seq_config):
    name       = seq_config['name']
    algorithms = seq_config['algorithms']
    lambdas    = seq_config['lambdas']

    md_path  = f"result_compare_PFE_{name}.md"
    png_path = f"result_compare_PFE_{name}.png"

    fig, axes = plt.subplots(1, len(lambdas), figsize=(60, 15), squeeze=False)
    axes = axes.flatten()
    active_labels = ' / '.join(LABELS[a] for a in algorithms)
    fig.suptitle(
        f"6x6 Grid — [{name}] PFE 변형 ({active_labels}) "
        f"(seeds={SEEDS[0]}~{SEEDS[-1]}, n={len(SEEDS)}, median + IQR band)"
    )

    with open(md_path, 'w', encoding='utf-8') as md:
        md.write(f'# [{name}] 6x6 Grid PFE 변형 comparison (seed sweep)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in algorithms]}\n')
        md.write(f'- Lambdas: {lambdas}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n\n')

        for ax, lam in zip(axes, lambdas):
            x_axis = np.arange(1, TOTAL_TICKS // STAT_INTERVAL + 1) * STAT_INTERVAL

            md.write(f"## λ={lam} ({TOTAL_TICKS} ticks)\n\n")
            print(f"\n========== [{name}] λ={lam} ==========")

            for algo in algorithms:
                label = LABELS[algo]
                color = COLORS[algo]
                print(f"\n--- {label} ---")

                md.write(f"### {label}\n\n")
                md.write("| seed | generated | delivered | undelivered | delivery_rate |\n")
                md.write("|------|-----------|-----------|-------------|---------------|\n")

                adt_runs = []
                for seed in SEEDS:
                    print(f"  Running seed={seed}...")
                    sim, adt = run_one(algo, lam, TOTAL_TICKS, seed)
                    gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                    rate = (dlv / gen * 100) if gen > 0 else 0.0
                    print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                          f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                    md.write(f"| {seed} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")
                    adt_runs.append(adt)

                # ---- 시드별 시리즈를 (n_seeds, n_windows) 배열로 쌓고 백분위수 계산 ----
                adt_arr = np.array(adt_runs)
                median = np.median(adt_arr, axis=0)
                q25 = np.percentile(adt_arr, 25, axis=0)
                q75 = np.percentile(adt_arr, 75, axis=0)

                # ---- median 굵은 실선 + IQR 음영 ----
                ax.plot(x_axis, median, label=label, color=color, linewidth=2.0)
                ax.fill_between(x_axis, q25, q75, color=color, alpha=0.2)

                # ---- steady-state 요약 (마지막 절반 평균 → 시드별로 → median/IQR) ----
                half = adt_arr.shape[1] // 2
                ss_per_seed = np.mean(adt_arr[:, half:], axis=1)
                ss_median = float(np.median(ss_per_seed))
                ss_q25 = float(np.percentile(ss_per_seed, 25))
                ss_q75 = float(np.percentile(ss_per_seed, 75))
                ss_min  = float(np.min(ss_per_seed))
                ss_max  = float(np.max(ss_per_seed))
                print(f"    [SS ADT] median={ss_median:6.2f}  "
                      f"IQR=[{ss_q25:6.2f}, {ss_q75:6.2f}]  "
                      f"range=[{ss_min:6.2f}, {ss_max:6.2f}]")
                md.write(f"\n**Steady-state ADT (last half mean per seed)** "
                         f"median={ss_median:.2f}  "
                         f"IQR=[{ss_q25:.2f}, {ss_q75:.2f}]  "
                         f"range=[{ss_min:.2f}, {ss_max:.2f}]\n\n")

            ax.set_title(f"λ={lam}")
            ax.set_xlabel('Simulator Time')
            ax.set_ylabel('Average Delivery Time')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"\n[{name}] PNG 저장: {png_path}")
    print(f"[{name}] MD  저장: {md_path}")


# -------------------------------------------------------------------------
# main — 3 개 실험 순차 실행
# -------------------------------------------------------------------------
if __name__ == '__main__':
    total = len(EXPERIMENT_SEQUENCES)
    for i, seq in enumerate(EXPERIMENT_SEQUENCES, 1):
        print(f"\n############### Experiment {i}/{total}: {seq['name']} ###############")
        print(f"  algorithms : {seq['algorithms']}")
        print(f"  lambdas    : {seq['lambdas']}")
        run_experiment(seq)
    print(f"\n=== 전체 {total} 개 sequential 실험 완료 ===")
