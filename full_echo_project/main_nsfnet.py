import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_nsfnet import NUM_NODES as NSFNET_NUM_NODES, ADJACENCY as NSFNET_ADJACENCY

SEED = 800

TOPOLOGY_NSFNET = {'num_nodes': NSFNET_NUM_NODES, 'adjacency': NSFNET_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정 (AQRERM 논문 기준)
# -------------------------------------------------------------------------
ETA = 0.9
K = 0.5 / ETA       # eta*k = 0.5 이므로 k = 0.5/0.9 ≈ 0.556
L = 3

BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L}

ALGORITHMS = ['aqrerm', 'aqlrerm_low_c', 'aqlrerm_c03', 'aqlrerm',
              'aqlrerm_c07', 'aqlrerm_high_c',
              'aqlrerm_c05_l0']
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqlrerm': 'AQLRERM_c=0.5',
          'aqlrerm_low_c': 'AQLRERM_c=0.1',
          'aqlrerm_c03':   'AQLRERM_c=0.3',
          'aqlrerm_c07':   'AQLRERM_c=0.7',
          'aqlrerm_high_c': 'AQLRERM_c=1.0',
          'aqlrerm_c05_l0': 'AQLRERM_C=0.5_L=0',  # c=0.5, L=0 (memory_cut_tick 없으면 전 구간 L=0)
          'aqlrerm_7000_no_mem': 'AQLRERM_7000_NO_MEM',
          'aqlrerm_all_no_mem': 'AQLRERM_ALL_NO_MEM',
          'aqlrerm_l_train': 'AQLRERM_L_TRAIN',
          'aqlrerm_l_close': 'AQLRERM_L_CLOSE',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange', 'aqrerm': 'navy',
          'aqlrerm':        'darkorange',         # c=0.5 (기본)
          'aqlrerm_low_c':  'gold',               # c=0.1
          'aqlrerm_c03':    'chocolate',          # c=0.3 (갈색 — darkorange 와 명확히 구분)
          'aqlrerm_c07':    'magenta',            # c=0.7 (밝은 분홍 — teal 과 명확히 구분)
          'aqlrerm_high_c': 'darkmagenta',        # c=1.0
          'aqlrerm_c05_l0': 'crimson',            # c=0.5 + L=0
          'aqlrerm_7000_no_mem': 'cyan',
          'aqlrerm_all_no_mem': 'teal',
          'aqlrerm_l_train': 'black',
          'aqlrerm_l_close': 'olive',
          'learned_aqrerm': 'brown', 'bandit_aqrerm': 'purple'}

STAT_INTERVAL = 100

# c-sweep 설정
C_VALUES = [0.5]
MD_PATH = 'result_nsfnet.md'

EXPERIMENTS = [
    {'lam': 2.5,   'total_ticks': 14000,  'title': 'λ=2.5'},
    # {'lam': 2,   'total_ticks': 10000, 'title': 'λ=2'},
    {'lam': 3, 'total_ticks': 14000, 'title': 'λ=3'},
    {'lam': 4,   'total_ticks': 14000, 'title': 'λ=4'},
    {'lam': 5, 'total_ticks': 14000, 'title': 'λ=5'},
]


# -------------------------------------------------------------------------
# 한 c 값에 대한 실험: 4개 부하별 ADT 그래프 + MD 로그 누적
# -------------------------------------------------------------------------
def run_one_c(c, md_file):
    params = {**BASE_PARAMS, 'c': c}

    fig, axes = plt.subplots(1, 4, figsize=(30, 8))
    fig.suptitle(f"NSFNET (c={c}, L={L})")

    md_file.write(f"## c = {c}\n\n")

    for ax, exp in zip(axes, EXPERIMENTS):
        lam = exp['lam']
        total_ticks = exp['total_ticks']
        x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

        md_file.write(f"### λ={lam} ({total_ticks} ticks)\n\n")
        md_file.write("| algo | generated | delivered | undelivered | delivery_rate |\n")
        md_file.write("|------|-----------|-----------|-------------|---------------|\n")

        print(f"\n=== c={c} λ={lam} ===")
        for algo in ALGORITHMS:
            random.seed(SEED)
            np.random.seed(SEED)
            print(f"  Running {LABELS[algo]}...")

            sim = Simulator(algorithm=algo, params=params, seed=SEED, topology=TOPOLOGY_NSFNET)
            adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    {LABELS[algo]:18s} generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            md_file.write(f"| {LABELS[algo]} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

            # ---- T_est / T_max 1000-tick 간격 평균 (시간 흐름 진단) ----
            t_est_series = getattr(sim, 't_est_series', None)
            t_max_series = getattr(sim, 't_max_series', None)
            if t_est_series and t_max_series:
                n_chunks   = max(1, total_ticks // 1000)
                chunk_size = max(1, len(t_est_series) // n_chunks)
                t_est_chunks = [
                    float(np.mean(t_est_series[i:i + chunk_size]))
                    for i in range(0, chunk_size * n_chunks, chunk_size)
                ]
                t_max_chunks = [
                    float(np.mean(t_max_series[i:i + chunk_size]))
                    for i in range(0, chunk_size * n_chunks, chunk_size)
                ]
                print(f"      [T_est ] {' '.join(f'{m:6.2f}' for m in t_est_chunks)}   (1000-tick 평균, 네트워크 avg)")
                print(f"      [T_max ] {' '.join(f'{m:6.2f}' for m in t_max_chunks)}   (1000-tick 평균, 네트워크 avg)")

            ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

        md_file.write("\n")
        ax.set_title(exp['title'])
        ax.set_xlabel('Simulator Time')
        ax.set_ylabel('Average Delivery Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"result_nsfnet_c_{c}.png"
    plt.savefig(filename, dpi=150)
    print(f"결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# NSFNET c-sweep\n\n')
        for c in C_VALUES:
            run_one_c(c, md)
    print(f"\n모든 c-sweep 완료. 로그: {MD_PATH}")
