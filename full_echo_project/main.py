import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

SEED = 100


TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정 (AQRERM 논문 기준)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5           # AQRERM 논문 기준 — eta2 = k · R_x, k=0.5
L   = 3

BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L}

ALGORITHMS = [
    'aqrerm',
    'aqlrerm_c03',
    'aqlrerm',
    # 'aqlrerm_tdec',
    # 'pfe',
    # 'pfe_tdec',
    'pfe_c',
    # 'pfe_c03',
    'aqlrerm_c_ade',
    'pfe_c_ade',
    ]
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqlrerm': 'AQLRERM_c=0.5',
          'aqlrerm_tdec': 'AQLRERM_c=0.5_Tdec',
          'aqlrerm_low_c': 'AQLRERM_c=0.1',
          'aqlrerm_c03':   'AQLRERM_c=0.3',
          'aqlrerm_c07':   'AQLRERM_c=0.7',
          'aqlrerm_high_c': 'AQLRERM_c=1.0',
          'aqlrerm_7000_no_mem': 'AQLRERM_7000_NO_MEM',
          'aqlrerm_all_no_mem': 'AQLRERM_ALL_NO_MEM',
          'aqlrerm_l_train': 'AQLRERM_L_TRAIN',
          'aqlrerm_l_close': 'AQLRERM_L_CLOSE',
          'pfe': 'PFE',
          'pfe_tdec': 'PFE_Tdec',
          'pfe_c':    'PFE_c=0.5',
          'pfe_c03':  'PFE_c=0.3',
          'aqlrerm_c_ade': 'AQLRERM_c=0.5_AdE',
          'pfe_c_ade':     'PFE_c=0.5_AdE',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM'}
COLORS = {'q_routing': 'blue', 'aqfe': 'orange',
          # === 활성화 변형: family 별 hue 분리, c 값 따라 톤 차이 ===
          'aqrerm':         'navy',               # 기준선 — 차가운 단일색
          'aqlrerm':        'darkorange',         # AQLRERM family (오렌지) — c=0.5
          'aqlrerm_c03':    'gold',               # AQLRERM family — c=0.3 (밝은 오렌지)
          'aqlrerm_c_ade':  'green',              # AQLRERM + AdE — family 와 구분
          'pfe_c':          'red',                # PFE family (빨강) — c=0.5
          'pfe_c03':        'lightcoral',         # PFE family — c=0.3 (밝은 빨강)
          'pfe_c_ade':      'magenta',            # PFE + AdE — family 와 구분

          # === 비활성화 변형: 기존 매핑 유지 ===
          'aqlrerm_tdec':   'skyblue',
          'aqlrerm_low_c':  'chocolate',
          'aqlrerm_c07':    'magenta',
          'aqlrerm_high_c': 'darkmagenta',
          'aqlrerm_7000_no_mem': 'cyan',
          'aqlrerm_all_no_mem': 'teal',
          'aqlrerm_l_train': 'black',
          'aqlrerm_l_close': 'olive',
          'pfe':            'black',
          'pfe_tdec':       'crimson',
          'learned_aqrerm': 'brown', 'bandit_aqrerm': 'purple'}

STAT_INTERVAL = 100

# c-sweep 설정
C_VALUES = [0.5]
MD_PATH = 'result_grid.md'

EXPERIMENTS = [
    {'lam': 2.5,   'total_ticks': 14000,  'title': 'λ=2.5'},
    {'lam': 3,   'total_ticks': 10000, 'title': 'λ=3'},
    {'lam': 3.5, 'total_ticks': 14000, 'title': 'λ=3.5'},
    {'lam': 4, 'total_ticks': 14000, 'title': 'λ=4'},
]


# -------------------------------------------------------------------------
# 한 c 값에 대한 실험: 4개 부하별 ADT 그래프 + MD 로그 누적
# -------------------------------------------------------------------------
def run_one_c(c, md_file):
    params = {**BASE_PARAMS, 'c': c}

    fig, axes = plt.subplots(1, 4, figsize=(40, 8))
    fig.suptitle(f"6x6 Grid (c={c}, L={L})")

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

            sim = Simulator(algorithm=algo, params=params, seed=SEED, topology=TOPOLOGY_GRID)
            adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    {LABELS[algo]:18s} generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            md_file.write(f"| {LABELS[algo]} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

            # ---- T_est / T_max 1000-tick 간격 평균 (시간 흐름 진단) ----
            # 각 experiment 의 total_ticks 가 다르므로 chunk 수는 total_ticks/1000 으로 자동 계산
            t_est_series = getattr(sim, 't_est_series', None)
            t_max_series = getattr(sim, 't_max_series', None)
            if t_est_series and t_max_series:
                n_chunks   = max(1, total_ticks // 1000)            # 1000 tick 마다 1 chunk
                chunk_size = max(1, len(t_est_series) // n_chunks)  # chunk 당 entry 수 (= 10 if stat_interval=100)
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

                # ---- PFE 진단: Full Echo 발동 비율, 평균 누적 포인트 (PFE 일 때만) ----
                # ratio: 윈도우 동안 (Full Echo 실행 라우팅 / 전체 라우팅 호출) — 0~1
                # total_point: stat_interval 시점의 네트워크 평균 포인트 잔고
                if algo in ('pfe', 'pfe_tdec', 'pfe_c', 'pfe_c03', 'pfe_c_ade'):
                    fe_series = getattr(sim, 'pfe_full_echo_ratio_series', None)
                    tp_series = getattr(sim, 'pfe_total_point_series', None)
                    if fe_series and tp_series:
                        fe_chunks = [
                            float(np.mean(fe_series[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        tp_chunks = [
                            float(np.mean(tp_series[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        print(f"      [FE_rt ] {' '.join(f'{m:6.3f}' for m in fe_chunks)}   (1000-tick 평균, Full Echo 발동 비율)")
                        print(f"      [Point ] {' '.join(f'{m:6.2f}' for m in tp_chunks)}   (1000-tick 평균, 노드별 total_point 평균)")

            ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

        md_file.write("\n")
        ax.set_title(exp['title'])
        ax.set_xlabel('Simulator Time')
        ax.set_ylabel('Average Delivery Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"result_grid_c_{c}.png"
    plt.savefig(filename, dpi=150)
    print(f"결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# 6x6 Grid c-sweep\n\n')
        for c in C_VALUES:
            run_one_c(c, md)
    print(f"\n모든 c-sweep 완료. 로그: {MD_PATH}")
