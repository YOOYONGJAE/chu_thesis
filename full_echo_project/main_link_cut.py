import random
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from echo_controller import LTrainController
from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A

SEED = 100
BASE_PARAMS = {'eta': 0.9, 'k': 0.5, 'L': 3}
TOPO = {'num_nodes': G_N, 'adjacency': G_A}

ALGORITHMS = [
    # 'q_routing',
    # 'aqfe',
    'aqrerm',
    # 'aqlrerm_c03',
    # 'aqlrerm',
    # 'aqlrerm_tdec',
    # 'pfe',
    # 'pfe_tdec',
    # 'pfe_ppc',
    # 'pfe_c03',
    # 'aqlrerm_c_ade',
    # 'pfe_c_ade',
    # 'pfe_c_pre_echo',
    'pfe_c_pre_echo_tick',  # ★ 변형: 라우팅 안 해도 매 tick gr 적립 (idle 노드도 포인트 누적)
    # 'pfe_pre_echo_tick',    # ★ 변형: 큐 항 (c · queue) 제거, 선택은 t / Q 만
    # 'pfe_then_aqrerm',      # ★ Hybrid: switch_tick 까지 PFE_c_pre_echo_tick, 이후 AQRERM
    # === link_cut 전용 (_l0 / no_mem variants) — 필요 시 활성 ===
    # 'aqrerm_l0',                # AQRERM + 7000 이후 L=0
    # 'aqlrerm_c03_l0',
    # 'aqlrerm_c05_l0',
    # 'aqlrerm_c_ade_l0',         # AQLRERM_c=0.5_AdE + 7000 이후 L=0
    # 'pfe_c05_l0',
    # 'pfe_c03_l0',
    # 'pfe_c01_ade_l0',           # L=0 + c=0.1
    # 'pfe_c_ade_l0',             # L=0 + c=0.5
    # 'pfe_c10_ade_l0',           # L=0 + c=1.0
    # 'pfe_c_pre_echo_l0',        # Pre-echo + 7000 이후 L=0 강제
    'pfe_c_pre_echo_tick_l0',   # Pre-echo + tick 적립 + 7000 이후 L=0 강제
    ]
LABELS = {'q_routing': 'Q-routing', 'aqfe': 'AQFE', 'aqrerm': 'AQRERM',
          'aqlrerm': 'AQLRERM_c',
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
          'pfe_c_pre_echo': 'PFE_c_PreEcho',
          'pfe_c_pre_echo_tick': 'PFE_c_PreEcho_Tick',
          'pfe_pre_echo_tick':   'PFE_PreEcho_Tick',
          'pfe_then_aqrerm': 'Hybrid_PFE→AQRERM',
          'learned_aqrerm': 'Learned AQRERM', 'bandit_aqrerm': 'Bandit AQRERM',
          # === link_cut 전용 (_l0 / no_mem variants) ===
          'aqrerm_l0': 'AQRERM_L=0',
          'aqrerm_no_mem': 'AQRERM_no_mem',
          'aqlrerm_c01_l0': 'AQLRERM_c=0.1_L=0',
          'aqlrerm_c03_l0': 'AQLRERM_c=0.3_L=0',
          'aqlrerm_c05_l0': 'AQLRERM_c=0.5_L=0',
          'aqlrerm_c05_l0_tdec': 'AQLRERM_c=0.5_L=0_Tdec',
          'aqlrerm_c07_l0': 'AQLRERM_c=0.7_L=0',
          'pfe_c05_l0':     'PFE_c=0.5_L=0',
          'pfe_c03_l0':     'PFE_c=0.3_L=0',
          'pfe_c01_ade_l0': 'PFE_c=0.1_L=0_AdE',
          'pfe_c_ade_l0':   'PFE_c=0.5_L=0_AdE',
          'pfe_c10_ade_l0': 'PFE_c=1.0_L=0_AdE',
          'aqlrerm_c_ade_l0': 'AQLRERM_c=0.5_L=0_AdE',
          'pfe_c_pre_echo_l0':      'PFE_c_PreEcho_L=0',
          'pfe_c_pre_echo_tick_l0': 'PFE_c_PreEcho_Tick_L=0',
          'aqlrerm_7000_no_c':  'AQLRERM_c=0_L=0',
          }
COLORS = {'q_routing': 'blue', 'aqfe': 'orange',
          # === 활성화 변형: family 별 hue 분리, c 값 따라 톤 차이 ===
          'aqrerm':         'navy',               # 기준선 — 차가운 단일색
          'aqlrerm':        'darkorange',         # AQLRERM family (오렌지) — c=0.5
          'aqlrerm_c03':    'gold',               # AQLRERM family — c=0.3 (밝은 오렌지)
          'aqlrerm_c_ade':  'green',              # AQLRERM + AdE — family 와 구분
          'pfe_c':          'red',                # PFE family (빨강) — c=0.5
          'pfe_c03':        'lightcoral',         # PFE family — c=0.3 (밝은 빨강)
          'pfe_c_ade':      'magenta',            # PFE + AdE — family 와 구분
          'pfe_c_pre_echo': 'purple',             # PFE + Pre-echo — 보라 (magenta 와 다른 톤)
          'pfe_c_pre_echo_tick': 'red',        # PFE + Pre-echo + tick 적립 — 같은 family, 진한 톤
          'pfe_pre_echo_tick':   'teal',       # PFE + Pre-echo + tick 적립 + 큐 항 제거 — 청록
          'pfe_then_aqrerm': 'darkgreen',      # Hybrid — 두 family 와 명확히 구분되는 녹색

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
          'learned_aqrerm': 'brown', 'bandit_aqrerm': 'purple',

          # === link_cut 전용 (_l0 / no_mem variants) ===
          'aqrerm_l0':      '#D55E00',            # 주홍 (AQRERM family L=0 변형)
          'aqrerm_no_mem':  'magenta',
          'aqlrerm_c01_l0': 'gold',
          'aqlrerm_c03_l0': '#0072B2',
          'aqlrerm_c05_l0': '#E69F00',            # 오렌지
          'aqlrerm_c05_l0_tdec': 'skyblue',
          'aqlrerm_c07_l0': 'magenta',
          'pfe_c05_l0':     'red',
          'pfe_c03_l0':     'lightcoral',
          'pfe_c01_ade_l0': '#F0E442',            # 노랑 (c=0.1, 옅은 톤 느낌)
          'pfe_c_ade_l0':   "#CFD66B",            # 분홍보라 (c=0.5, 기준)
          'pfe_c10_ade_l0': '#0072B2',            # 파랑 (c=1.0, 비활성)
          'aqlrerm_c_ade_l0': "#0044FF",          # 청록 (AQLRERM_c=0.5_L=0_AdE)
          'pfe_c_pre_echo_l0':      "#9CAEE9",    # PFE Pre-echo + L=0 at 7000
          'pfe_c_pre_echo_tick_l0': "#E983BB",    # 분홍 (Pre-echo + tick 적립 + L=0 at 7000)
          'aqlrerm_7000_no_c':  'crimson',        # c=0, L=0 at 7000
          }

STAT_INTERVAL = 200
CUT_TICK = 7000
TOTAL_TICKS = 80000

# 절단 시나리오: 두 다리만
CUT_SCENARIOS = [
    {'name': '(14,15) [top bridge]',      'cuts': [(CUT_TICK, 14, 15)]},
    {'name': '(2,3) [bottom bridge]', 'cuts': [(CUT_TICK, 2, 3)]},
]

# 부하
LAMBDAS = [1.5, 2]

# c-sweep 설정
C_VALUES = [0.22]
MD_PATH = 'result_link_cut.md'


# -------------------------------------------------------------------------
# 한 c 값에 대한 실험: 2개 절단 x 3개 부하 = 6개 패널
# -------------------------------------------------------------------------
def run_one_c(c, md_file):
    # memory_cut_tick : link_cut 이후 L=0 강제용 (link_cut 전용)
    # switch_tick     : pfe_then_aqrerm 전용 (다른 알고리즘은 무시)
    params = {**BASE_PARAMS, 'c': c, 'memory_cut_tick': CUT_TICK, 'switch_tick': 15000}

    fig, axes = plt.subplots(len(CUT_SCENARIOS), len(LAMBDAS), figsize=(45, 20), squeeze=False)
    train_l_range = f"{min(LTrainController.ACTIONS)}~{max(LTrainController.ACTIONS)}"
    fig.suptitle(f"6x6 Grid (L={BASE_PARAMS['L']}, TRAIN_L={train_l_range}, seed={SEED})")

    md_file.write(f"## c = {c}\n\n")

    for row, scenario in enumerate(CUT_SCENARIOS):
        md_file.write(f"### Cut {scenario['name']}\n\n")
        for col, lam in enumerate(LAMBDAS):
            ax = axes[row, col]
            md_file.write(f"#### λ={lam} ({TOTAL_TICKS} ticks)\n\n")
            md_file.write("| algo | generated | delivered | undelivered | delivery_rate |\n")
            md_file.write("|------|-----------|-----------|-------------|---------------|\n")

            print(f"\n=== c={c} Cut {scenario['name']} λ={lam} ===")
            for algo in ALGORITHMS:
                random.seed(SEED)
                np.random.seed(SEED)
                # AQLRERM_L_TRAIN 은 학습 후보 L 범위도 같이 표기
                if algo == 'aqlrerm_l_train':
                    train_l_range = f"{min(LTrainController.ACTIONS)}~{max(LTrainController.ACTIONS)}"
                    print(f"  알고리즘 시작 >> {LABELS[algo]}... (TRAIN_L={train_l_range})")
                else:
                    print(f"  알고리즘 시작 >> {LABELS[algo]}...")

                # 시뮬레이터 생성, 링크 절단 시나리오 전달
                sim = Simulator(algorithm=algo, params=params, seed=SEED, topology=TOPO)
                # 링크 절단 시나리오 전달
                adt = sim.run(
                    lam=lam,
                    total_ticks=TOTAL_TICKS,
                    stat_interval=STAT_INTERVAL,
                    link_cuts=scenario['cuts']
                )

                gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                rate = (dlv / gen * 100) if gen > 0 else 0.0
                print(f"    {LABELS[algo]:18s} generated={gen:6d}  delivered={dlv:6d}  "
                      f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                md_file.write(f"| {LABELS[algo]} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

                # ---- T_est / T_max 시간순 10등분 평균 (절단 전후 추이 진단) ----
                t_est_series = getattr(sim, 't_est_series', None)
                t_max_series = getattr(sim, 't_max_series', None)
                if t_est_series and t_max_series:
                    n_t = len(t_est_series)
                    n_chunks = 25
                    chunk_size = max(1, n_t // n_chunks)
                    t_est_chunks = [
                        float(np.mean(t_est_series[i:i + chunk_size]))
                        for i in range(0, chunk_size * n_chunks, chunk_size)
                    ]
                    t_max_chunks = [
                        float(np.mean(t_max_series[i:i + chunk_size]))
                        for i in range(0, chunk_size * n_chunks, chunk_size)
                    ]
                    print(f"      [T_est ] {' '.join(f'{m:6.2f}' for m in t_est_chunks)}   (시간순 10등분, 네트워크 평균)")
                    print(f"      [T_max ] {' '.join(f'{m:6.2f}' for m in t_max_chunks)}   (시간순 10등분, 네트워크 평균)")

                # ---- selected_L 진단 로그 (L_TRAIN / L_CLOSE) ----
                if algo in ('aqlrerm_l_train', 'aqlrerm_l_close') and sim.controller is not None:
                    L_hist = sim.controller.L_history
                    if L_hist:
                        total = len(L_hist)
                        mean_L = sum(L_hist) / total
                        counts = Counter(L_hist)
                        dist_str = "  ".join(
                            f"L={k}:{v:>6d}({v/total*100:5.1f}%)"
                            for k, v in sorted(counts.items())
                        )
                        n_chunks = 14
                        chunk_size = max(1, total // n_chunks)
                        chunk_means = [
                            float(np.mean(L_hist[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        chunks_str = " ".join(f"{m:5.2f}" for m in chunk_means)
                        print(f"      [L stats] n={total}  mean={mean_L:.2f}  "
                              f"min={min(L_hist)}  max={max(L_hist)}")
                        print(f"      [L dist ] {dist_str}")
                        print(f"      [L time ] {chunks_str}   (시간순 10등분 평균, per-routing)")

                    # per-window 분포 (window 마다 한 번씩 기록된 L)
                    L_w_hist = getattr(sim.controller, 'L_window_history', None)
                    if L_w_hist:
                        n_w = len(L_w_hist)
                        mean_w = sum(L_w_hist) / n_w
                        counts_w = Counter(L_w_hist)
                        dist_w_str = "  ".join(
                            f"L={k}:{v:>4d}({v/n_w*100:5.1f}%)"
                            for k, v in sorted(counts_w.items())
                        )
                        n_chunks = 14
                        chunk_size = max(1, n_w // n_chunks)
                        chunk_means_w = [
                            float(np.mean(L_w_hist[i:i + chunk_size]))
                            for i in range(0, chunk_size * n_chunks, chunk_size)
                        ]
                        chunks_w_str = " ".join(f"{m:5.2f}" for m in chunk_means_w)
                        print(f"      [Lw stat] n={n_w}  mean={mean_w:.2f}  "
                              f"min={min(L_w_hist)}  max={max(L_w_hist)}")
                        print(f"      [Lw dist] {dist_w_str}")
                        print(f"      [Lw time] {chunks_w_str}   (시간순 10등분 평균, per-window)")

                x_axis = np.arange(1, len(adt) + 1) * STAT_INTERVAL
                # ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo], linestyle='--')
                ax.plot(x_axis, adt, label=LABELS[algo], color=COLORS[algo])

            md_file.write("\n")
            ax.axvline(x=CUT_TICK, color='red', linestyle='--', linewidth=1.5, label='Link cut')
            ax.set_title(f"λ={lam} — Cut {scenario['name']}")
            ax.set_xlabel('Simulator Time')
            ax.set_ylabel('Average Delivery Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"result_link_cut_c_{c}.png"
    plt.savefig(filename, dpi=150)
    print(f"결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# Link cut c-sweep (6x6 Grid)\n\n')
        for c in C_VALUES:
            run_one_c(c, md)
    print(f"\n모든 c-sweep 완료. 로그: {MD_PATH}")
