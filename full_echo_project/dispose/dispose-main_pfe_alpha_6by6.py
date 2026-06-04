import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

SEED = 300

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

# -------------------------------------------------------------------------
# 파라미터 설정 (AQRERM 논문 기준)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5           # AQRERM 논문 기준 — eta2 = k · R_x, k=0.5
L   = 3
C   = 0.5           # AQLRERM 식 큐 페널티 — 본 sweep 에선 고정

BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

# -------------------------------------------------------------------------
# ade_alpha sweep — PFE_c_AdE 의 advantage boost 환산 계수만 바꿔가며 비교
# eta_n = min(eta, eta2_base + alpha · max(0, Score_y* − Score_n))
# alpha 가 클수록 advantageous 이웃에 더 강한 boost
# -------------------------------------------------------------------------
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.8, 1.0, 20, 30, 50, 100]  # AQRERM 논문 기준 sweep 포인트 + 추가 점검용 포인트

# 10 색 그라데이션 — cividis (적녹색약 친화 perceptually uniform colormap)
_cmap = plt.cm.cividis
ALPHA_COLORS = [_cmap(i / (len(ALPHA_VALUES) - 1)) for i in range(len(ALPHA_VALUES))]

STAT_INTERVAL = 100
MD_PATH = 'result_pfe_alpha_6by6.md'

EXPERIMENTS = [
    # {'lam': 2.5, 'total_ticks': 10000, 'title': 'λ=2.5'},
    {'lam': 3,   'total_ticks': 10000, 'title': 'λ=3'},
    {'lam': 3.5, 'total_ticks': 10000, 'title': 'λ=3.5'},
    # {'lam': 4,   'total_ticks': 10000, 'title': 'λ=4'},
]


# -------------------------------------------------------------------------
# 진단 출력 헬퍼 — sim 객체에서 T_est/T_max + (PFE 면) FE_rt/Point 시계열 추출
# -------------------------------------------------------------------------
def print_diagnostics(sim, total_ticks, is_pfe):
    t_est_series = getattr(sim, 't_est_series', None)
    t_max_series = getattr(sim, 't_max_series', None)
    if not (t_est_series and t_max_series):
        return

    n_chunks   = max(1, total_ticks // 1000)
    chunk_size = max(1, len(t_est_series) // n_chunks)

    def chunk_mean(series):
        return [
            float(np.mean(series[i:i + chunk_size]))
            for i in range(0, chunk_size * n_chunks, chunk_size)
        ]

    t_est_chunks = chunk_mean(t_est_series)
    t_max_chunks = chunk_mean(t_max_series)
    print(f"      [T_est ] {' '.join(f'{m:6.2f}' for m in t_est_chunks)}   (1000-tick 평균)")
    print(f"      [T_max ] {' '.join(f'{m:6.2f}' for m in t_max_chunks)}   (1000-tick 평균)")

    if is_pfe:
        fe_series = getattr(sim, 'pfe_full_echo_ratio_series', None)
        tp_series = getattr(sim, 'pfe_total_point_series', None)
        if fe_series and tp_series:
            fe_chunks = chunk_mean(fe_series)
            tp_chunks = chunk_mean(tp_series)
            print(f"      [FE_rt ] {' '.join(f'{m:6.3f}' for m in fe_chunks)}   (Full Echo 발동 비율)")
            print(f"      [Point ] {' '.join(f'{m:6.2f}' for m in tp_chunks)}   (노드별 평균 포인트)")

        # AdE 진단 — Adv > 0 이벤트의 빈도 / 평균 Score / 평균 Adv
        adv_rt_series = getattr(sim, 'pfe_adv_event_ratio_series', None)
        adv_av_series = getattr(sim, 'pfe_adv_avg_series',         None)
        sy_avg_series = getattr(sim, 'pfe_score_y_avg_series',     None)
        sn_avg_series = getattr(sim, 'pfe_score_n_avg_series',     None)
        if adv_rt_series and adv_av_series:
            adv_rt_chunks = chunk_mean(adv_rt_series)
            adv_av_chunks = chunk_mean(adv_av_series)
            sy_chunks     = chunk_mean(sy_avg_series)
            sn_chunks     = chunk_mean(sn_avg_series)
            print(f"      [Adv_rt] {' '.join(f'{m:6.3f}' for m in adv_rt_chunks)}   (Adv>0 이벤트 / 전체 라우팅)")
            print(f"      [Adv_av] {' '.join(f'{m:6.2f}' for m in adv_av_chunks)}   (Adv 이벤트의 평균 max Adv)")
            print(f"      [S_y*  ] {' '.join(f'{m:6.2f}' for m in sy_chunks)}   (그때 평균 Score_y_star)")
            print(f"      [S_n   ] {' '.join(f'{m:6.2f}' for m in sn_chunks)}   (그때 평균 최선 Score_n)")

        # AdE eta_n 분포 — clip rate / 평균 / 분산 / y_star switch
        clip_series  = getattr(sim, 'pfe_eta_n_clip_rate_series', None)
        eta_n_series = getattr(sim, 'pfe_eta_n_avg_series',       None)
        var_series   = getattr(sim, 'pfe_eta_n_var_series',       None)
        sw_series    = getattr(sim, 'pfe_switch_rate_series',     None)
        if clip_series and eta_n_series:
            clip_chunks  = chunk_mean(clip_series)
            eta_n_chunks = chunk_mean(eta_n_series)
            var_chunks   = chunk_mean(var_series)
            sw_chunks    = chunk_mean(sw_series)
            print(f"      [η_clip] {' '.join(f'{m:6.3f}' for m in clip_chunks)}   (eta_n == eta cap 비율)")
            print(f"      [η_avg ] {' '.join(f'{m:6.3f}' for m in eta_n_chunks)}   (eta_n 평균)")
            print(f"      [η_var ] {' '.join(f'{m:6.4f}' for m in var_chunks)}   (eta_n 분산)")
            print(f"      [Sw_rt ] {' '.join(f'{m:6.3f}' for m in sw_chunks)}   (y_star 변경 비율)")


# -------------------------------------------------------------------------
# 메인 실험: 부하 N 개 × alpha 10 개 그래프
# -------------------------------------------------------------------------
def run_alpha_sweep(md_file):
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(45, 8))
    if len(EXPERIMENTS) == 1:
        axes = [axes]   # 단일 Axes 를 list 로 wrap (zip 호환)
    fig.suptitle(
        f"6x6 Grid — PFE_c_AdE ade_alpha sweep (c={C}, L={L})  "
        f"α ∈ {ALPHA_VALUES}"
    )

    md_file.write(f"# PFE_c_AdE ade_alpha sweep (6x6 Grid, c={C}, L={L})\n\n")
    md_file.write(f"alpha values: {ALPHA_VALUES}\n\n")

    for ax, exp in zip(axes, EXPERIMENTS):
        lam = exp['lam']
        total_ticks = exp['total_ticks']
        x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

        md_file.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
        md_file.write("| algo | ade_alpha | generated | delivered | undelivered | delivery_rate |\n")
        md_file.write("|------|-----------|-----------|-----------|-------------|---------------|\n")

        print(f"\n=== λ={lam} ===")

        # ---- PFE_c_AdE with ade_alpha sweep (점선) ----
        for alpha, color in zip(ALPHA_VALUES, ALPHA_COLORS):
            params = {**BASE_PARAMS, 'ade_alpha': alpha}
            label = f"PFE_c_AdE α={alpha}"

            random.seed(SEED)
            np.random.seed(SEED)
            print(f"  Running {label}...")

            sim = Simulator(algorithm='pfe_c_ade', params=params, seed=SEED, topology=TOPOLOGY_GRID)
            adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)

            gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
            rate = (dlv / gen * 100) if gen > 0 else 0.0
            print(f"    {label:22s} generated={gen:6d}  delivered={dlv:6d}  "
                  f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
            md_file.write(f"| PFE_c_AdE | {alpha} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")

            print_diagnostics(sim, total_ticks, is_pfe=True)

            # 링크 사용량 top-5 — alpha 별 라우팅 편향 진단
            if hasattr(sim, 'link_usage') and sim.link_usage:
                top5 = sorted(sim.link_usage.items(), key=lambda x: -x[1])[:5]
                top5_str = '  '.join(f"{link}:{count}" for link, count in top5)
                print(f"      [Top5  ] {top5_str}")
                md_file.write(f"  - Top 5 links: {top5_str}\n")

            ax.plot(x_axis, adt, label=label, color=color, linestyle='--')

        md_file.write("\n")
        ax.set_title(exp['title'])
        ax.set_xlabel('Simulator Time')
        ax.set_ylabel('Average Delivery Time')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "result_pfe_alpha_6by6.png"
    plt.savefig(filename, dpi=150)
    print(f"\n결과 저장: {filename}")
    plt.close()


if __name__ == '__main__':
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        run_alpha_sweep(md)
    print(f"로그: {MD_PATH}")
