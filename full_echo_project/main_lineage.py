# =============================================================================
# [요약] 계보 비교 허브 — Q-routing / AQFE / AQRERM / AQPRICE 4종 비교
# - 6x6 grid, 10 시드 sweep, λ 별 ADT median + IQR 그래프 + markdown 결과표
# - ALGORITHMS 주석 토글로 원하는 조합만 실행 가능
# - 구세대 변형 포함 버전은 legacy_algorithm_files/main_compare_PFE.py 참조
# - 산출물: result_lineage.md / .png
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트 (그래프 한글 깨짐 방지)
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from simulator import Simulator
import diagnostics as diag
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_nsfnet import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_sub_test_pq import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_sub_test_grid_full import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 설정
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
C   = 0.22   # aqprice 의 큐 페널티 가중치 (다른 알고리즘은 이 키를 안 읽음)
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}

def topo_name(n):
    """토폴로지 이름 자동 판별. 노드 수가 같은 격자 두 종은 링크 수로 가른다.
    (import 한 줄만 바꾸면 제목과 파일명이 함께 따라온다.)"""
    return diag.topo_name(n, GRID_ADJACENCY)


# -------------------------------------------------------------------------
# 비교 대상 알고리즘
# - q_routing   : Boyan & Littman 1994. y* 만 단일 업데이트, echo 없음
# - aqfe        : Adaptive Q-routing with Full Echo. 에코는 하되 선택은 갱신 전 Q-table 로
# - aqrerm      : Random Echo + Route Memory. 확률 p = T_est/T_max 로 부분 echo
# - aqprice     : 포인트 예산 게이트 + pre-echo + c·queue 페널티 (제안 기법)
# -------------------------------------------------------------------------
ALGORITHMS = [
    # 'q_routing',
    # 'aqfe',
    'aqrerm',
    'aqprice',
]
LABELS = {
    'q_routing': 'Q-routing',
    'aqfe':      'AQFE',
    # 'aqrerm':    'AQRERM',
    # 'aqprice':    'AQPRICE',
    'aqrerm': 'AQRERM',
    'aqprice': 'PROPOSED ALGORITHM',
}
# 적녹색약 친화 (Wong palette)
COLORS = {
    'q_routing': '#117733',  # 진녹 (baseline 최단순)
    'aqfe':      '#44AA99',  # teal (AQRERM 의 부모)
    'aqrerm':    "#FF0000",  # 분홍보라 (baseline)
    'aqprice':    '#56B4E9',  # 하늘색 (제안 기법)
}

# 10 개 시드 × 알고리즘 × 부하별 반복
SEEDS = list(range(100, 1001, 100))  # [100, 200, ..., 1000]

STAT_INTERVAL = 100

EXPERIMENTS = [
    # 4 단계 부하 — 저 / 중저 / 중고 / 고. 모든 실험 공통.
    # 고부하 상한을 3.8 로 둔 이유 : AQRERM 은 발산하고 AQPRICE 는 정착하는 경계 지점.
    # {'lam': 2.5, 'total_ticks': 20000},   
    {'lam': 3.0, 'total_ticks': 20000},   
    {'lam': 3.5, 'total_ticks': 20000},   
    {'lam': 3.8, 'total_ticks': 20000},   
    # {'lam': 4.1, 'total_ticks': 20000},   
    # {'lam': 4.4, 'total_ticks': 20000},   
]

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = diag.topo_tag(GRID_NUM_NODES, GRID_ADJACENCY)
# RESULT_TAG = 'LOW'   # 부하 표기 (수동): LOW / HIGH 등
# RESULT_TAG = 'HIGH'   # 부하 표기 (수동): LOW / HIGH 등
RESULT_TAG = '3LVL'   # 부하 표기 (수동): LOW / HIGH 등
_BASENAME  = f'result_lineage({TOPO_TAG},{RESULT_TAG},{EXPERIMENTS[0]["total_ticks"]})'
MD_PATH        = f'{_BASENAME}.md'
PNG_PATH       = f'{_BASENAME}.png'
LINK_PNG_PATH  = f'{_BASENAME}_linkuse.png'
SIGNAL_PNG_PATH = f'{_BASENAME}_signal.png'   # R_x / T_est / T_max 전용 그림


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
# 메인: 각 부하별로 알고리즘들을 10 시드씩 돌려 median + IQR 시각화
# -------------------------------------------------------------------------
def run_all():
    # 패널 배치 — 부하 4개면 2x2. 패널 하나당 크기가 고정이라 실험 수가 달라져도 비율 유지.
    fig, axes = diag.make_panel_grid(len(EXPERIMENTS))
    # 혼잡 신호 전용 그림 — 부하마다 위(T_est/T_max) · 아래(R_x) 두 축
    figS, sig_axes = diag.make_signal_panel_grid(len(EXPERIMENTS))
    active_labels = ' / '.join(LABELS[a] for a in ALGORITHMS)
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — {active_labels} "
        f"(c={C}, seeds n={len(SEEDS)}, median + IQR band)",
        fontsize=22,
        y=0.99,
    )
    figS.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — {active_labels} "
        f"(congestion signal: T_est / T_max / R_x, seeds n={len(SEEDS)})",
        fontsize=22,
        y=0.99,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write(f'# {topo_name(GRID_NUM_NODES)} Algorithm comparison (seed sweep)\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Algorithms: {[LABELS[a] for a in ALGORITHMS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n\n')

        diag.write_metric_guide_md(md)

        link_data = []   # [(lam, {algo: link_sum}), ...] — 링크 다이어그램용
        for panel_idx, (ax, exp) in enumerate(zip(axes, EXPERIMENTS)):
            ax_sig_t, ax_sig_r = sig_axes[panel_idx]   # 위: T_est/T_max, 아래: R_x
            lam = exp['lam']
            total_ticks = exp['total_ticks']
            x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

            md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
            print(f"\n========== λ={lam} ==========")

            per_algo_link = {}
            for algo in ALGORITHMS:
                label = LABELS[algo]
                color = COLORS[algo]
                print(f"\n--- {label} ---")

                md.write(f"### {label}\n\n")
                md.write("| seed | generated | delivered | undelivered | delivery_rate |\n")
                md.write("|------|-----------|-----------|-------------|---------------|\n")

                adt_runs = []
                echo_costs = []
                echo_totals, route_totals = [], []   # 나누기 전 총횟수 (시드별, 36개 노드 합)
                switch_rates = []
                rx_runs, echo_runs, switch_runs, link_dicts = [], [], [], []
                te_runs, tm_runs = [], []          # R_x 의 분자·분모 시계열
                gens, dlvs, unds, rates = [], [], [], []   # 패킷 전달 통계 평균용
                for seed in SEEDS:
                    print(f"  Running seed={seed}...")
                    sim, adt = run_one(algo, lam, total_ticks, seed)
                    gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                    rate = (dlv / gen * 100) if gen > 0 else 0.0
                    print(f"    seed={seed:4d}  generated={gen:6d}  delivered={dlv:6d}  "
                          f"undelivered={und:6d}  delivery_rate={rate:5.1f}%")
                    md.write(f"| {seed} | {gen} | {dlv} | {und} | {rate:.1f}% |\n")
                    gens.append(gen); dlvs.append(dlv); unds.append(und); rates.append(rate)
                    adt_runs.append(adt)
                    ec = (sim.total_echo_queries / sim.total_route_calls
                          if sim.total_route_calls > 0 else 0.0)
                    echo_costs.append(ec)
                    echo_totals.append(sim.total_echo_queries)
                    route_totals.append(sim.total_route_calls)
                    sw =(sim.total_switches / sim.total_route_calls
                          if sim.total_route_calls > 0 else 0.0)
                    switch_rates.append(sw)
                    rx_runs.append(sim.r_ratio_series)
                    te_runs.append(sim.t_est_series)
                    tm_runs.append(sim.t_max_series)
                    echo_runs.append(sim.echo_per_decision_series)
                    switch_runs.append(sim.switch_rate_series)
                    link_dicts.append(sim.link_usage)

                # 시드 표 끝에 평균 행 — 다른 비교 스크립트의 '패킷 전달 통계' 와 같은 항목.
                # 성공률은 시드별 비율의 평균이며, 최고·최저를 함께 적어 특정 시드의 붕괴를 확인한다.
                md.write(f"| 평균 | {np.mean(gens):.0f} | {np.mean(dlvs):.0f} | "
                         f"{np.mean(unds):.0f} | {np.mean(rates):.1f}% |\n")
                md.write(f"| 최고/최저 성공률 | - | - | - | "
                         f"{np.max(rates):.1f}% / {np.min(rates):.1f}% |\n")

                # ---- 시드별 시리즈를 (n_seeds, n_windows) 배열로 쌓고 백분위수 계산 ----
                # 전달 패킷이 0인 통계 구간은 simulator 가 NaN 을 넣으므로,
                # 그 시드 자리만 제외하고 나머지 시드로 집계하는 nan 무시 함수를 사용
                adt_arr = np.array(adt_runs)
                median = np.nanmedian(adt_arr, axis=0)
                q25 = np.nanpercentile(adt_arr, 25, axis=0)
                q75 = np.nanpercentile(adt_arr, 75, axis=0)

                # ---- median 실선 + IQR 오차 막대 (일정 간격 세로선) ----
                # yerr 는 (아래 길이, 위 길이) 두 행 — median 기준 비대칭 IQR 범위
                # offset : 알고리즘마다 시작점을 어긋내 막대가 같은 x 에 겹치지 않게 함
                yerr = np.vstack([median - q25, q75 - median])
                spacing = max(1, len(x_axis) // 10)
                offset = ALGORITHMS.index(algo) * spacing // len(ALGORITHMS)
                ax.errorbar(x_axis, median, yerr=None, # 에러바 숨기기
                # ax.errorbar(x_axis, median, yerr=yerr, # 에러바 표출하기
                            errorevery=(offset, spacing),
                            capsize=3, elinewidth=1.2, capthick=1.2,
                            label=label, color=color, linewidth=2.0)

                # ---- ADT 전체구간 요약 + 진단 시계열 + 링크. 논문에선 필요한 것만 발췌 ----
                # 구간을 나누지 않고 위에서 그린 실선(median)·띠(q75−q25)를 그대로 시간 평균한다.
                overall_adt = float(np.nanmean(median))
                overall_iqr = float(np.nanmean(q75 - q25))
                echo_cost   = float(np.mean(echo_costs))      # 시드 10개 값의 평균 (패킷 전달 통계와 같은 규칙)
                switch_rate = float(np.mean(switch_rates))
                # R_x 는 진단 시계열의 전체구간 평균 — 아래 시계열 표의 끝 행과 같은 값
                rx_med = diag.median_series(rx_runs)
                rx_overall = diag.series_overall(rx_med)
                te_med = diag.median_series(te_runs)     # R_x 의 분자 (T_est 노드평균)
                tm_med = diag.median_series(tm_runs)     # R_x 의 분모 (T_max 노드평균, 단조 증가)

                # ---- 혼잡 신호 그림 — ADT 와 같은 100 tick 해상도로 그대로 그린다 ----
                # 위 축은 틱 단위 두 값(T_est 실선 / T_max 파선), 아래 축은 그 비율인 R_x.
                ax_sig_t.plot(x_axis, te_med, color=color, linestyle='-',
                              linewidth=2.0, label=f'{label} T_est')
                ax_sig_t.plot(x_axis, tm_med, color=color, linestyle='--',
                              linewidth=2.0, label=f'{label} T_max')
                ax_sig_r.plot(x_axis, rx_med, color=color, linestyle='-',
                              linewidth=2.0, label=label)
                print(f"    [지표] ADT={overall_adt:6.2f} IQR폭={overall_iqr:6.2f} "
                      f"에코={echo_cost:.2f} 변경률={switch_rate * 100:.1f}% R_x={rx_overall:.3f}")
                md.write(f"\n- 전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱) : {overall_adt:.2f}\n")
                md.write(f"- 전체 IQR 폭 (띠 평균 폭 // 단위 = 틱) : {overall_iqr:.2f}\n")
                md.write(f"- 라우팅 결정당 이웃 조회 수 (전체 평균 // 단위 = 개/결정) : {echo_cost:.2f}\n")
                # 위 비율의 분자·분모. 시드마다 실행 전체 36개 노드 합을 구하고 시드 10개를 평균.
                md.write(f"- 이웃 조회 총횟수 (시드 10개 평균 // 단위 = 회) : {np.mean(echo_totals):,.0f}\n")
                md.write(f"- 라우팅 결정 총횟수 (시드 10개 평균 // 단위 = 회) : {np.mean(route_totals):,.0f}\n")
                md.write(f"- 경로 변경률 (전체 평균 // 단위 = %) : {switch_rate * 100:.1f}%\n")
                md.write(f"- R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율) : {rx_overall:.3f}\n")
                # R_x 의 분자·분모. 둘 다 전체구간 시간평균 (R_x 와 같은 요약 규칙).
                md.write(f"- T_est (전체 평균, R_x 분자 // 단위 = 틱) : {diag.series_overall(te_med):.2f}\n")
                md.write(f"- T_max (전체 평균, R_x 분모 // 단위 = 틱) : {diag.series_overall(tm_med):.2f}\n\n")

                # 진단 시계열 (구간 평균 + 전체 평균) + 링크 사용량 수치
                diag.write_diag_series_md(md, label,
                                          rx_med,
                                          diag.median_series(echo_runs),
                                          diag.median_series(switch_runs),
                                          echo_cost, switch_rate,
                                          t_est_med=te_med, t_max_med=tm_med,
                                          stat_interval=STAT_INTERVAL)
                link_sum = diag.sum_link_usage(link_dicts)
                md.write(f"#### 링크 사용량 ({label}, 통과량 순, 시드 합)\n\n")
                md.write(diag.link_usage_md(link_sum))
                md.write("\n")
                per_algo_link[algo] = link_sum

            ax.set_title(f"λ={lam}", fontsize=20)
            ax.set_xlabel('Simulator Time', fontsize=18)
            ax.set_ylabel('Average Delivery Time', fontsize=18)
            ax.tick_params(axis='both', labelsize=16)
            ax.legend(loc='best', fontsize=17)
            ax.grid(True, alpha=0.3)
            link_data.append((lam, per_algo_link))

            # ---- 혼잡 신호 패널 마무리 ----
            ax_sig_t.set_title(f"λ={lam}", fontsize=20)
            ax_sig_t.set_ylabel('T_est / T_max (tick)', fontsize=18)
            ax_sig_t.tick_params(axis='y', labelsize=16)
            ax_sig_t.legend(loc='best', fontsize=15)
            ax_sig_t.grid(True, alpha=0.3)
            # AQPRICE 의 Echo 가격은 max(0.1, 1 - R_x) 이고 포인트 상한이 0.5 이므로
            # R_x 가 0.5 아래로 내려가면 아무리 모아도 Full Echo 를 살 수 없다.
            ax_sig_r.axhline(0.5, color='#666666', linestyle=':', linewidth=2.0,
                             label='R_x = 0.5 (point cap)')
            ax_sig_r.set_ylim(0.0, 1.0)
            ax_sig_r.set_xlabel('Simulator Time', fontsize=18)
            ax_sig_r.set_ylabel('R_x', fontsize=18)
            ax_sig_r.tick_params(axis='both', labelsize=16)
            ax_sig_r.legend(loc='best', fontsize=15)
            ax_sig_r.grid(True, alpha=0.3)

    # rect 위쪽을 0.96 으로 잘라 suptitle 자리를 남긴다 (제목이 패널 제목과 겹치지 않게)
    # 그림이 둘이므로 plt 전역이 아니라 각 figure 객체를 명시해 저장한다.
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PNG_PATH, dpi=150)
    print(f"\n결과 저장: {PNG_PATH}")
    plt.close(fig)

    # 신호 그림은 중첩 격자라 tight_layout 을 쓰지 않는다 (여백은 격자 생성 때 지정).
    figS.savefig(SIGNAL_PNG_PATH, dpi=150)
    print(f"혼잡 신호 저장: {SIGNAL_PNG_PATH}")
    plt.close(figS)

    # ---- 링크 사용량 다이어그램 (그리드: 행=알고리즘, 열=부하). grid 만 ----
    if GRID_NUM_NODES == 36:
        nrows, ncols = len(ALGORITHMS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        figL.suptitle(
            f"{topo_name(GRID_NUM_NODES)} — link usage  [color & width = usage level]",
            fontsize=20)
        vmin, vmax = diag.link_usage_range([per[a] for (lam, per) in link_data for a in per])
        for r, algo in enumerate(ALGORITHMS):
            for c, (lam, per_algo) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_algo[algo],
                                         GRID_NUM_NODES, f"{LABELS[algo]}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
