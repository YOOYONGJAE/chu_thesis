# =============================================================================
# [요약] AQPRICE + Dual Reinforcement Q-Routing 역방향 갱신 효과 비교 — VARIANTS 로 조건 선택
# - 같은 aqprice 를 여러 설정으로 비교 (Inline·c·포인트 게이트·라우트메모리 미사용은 모두 동일):
#     AQPRICE           : 정식 (역방향 갱신 없음)
#     +DRQ-qmode(eta)   : Q-routing 모드에서만 y_star 갱신, 학습률 eta     (실험 5·7)
#     +DRQ-qmode(eta2)  : Q-routing 모드에서만 y_star 갱신, 학습률 R_x·k   (실험 5·7)
#     +DRQ-all(eta)     : Full Echo 모드에서 에코 이웃 전부 갱신, 학습률 eta
#     +DRQ-all(eta2)    : Full Echo 모드에서 에코 이웃 전부 갱신, 학습률 R_x·k
#     +DRQ-ystar(eta)   : Full Echo 모드에서 y_star 만 갱신, 학습률 eta
#     +DRQ-ystar(eta2)  : Full Echo 모드에서 y_star 만 갱신, 학습률 R_x·k
#   (aqprice_dual = 학습률 'eta'/'eta2', aqprice_dual_fe = Full Echo 범위 'all'/'ystar')
# - 역방향 갱신 : 노드 n 이 Q[src][x] 를 'n 의 큐 길이 + 1 + (x 의 모든 이웃 중 최소 src 추정)'
#   쪽으로 갱신한다. 포인트 부족(Q-routing 모드)에서는 네 DRQ 조건 모두 y_star 하나만 갱신한다.
#   eta2 의 R_x 는 갱신받는 노드 n 의 값. (node.py 의 dual_backward_update)
# - 이전 버전 결과 (같은 스크립트의 앞선 설정):
#     result_aqprice_dual_compare(...)         : 에코 이웃 전부 갱신, 큐 항 없음, eta / eta2
#     result_aqprice_dual_q_compare(...)       : 에코 이웃 전부 갱신, 큐 항 있음, eta2
#     result_aqprice_dual_chosen_compare(...)  : y_star 만 갱신 (두 모드), 큐 항 있음, eta / eta2
#     result_aqprice_dual_qmode_compare(...)   : y_star 만, Q-routing 모드만, x 가 y_star 를 빼고 계산
#     result_aqprice_dual_allmin_compare(...)  : y_star 만, Q-routing 모드만, x 는 모든 이웃 중 최솟값
# - 6x6 grid, 10 시드, EXPERIMENTS 의 부하
#     result_aqprice_dual_fe_compare(...)      : 위 + Full Echo 모드에서도 갱신 (all / ystar)
# - 산출물: _BASENAME 참고 (실험마다 이름을 바꿔 이전 결과를 덮어쓰지 않게 한다)
# =============================================================================
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트 (그래프 한글 깨짐 방지)
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from simulator import Simulator
import diagnostics as diag
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_sub_test_grid_full import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY
# from topology_nsfnet import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# -------------------------------------------------------------------------
# 파라미터 (변형별로 키 하나씩만 덮어씀)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3      # AQRERM 용 키. aqprice 는 aqprice_L 을 따로 읽으므로 여기 값에 영향받지 않는다.
C   = 0.22
BASE_PARAMS = {'eta': ETA, 'k': K, 'L': L, 'c': C}

TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}


def topo_name(n):
    """토폴로지 이름 자동 판별. 노드 수가 같은 격자 두 종은 링크 수로 가른다.
    (import 한 줄만 바꾸면 제목과 파일명이 함께 따라온다.)"""
    return diag.topo_name(n, GRID_ADJACENCY)


# -------------------------------------------------------------------------
# 변형 정의 — 정식을 기준으로 역방향 갱신만 추가
#   qmode_* : Q-routing 모드에서만 갱신 (실험 5 설정, aqprice_dual_fe 없음)
#   all_* / ystar_* : Full Echo 모드에서도 갱신 (실험 6 설정)
# 돌릴 조건은 VARIANTS 로 고른다 (REF_KEY 는 항상 남길 것).
# -------------------------------------------------------------------------
VARIANTS = ['aqprice', 'qmode_eta', 'qmode_eta2']   # 실험 7 : 실험 5 설정을 낮은 부하에서
REF_KEY  = 'aqprice'          # 정식. 차이 표의 기준

VARIANT_CFG = {
    'aqprice':    {},                                                    # 정식 (기본값 그대로)
    'qmode_eta':  {'aqprice_dual': 'eta'},                               # Q-routing 모드만, eta
    'qmode_eta2': {'aqprice_dual': 'eta2'},                              # Q-routing 모드만, R_x·k
    'all_eta':    {'aqprice_dual': 'eta',  'aqprice_dual_fe': 'all'},     # 이웃 전부, eta
    'all_eta2':   {'aqprice_dual': 'eta2', 'aqprice_dual_fe': 'all'},     # 이웃 전부, R_x·k
    'ystar_eta':  {'aqprice_dual': 'eta',  'aqprice_dual_fe': 'ystar'},   # y_star 만, eta
    'ystar_eta2': {'aqprice_dual': 'eta2', 'aqprice_dual_fe': 'ystar'},   # y_star 만, R_x·k
}
# 그래프 범례와 md 표에 그대로 나가는 이름 (탐색용 임시 표기)
LABELS = {
    'aqprice':    'AQPRICE',
    'qmode_eta':  'DRQ-qmode(eta)',
    'qmode_eta2': 'DRQ-qmode(eta2)',
    'all_eta':    'DRQ-all(eta)',
    'all_eta2':   'DRQ-all(eta2)',
    'ystar_eta':  'DRQ-ystar(eta)',
    'ystar_eta2': 'DRQ-ystar(eta2)',
}
LABELS_SHORT = dict(LABELS)
# 적녹 구분에 의존하지 않는 색 (Okabe-Ito) + 선 모양 이중 인코딩
# 선 모양 : 학습률 (eta 파선 / eta2 일점쇄선). 색 : 한 실행 안에서 조건마다 다르게 (신호 그림은 선 모양을 T_est/T_max 에 씀).
COLORS = {
    'aqprice':    '#56B4E9',   # 하늘색 (정식)
    'qmode_eta':  '#D55E00',   # 주홍
    'qmode_eta2': '#0072B2',   # 진파랑
    'all_eta':    '#D55E00',   # 주홍
    'all_eta2':   '#E69F00',   # 주황
    'ystar_eta':  '#0072B2',   # 진파랑
    'ystar_eta2': '#CC79A7',   # 분홍보라
}
LINESTYLES = {
    'aqprice':    '-',
    'qmode_eta':  '--',
    'qmode_eta2': '-.',
    'all_eta':    '--',
    'all_eta2':   '-.',
    'ystar_eta':  '--',
    'ystar_eta2': '-.',
}

SEEDS = list(range(100, 1001, 100))   # 10 개 시드
STAT_INTERVAL = 100

EXPERIMENTS = [
    # 4 단계 부하 — 저 / 중저 / 중고 / 고. 다른 비교 실험과 동일하게 맞춘다.
    {'lam': 2.5, 'total_ticks': 20000},   # 저부하
    {'lam': 3.0, 'total_ticks': 20000},   # 중저부하
    # {'lam': 3.5, 'total_ticks': 20000},   # 중고부하
    # {'lam': 3.8, 'total_ticks': 20000},   # 고부하
]

# 파일명 태그: (토폴로지, 부하, 틱수). 토폴로지·틱수는 자동, 부하 태그는 수동.
TOPO_TAG   = diag.topo_tag(GRID_NUM_NODES, GRID_ADJACENCY)
RESULT_TAG = 'LOW'    # 부하 표기 (수동). λ=2.5, 3.0
_BASENAME  = f'result_aqprice_dual_allmin_compare({TOPO_TAG},{RESULT_TAG},{EXPERIMENTS[0]["total_ticks"]})'
MD_PATH       = f'{_BASENAME}.md'
PNG_PATH      = f'{_BASENAME}.png'
LINK_PNG_PATH = f'{_BASENAME}_linkuse.png'
SIGNAL_PNG_PATH = f'{_BASENAME}_signal.png'   # R_x / T_est / T_max 전용 그림


# -------------------------------------------------------------------------
# 단일 (variant, lam, seed) 실행 헬퍼 — 항상 aqprice, 변형별 키 하나만 덮어씀
# -------------------------------------------------------------------------
def run_one(variant, lam, total_ticks, seed):
    random.seed(seed)
    np.random.seed(seed)
    params = dict(BASE_PARAMS)
    params.update(VARIANT_CFG[variant])
    sim = Simulator(algorithm='aqprice', params=params, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(lam=lam, total_ticks=total_ticks, stat_interval=STAT_INTERVAL)
    return sim, adt


# -------------------------------------------------------------------------
# Metric 계산 — 전체구간 ADT 요약 2 지표
# - 정착 임계값을 정당화할 필요가 없도록 구간을 나누지 않고 실행 전체를 요약
# - 표의 두 숫자는 ADT 그래프의 실선(중앙값)·띠(IQR)를 전체 구간 평균낸 값과 같다
# -------------------------------------------------------------------------
def compute_metrics(adt_arr):
    """전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱) + 전체 IQR 폭 (띠 평균 폭 // 단위 = 틱) 을 dict 로 반환."""
    med  = np.nanmedian(adt_arr, axis=0)
    band = (np.nanpercentile(adt_arr, 75, axis=0)
            - np.nanpercentile(adt_arr, 25, axis=0))
    return {
        'overall':     float(np.nanmean(med)),
        'overall_iqr': float(np.nanmean(band)),
    }


# -------------------------------------------------------------------------
# 한 부하 실험 (한 패널)
# -------------------------------------------------------------------------
def run_lambda(ax, sig_pair, lam, total_ticks, md):
    """sig_pair : (위 축, 아래 축) — 위는 T_est/T_max, 아래는 R_x 를 그린다."""
    x_axis = np.arange(1, total_ticks // STAT_INTERVAL + 1) * STAT_INTERVAL

    md.write(f"## λ={lam} ({total_ticks} ticks)\n\n")
    print(f"\n========== λ={lam} ==========")

    results = {}
    for variant in VARIANTS:
        print(f"\n--- {LABELS[variant]} ---")
        adt_runs = []
        echo_costs = []
        switch_rates = []
        rx_runs, echo_runs, switch_runs, link_dicts = [], [], [], []
        te_runs, tm_runs = [], []          # R_x 의 분자·분모 시계열
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
            sw = (sim.total_switches / sim.total_route_calls
                  if sim.total_route_calls > 0 else 0.0)
            switch_rates.append(sw)
            rx_runs.append(sim.r_ratio_series)
            te_runs.append(sim.t_est_series)
            tm_runs.append(sim.t_max_series)
            echo_runs.append(sim.echo_per_decision_series)
            switch_runs.append(sim.switch_rate_series)
            link_dicts.append(sim.link_usage)
        adt_arr = np.array(adt_runs)
        results[variant] = {
            'adt_arr':   adt_arr,
            'median':    np.nanmedian(adt_arr, axis=0),
            'q25':       np.nanpercentile(adt_arr, 25, axis=0),
            'q75':       np.nanpercentile(adt_arr, 75, axis=0),
            'echo_cost':   float(np.mean(echo_costs)),      # 시드 10개 값의 평균 (패킷 전달 통계와 같은 규칙)
            'switch_rate': float(np.mean(switch_rates)),
            'rx_med':     diag.median_series(rx_runs),
            't_est_med':  diag.median_series(te_runs),
            't_max_med':  diag.median_series(tm_runs),
            'echo_med':   diag.median_series(echo_runs),
            'switch_med': diag.median_series(switch_runs),
            'link_sum':   diag.sum_link_usage(link_dicts),
            'gen_mean':  float(np.mean(gens)),
            'dlv_mean':  float(np.mean(dlvs)),
            'und_mean':  float(np.mean(unds)),
            'rate_mean': float(np.mean(rates)),
            'rate_max':  float(np.max(rates)),
            'rate_min':  float(np.min(rates)),
        }

    metrics = {v: compute_metrics(results[v]['adt_arr']) for v in VARIANTS}

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
                    label=LABELS[variant], color=COLORS[variant],
                    linestyle=LINESTYLES[variant], linewidth=2.0)

    ax.set_title(f"λ={lam} ({total_ticks} ticks)", fontsize=20)
    ax.set_xlabel('Simulator Time', fontsize=18)
    ax.set_ylabel('Average Delivery Time', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc='upper right', fontsize=17)
    ax.grid(True, alpha=0.3)

    # ---- 혼잡 신호 그림 — ADT 와 같은 100 tick 해상도 (main_lineage 와 같은 형식) ----
    # 변형은 색으로, 위 축의 두 값은 선 모양으로 구분한다 (T_est 실선 / T_max 파선).
    ax_sig_t, ax_sig_r = sig_pair
    for variant in VARIANTS:
        r = results[variant]
        ax_sig_t.plot(x_axis, r['t_est_med'], color=COLORS[variant], linestyle='-',
                      linewidth=2.0, label=f'{LABELS[variant]} T_est')
        ax_sig_t.plot(x_axis, r['t_max_med'], color=COLORS[variant], linestyle='--',
                      linewidth=2.0, label=f'{LABELS[variant]} T_max')
        ax_sig_r.plot(x_axis, r['rx_med'], color=COLORS[variant], linestyle='-',
                      linewidth=2.0, label=LABELS[variant])
    ax_sig_t.set_title(f"λ={lam}", fontsize=20)
    ax_sig_t.set_ylabel('T_est / T_max (tick)', fontsize=18)
    ax_sig_t.tick_params(axis='y', labelsize=16)
    ax_sig_t.legend(loc='best', fontsize=15)
    ax_sig_t.grid(True, alpha=0.3)
    # Echo 가격 max(0.1, 1 - R_x) 와 포인트 상한 0.5 때문에 R_x < 0.5 면 Full Echo 를 살 수 없다.
    ax_sig_r.axhline(0.5, color='#666666', linestyle=':', linewidth=2.0,
                     label='R_x = 0.5 (point cap)')
    ax_sig_r.set_ylim(0.0, 1.0)
    ax_sig_r.set_xlabel('Simulator Time', fontsize=18)
    ax_sig_r.set_ylabel('R_x', fontsize=18)
    ax_sig_r.tick_params(axis='both', labelsize=16)
    ax_sig_r.legend(loc='best', fontsize=15)
    ax_sig_r.grid(True, alpha=0.3)

    # ---- 지표 접근 헬퍼 (원값 표와 차이 표가 같은 정의를 공유하도록 한 곳에 모음) ----
    def v_adt(v):    return metrics[v]['overall']
    def v_iqr(v):    return metrics[v]['overall_iqr']
    def v_echo(v):   return results[v]['echo_cost']
    def v_switch(v): return results[v]['switch_rate'] * 100.0     # 백분율
    def v_rx(v):     return diag.series_overall(results[v]['rx_med'])
    def v_te(v):     return diag.series_overall(results[v]['t_est_med'])
    def v_tm(v):     return diag.series_overall(results[v]['t_max_med'])

    cols  = " | ".join(LABELS_SHORT[v] for v in VARIANTS)
    align = "|---" + "|---:" * len(VARIANTS) + "|\n"

    def row(name, fmt):
        return f"| {name} | " + " | ".join(fmt(v) for v in VARIANTS) + " |\n"

    # ---- MD 표 1 : 변형별 원값 ----
    md.write(f"| 지표 | {cols} |\n")
    md.write(align)
    md.write(row("전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱)", lambda v: f"{v_adt(v):.2f}"))
    md.write(row("전체 IQR 폭 (띠 평균 폭 // 단위 = 틱)",              lambda v: f"{v_iqr(v):.2f}"))
    md.write(row("라우팅 결정당 이웃 조회 수 (전체 평균 // 단위 = 개/결정)", lambda v: f"{v_echo(v):.2f}"))
    md.write(row("경로 변경률 (전체 평균 // 단위 = %)",                lambda v: f"{v_switch(v):.1f}%"))
    md.write(row("R_x (전체 평균, 혼잡 신호 // 단위 없음, 0~1 비율)",   lambda v: f"{v_rx(v):.3f}"))
    md.write(row("T_est (전체 평균, R_x 분자 // 단위 = 틱)",           lambda v: f"{v_te(v):.2f}"))
    md.write(row("T_max (전체 평균, R_x 분모 // 단위 = 틱)",           lambda v: f"{v_tm(v):.2f}"))
    md.write("\n")

    # ---- MD 표 2 : 정식 대비 차이 (DRQ 조건들만) ----
    # 모두 (DRQ 조건 − 정식) 절대 차이. 음수면 역방향 갱신을 넣어 값이 작아진 것이다.
    # ADT·IQR·경로 변경률은 작을수록 좋으므로, 음수면 역방향 갱신이 도움이 됐다는 뜻이 된다.
    alts = [v for v in VARIANTS if v != REF_KEY]
    if alts:
        dcols  = " | ".join(f"{LABELS_SHORT[v]} − {LABELS_SHORT[REF_KEY]}" for v in alts)
        dalign = "|---" + "|---:" * len(alts) + "|\n"

        def drow(name, fmt):
            return f"| {name} | " + " | ".join(fmt(v) for v in alts) + " |\n"

        md.write(f"#### 정식 대비 차이\n\n")
        md.write(f"| 지표 | {dcols} |\n")
        md.write(dalign)
        md.write(drow("전체 평균 ADT (틱)",
                      lambda v: f"{v_adt(v) - v_adt(REF_KEY):+.2f}"))
        md.write(drow("전체 IQR 폭 (틱)",
                      lambda v: f"{v_iqr(v) - v_iqr(REF_KEY):+.2f}"))
        md.write(drow("라우팅 결정당 이웃 조회 수 (개/결정)",
                      lambda v: f"{v_echo(v) - v_echo(REF_KEY):+.2f}"))
        # 경로 변경률은 비율이라 백분율로, 차이는 두 백분율의 차이이므로 %p 로 적는다.
        md.write(drow("경로 변경률 (%p)",
                      lambda v: f"{v_switch(v) - v_switch(REF_KEY):+.1f}%p"))
        md.write(drow("R_x",
                      lambda v: f"{v_rx(v) - v_rx(REF_KEY):+.3f}"))
        md.write(drow("T_est (틱)",
                      lambda v: f"{v_te(v) - v_te(REF_KEY):+.2f}"))
        md.write(drow("T_max (틱)",
                      lambda v: f"{v_tm(v) - v_tm(REF_KEY):+.2f}"))
        md.write("\n")

    # ---- MD 진단 시계열 (변형별 구간 평균 + 전체 평균) ----
    for variant in VARIANTS:
        r = results[variant]
        diag.write_diag_series_md(md, LABELS[variant],
                                  r['rx_med'], r['echo_med'], r['switch_med'],
                                  r['echo_cost'], r['switch_rate'],
                                  t_est_med=r['t_est_med'], t_max_med=r['t_max_med'],
                                  stat_interval=STAT_INTERVAL)

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

    # ---- MD 링크 사용량 수치 (변형별, 통과량 순) ----
    for variant in VARIANTS:
        md.write(f"#### 링크 사용량 ({LABELS[variant]}, 통과량 순, 시드 합)\n\n")
        md.write(diag.link_usage_md(results[variant]['link_sum']))
        md.write("\n")

    # ---- 콘솔 출력 ----
    head = " | ".join(f"{LABELS_SHORT[v]:>10}" for v in VARIANTS)
    print(f"  {'지표':<26} | {head}")

    def prow(name, fmt):
        vals = " | ".join(f"{fmt(v):>10}" for v in VARIANTS)
        print(f"  {name:<26} | {vals}")

    prow("전체 평균 ADT",              lambda v: f"{v_adt(v):.2f}")
    prow("전체 IQR 폭",                lambda v: f"{v_iqr(v):.2f}")
    prow("라우팅 결정당 이웃 조회 수", lambda v: f"{v_echo(v):.2f}")
    print(f"  {'→ ADT 차이(정식 대비)':<26} | "
          + " | ".join(f"{(v_adt(v) - v_adt(REF_KEY)):>+10.2f}" for v in VARIANTS))

    # 링크 다이어그램용: 변형별 시드 합산 link_usage 반환
    return {v: results[v]['link_sum'] for v in VARIANTS}


# -------------------------------------------------------------------------
# 메인
# -------------------------------------------------------------------------
def run_all():
    # 패널 배치 — 부하 4개면 2x2. 패널 하나당 크기가 고정이라 실험 수가 달라져도 비율 유지.
    fig, axes = diag.make_panel_grid(len(EXPERIMENTS))
    fig.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — AQPRICE + Dual Reinforcement backward update  "
        f"(c={C}, seeds n={len(SEEDS)})",
        fontsize=22,
        y=0.99,
    )
    # 혼잡 신호 전용 그림 — 부하마다 위(T_est/T_max) · 아래(R_x) 두 축
    figS, sig_axes = diag.make_signal_panel_grid(len(EXPERIMENTS))
    figS.suptitle(
        f"{topo_name(GRID_NUM_NODES)} — AQPRICE + Dual Reinforcement backward update  "
        f"(congestion signal: T_est / T_max / R_x, seeds n={len(SEEDS)})",
        fontsize=22,
        y=0.99,
    )

    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# AQPRICE + Dual Reinforcement 역방향 갱신 비교\n\n')
        md.write(f'- Seeds: {SEEDS}\n')
        md.write(f'- Variants: {[LABELS[v] for v in VARIANTS]}\n')
        md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
        md.write(f'- 변형별 덮어쓴 키: '
                 + ', '.join(f'{LABELS_SHORT[v]}={VARIANT_CFG[v] or "기본값"}' for v in VARIANTS)
                 + '\n')
        md.write(f'- STAT_INTERVAL: {STAT_INTERVAL}\n\n')

        md.write('## 출력 구성\n\n')
        md.write('- 모든 조건의 알고리즘은 aqprice 이고 aqprice_dual (학습률) 과 aqprice_dual_fe (Full Echo 범위) 만 다르다. '
                 'Inline Echo, 큐 페널티, 포인트 게이트, 라우트 메모리 미사용, 요청 노드 제외는 모두 동일하다.\n')
        md.write('- 역방향 갱신 : 노드 x 가 출발지 src 인 패킷을 처리할 때, 갱신받는 노드 n 이 '
                 'Q[src][x] 를 (n 의 큐 길이) + 1 + (x 의 모든 이웃 중 최소인 src 까지의 추정) 쪽으로 갱신한다.\n')
        md.write('- 갱신받는 노드 : 포인트 부족(Q-routing 모드)에서는 모든 DRQ 조건이 y_star 하나. '
                 'Full Echo 모드에서는 DRQ-qmode 는 갱신 없음, DRQ-all 은 에코를 받은 이웃 전부, DRQ-ystar 는 y_star 하나.\n')
        md.write('- 큐 길이 : 원 논문 (Kumar & Miikkulainen, 1997) 식 (4) 의 q_y 에 해당. '
                 '노드는 tick 당 패킷 1 개를 처리하므로 현재 큐 길이를 예상 대기 틱으로 쓴다.\n')
        md.write('- 학습률 : (eta) 는 eta, (eta2) 는 갱신받는 노드 자신의 R_x · k 를 쓴다.\n')
        md.write('- 역방향 정보는 패킷·에코 요청에 얹어 보내는 것으로 보아 라우팅 결정당 이웃 조회 수에 더하지 않는다.\n')
        md.write('- 정식 대비 차이 표는 모두 (DRQ 조건 − 정식) 절대 차이다. '
                 'ADT·IQR 폭·경로 변경률은 작을수록 좋으므로, 음수면 역방향 갱신을 넣어 좋아졌다는 뜻이다.\n')
        md.write('- 진단 시계열 : R_x / 라우팅 결정당 이웃 조회 수 / 경로 변경률 구간 평균 + 끝의 전체 평균.\n')
        md.write('- 링크 사용량 : 별도 PNG 에 grid 링크 굵기·색 + md 통과량 표.\n\n')

        diag.write_metric_guide_md(md)

        link_data = []
        for ax, sig_pair, exp in zip(axes, sig_axes, EXPERIMENTS):
            per_var = run_lambda(ax, sig_pair, exp['lam'], exp['total_ticks'], md)
            link_data.append((exp['lam'], per_var))

    # rect 위쪽을 0.96 으로 잘라 suptitle 자리를 남긴다 (제목이 패널 제목과 겹치지 않게)
    # 그림이 둘이므로 plt 전역이 아니라 각 figure 객체를 명시해 저장한다.
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"\n결과 PNG : {PNG_PATH}")
    print(f"결과 MD  : {MD_PATH}")

    # 신호 그림은 중첩 격자라 tight_layout 을 쓰지 않는다 (여백은 격자 생성 때 지정).
    figS.savefig(SIGNAL_PNG_PATH, dpi=150)
    plt.close(figS)
    print(f"결과 신호 PNG : {SIGNAL_PNG_PATH}")

    # ---- 링크 사용량 다이어그램 (그리드: 행=변형, 열=부하). grid 만 ----
    if GRID_NUM_NODES == 36:
        nrows, ncols = len(VARIANTS), len(link_data)
        figL, axesL = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        figL.suptitle(
            f"{topo_name(GRID_NUM_NODES)} — link usage  [color & width = usage level]",
            fontsize=20)
        vmin, vmax = diag.link_usage_range([per[v] for (lam, per) in link_data for v in per])
        for r, variant in enumerate(VARIANTS):
            for c, (lam, per_var) in enumerate(link_data):
                diag.plot_link_thickness(axesL[r][c], per_var[variant],
                                         GRID_NUM_NODES,
                                         f"{LABELS_SHORT[variant]}  λ={lam}",
                                         vmin=vmin, vmax=vmax)
        diag.add_link_colorbar(figL, axesL.ravel().tolist(), vmin, vmax)
        figL.savefig(LINK_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close(figL)
        print(f"결과 링크 PNG : {LINK_PNG_PATH}")
    else:
        print("링크 다이어그램은 grid 에서만 생성 — 현재 grid 아님 (수치는 md 참조)")


if __name__ == '__main__':
    run_all()
