# =============================================================================
# [요약] 비교 스크립트 공용 진단 출력 헬퍼
# - 여러 비교 스크립트가 같은 방식으로 진단을 뽑도록 함수만 모아둔 모듈.
# - 링크 사용량 (그림은 다이어그램, 수치는 md):
#     sum_link_usage(dicts)            : 여러 시드 link_usage dict 합산
#     plot_link_thickness(ax, lu, N)   : grid 위 링크를 '점선 + 통과량 비례 굵기' 로 (색 아님)
#     link_usage_md(lu)                : 전체 링크 통과량을 정렬해 md 표 문자열로
# - 진단 시계열 (수치 표):
#     median_series(list_of_arrays)    : 시드별 시계열을 윈도우별 중앙값으로 합침
#     chunk_avg(arr, n)                : 1D 시계열을 n 구간 평균으로 압축
#     series_overall(med)              : 시계열의 전체구간 시간평균 (성능표와 끝 행이 공유)
#     write_diag_series_md(...)        : R_x / T_est / T_max / 에코 / 변경률 구간 평균 표
# - 링크 통과량은 선 두께(굵을수록 많음) + 색(연한 하늘색→빨강)으로 이중 표현.
# =============================================================================
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 링크 통과량 색: 연한 하늘색 → 파랑 → 주황 → 빨강 (굵기와 함께 이중 인코딩).
# 적녹색약 대비: 저~중 구간을 파랑 계열로 확실히 벌리고, 고 구간만 주황→빨강. 굵기가 크기를 한 번 더 보강.
LINK_CMAP = LinearSegmentedColormap.from_list(
    'linkuse', ['#d6ecfb', '#2b7bba', '#f39c12', '#e60023'])


# -------------------------------------------------------------------------
# link_usage 집계
# -------------------------------------------------------------------------
def sum_link_usage(dicts):
    """여러 시드의 link_usage dict 를 하나로 합산해 반환."""
    total = {}
    for d in dicts:
        for k, c in d.items():
            total[k] = total.get(k, 0) + c
    return total


def link_usage_sorted(link_usage):
    """link_usage dict 를 통과량 많은 순으로 정렬한 [(u,v), count] 리스트로 반환."""
    return sorted(link_usage.items(), key=lambda kv: kv[1], reverse=True)


def link_usage_md(link_usage):
    """전체 링크를 통과량 많은 순으로 정렬해 md 표 문자열로 반환 (수치는 md 로 뺄 때 사용)."""
    items = link_usage_sorted(link_usage)
    out = ["| 링크 | 통과 횟수 (시드 합) |\n", "|---|---:|\n"]
    for (u, v), c in items:
        out.append(f"| ({u}, {v}) | {c} |\n")
    return "".join(out)


# -------------------------------------------------------------------------
# grid 링크 굵기 플롯 (색 아님, 점선)
# -------------------------------------------------------------------------
def grid_positions(num_nodes, cols=6):
    """grid 노드 번호 → (x, y) 좌표. row=id//cols(위가 0행), col=id%cols.
    y 는 0행이 위로 가도록 뒤집는다."""
    rows = (num_nodes + cols - 1) // cols
    return {i: (i % cols, rows - 1 - i // cols) for i in range(num_nodes)}


N_LINK_LEVELS = 7   # 통과량을 몇 개 레벨로 이산화할지 (전역 vmin~vmax 를 균등 분할)


def level_colors(n_levels=N_LINK_LEVELS):
    """레벨별 고정 색 (LINK_CMAP 을 밴드 중앙에서 샘플)."""
    return [LINK_CMAP((i + 0.5) / n_levels) for i in range(n_levels)]


def link_usage_range(dicts):
    """여러 link_usage dict 전체의 (최소, 최대) 통과량 반환 (전역 색 기준용)."""
    vals = [c for d in dicts for c in d.values()]
    return (min(vals), max(vals)) if vals else (0, 1)


def plot_link_thickness(ax, link_usage, num_nodes, title, cols=6,
                        max_lw=15.0, min_lw=1.0, n_levels=N_LINK_LEVELS,
                        vmin=None, vmax=None):
    """grid 토폴로지 위에 각 링크를 실선으로 그리되, 통과량을 '레벨별 고정 색 + 단계 굵기' 로 표현.
    vmin~vmax(보통 그림 전체 전역 최소~최대) 를 n_levels 구간으로 나눠, 같은 레벨이면 어느 패널에서든
    같은 색·굵기가 나온다(패널 간 색 비교 가능). vmin/vmax 를 안 주면 이 패널 자체 최소~최대를 쓴다.
    num_nodes 가 grid(36) 형태일 때만 의미가 있다."""
    pos = grid_positions(num_nodes, cols)
    vals = list(link_usage.values())
    if vmin is None or vmax is None:
        vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    span = (vmax - vmin) if vmax > vmin else 1
    colors = level_colors(n_levels)

    for (u, v), c in link_usage.items():
        if u not in pos or v not in pos:
            continue
        frac = min(max((c - vmin) / span, 0.0), 0.999999)
        band = min(int(frac * n_levels), n_levels - 1)     # 0 .. n_levels-1
        lw = min_lw + (max_lw - min_lw) * (band / max(1, n_levels - 1))
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1],
                linestyle='-',
                linewidth=lw, color=colors[band],
                solid_capstyle='round', zorder=1)

    for i, (x, y) in pos.items():
        ax.plot(x, y, 'o', color='white', markeredgecolor='black',
                markersize=15, zorder=3)
        ax.text(x, y, str(i), ha='center', va='center', fontsize=8, zorder=4)

    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.margins(0.08)
    ax.axis('off')


def add_link_colorbar(fig, axes, vmin, vmax, n_levels=N_LINK_LEVELS,
                      label='link usage (sum over seeds)'):
    """레벨별 이산 색 ↔ 통과량 구간을 보여주는 컬러바를 그림 옆에 추가."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    boundaries = np.linspace(vmin, vmax, n_levels + 1)
    cmap = ListedColormap(level_colors(n_levels))
    norm = BoundaryNorm(boundaries, cmap.N)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, boundaries=boundaries, ticks=boundaries,
                      fraction=0.025, pad=0.02)
    cb.set_label(label)
    return cb


# -------------------------------------------------------------------------
# 진단 시계열 (R_x / Full Echo 비율 / 경로 변경률)
# -------------------------------------------------------------------------
def median_series(list_of_arrays):
    """시드별 시계열 리스트를 (n_seeds, n_windows) 로 쌓아 윈도우별 중앙값(1D)으로 반환.
    빈 구간의 NaN 은 무시한다. 비어 있으면 None."""
    if not list_of_arrays:
        return None
    arr = np.array(list_of_arrays, dtype=float)
    return np.nanmedian(arr, axis=0)


def series_overall(med):
    """진단 시계열(윈도우별 시드 중앙값 1D)의 전체구간 시간평균.
    성능 비교표의 R_x 와 진단 시계열 표의 '전체 평균' 이 같은 값이 되도록 양쪽이 이 함수를 공유한다."""
    if med is None or len(med) == 0:
        return 0.0
    return float(np.nanmean(med))


def chunk_avg(arr, n_chunks):
    """1D 시계열을 n_chunks 개 구간으로 나눠 각 구간 평균(리스트) 반환. NaN 은 무시."""
    if arr is None or len(arr) == 0:
        return [0.0] * n_chunks
    arr = np.asarray(arr, dtype=float)
    idx_groups = np.array_split(np.arange(len(arr)), min(n_chunks, len(arr)))
    out = []
    for ix in idx_groups:
        vals = arr[ix]
        out.append(float(np.nanmean(vals)) if len(vals) else 0.0)
    return out


def make_panel_grid(n, panel_w=20, panel_h=12, max_cols=2):
    """실험 개수 n 에 맞춰 패널 격자를 만들고 (fig, 패널 리스트) 를 반환.
    최대 max_cols 열로 채우므로 부하 4개면 2x2, 2개면 1x2, 1개면 1x1 이 된다.
    패널 하나당 크기를 고정해 실험 수가 달라져도 각 패널의 가로세로 비율이 유지된다.
    격자에 남는 칸이 생기면(예: 3개 → 2x2) 그 칸은 축을 숨겨 빈자리로 둔다."""
    import math
    import matplotlib.pyplot as plt
    cols = max(1, min(n, max_cols))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(panel_w * cols, panel_h * rows),
                             squeeze=False)
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.axis('off')
    return fig, list(flat[:n])


METRIC_DOC = '지표_정의서.md'   # 산정 방법·주의사항을 모아둔 별도 문서


def write_metric_guide_md(md, doc=METRIC_DOC):
    """지표 정의를 한 줄씩만 md 상단에 쓴다 (6개 비교 스크립트 공통).
    계산 절차와 주의사항은 doc 에 모아두고 여기서는 정의만 실어 결과 md 를 짧게 유지한다."""
    md.write('## 지표 정의\n\n')
    md.write('- 전체 평균 ADT (틱) : ADT 그래프 중앙값 실선의 전체구간 시간평균. 작을수록 빠름.\n')
    md.write('- 전체 IQR 폭 (틱) : ADT 그래프 오차 막대 길이의 전체구간 평균. 작을수록 시드 간 일관.\n')
    md.write('- 결정당 이웃 정보 조회 수 (개/결정) : 이웃 조회 총횟수 / 라우팅 결정 총횟수, 시드 간 중앙값.\n')
    md.write('- 경로 변경률 (%) : next-hop 변경 총횟수 / 라우팅 결정 총횟수, 시드 간 중앙값. 차이 열은 %p.\n')
    md.write('- R_x (0~1) : 각 노드의 T_est/T_max 를 노드 평균한 혼잡 신호의 전체구간 시간평균.\n')
    md.write('- T_est (틱) : R_x 의 분자. 노드 평균의 전체구간 시간평균.\n')
    md.write('- T_max (틱) : R_x 의 분모. 노드 평균의 전체구간 시간평균.\n')
    md.write('- 패킷 전달 통계 : 생성/전달/미전달은 시드 평균 패킷 수, 전달률은 시드 평균과 최고·최저.\n')
    md.write('- 링크 사용량 : 각 링크를 패킷이 지나간 횟수(양방향 합)를 시드 합산.\n\n')
    md.write(f'각 지표의 산정 방법과 주의사항은 `{doc}` 참고.\n\n')



def write_diag_series_md(md, label, rx_med, echo_med, switch_med,
                         echo_scalar, switch_scalar, n_chunks=10,
                         t_est_med=None, t_max_med=None):
    """진단 시계열을 구간 평균 표로 md 에 쓴다.
    t_est_med / t_max_med 를 넘기면 R_x 의 분자·분모 열이 함께 나온다.
    세 열 모두 같은 규칙(구간 평균 + 끝 행은 전체 시간평균)을 쓴다 — R_x 의 전체 평균이
    실제로 나눗셈에 쓰인 분자·분모의 시간평균과 대응해야 분해가 성립하기 때문.
    에코·변경률의 끝 행만 총합 기준 스칼라라 10구간 평균과 다를 수 있다."""
    has_t = t_est_med is not None and t_max_med is not None

    md.write(f"#### 진단 시계열 (구간 평균, {label})\n\n")
    if has_t:
        md.write("| 구간 | R_x | T_est (틱) | T_max (틱) | "
                 "결정당 이웃 정보 조회 수 (개/결정) | 경로 변경률 (%) |\n")
        md.write("|---|---:|---:|---:|---:|---:|\n")
    else:
        md.write("| 구간 | R_x | 결정당 이웃 정보 조회 수 (개/결정) | 경로 변경률 (%) |\n")
        md.write("|---|---:|---:|---:|\n")

    rx_c = chunk_avg(rx_med, n_chunks)
    ec_c = chunk_avg(echo_med, n_chunks)
    sw_c = chunk_avg(switch_med, n_chunks)
    te_c = chunk_avg(t_est_med, n_chunks) if has_t else None
    tm_c = chunk_avg(t_max_med, n_chunks) if has_t else None

    for i in range(len(rx_c)):
        if has_t:
            md.write(f"| {i + 1} | {rx_c[i]:.3f} | {te_c[i]:.2f} | {tm_c[i]:.2f} | "
                     f"{ec_c[i]:.2f} | {sw_c[i] * 100:.1f}% |\n")
        else:
            md.write(f"| {i + 1} | {rx_c[i]:.3f} | {ec_c[i]:.2f} | {sw_c[i] * 100:.1f}% |\n")

    if has_t:
        md.write(f"| 전체 평균 | {series_overall(rx_med):.3f} | "
                 f"{series_overall(t_est_med):.2f} | {series_overall(t_max_med):.2f} | "
                 f"{echo_scalar:.2f} | {switch_scalar * 100:.1f}% |\n\n")
    else:
        md.write(f"| 전체 평균 | {series_overall(rx_med):.3f} | "
                 f"{echo_scalar:.2f} | {switch_scalar * 100:.1f}% |\n\n")
