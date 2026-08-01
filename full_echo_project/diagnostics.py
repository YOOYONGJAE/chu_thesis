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
#     write_diag_series_md(...)        : R_x / 결정당 에코 이웃 수 / 경로 변경률 구간 평균 표
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


def series_final(med):
    """단조 증가 시계열(T_max)의 최종값. 성능 비교표와 진단 시계열 끝 행이 같은 값을 쓰도록 공유한다."""
    if med is None or len(med) == 0:
        return 0.0
    return float(med[-1])


def chunk_last(arr, n_chunks):
    """1D 시계열을 n_chunks 구간으로 나눠 각 구간의 '마지막 값' 리스트를 반환.
    T_max 처럼 단조 증가하는 값에 쓴다. 단조 증가라 구간 마지막 값 = 구간 최댓값이며,
    평균을 쓰면 이미 고정된 뒤에도 계속 오르는 것처럼 보여 고정 시점이 흐려진다."""
    if arr is None or len(arr) == 0:
        return [0.0] * n_chunks
    arr = np.asarray(arr, dtype=float)
    idx_groups = np.array_split(np.arange(len(arr)), min(n_chunks, len(arr)))
    return [float(arr[ix[-1]]) if len(ix) else 0.0 for ix in idx_groups]


METRIC_DOC = '지표_정의서.md'   # 산정 방법·주의사항을 모아둔 별도 문서


def write_metric_guide_md(md, doc=METRIC_DOC):
    """지표 정의를 한 줄씩만 md 상단에 쓴다 (6개 비교 스크립트 공통).
    계산 절차와 주의사항은 doc 에 모아두고 여기서는 정의만 실어 결과 md 를 짧게 유지한다."""
    md.write('## 지표 정의\n\n')
    md.write('- 전체 평균 ADT (틱) : ADT 그래프 중앙값 실선의 전체구간 시간평균. 작을수록 빠름.\n')
    md.write('- 전체 IQR 폭 (틱) : ADT 그래프 오차 막대 길이의 전체구간 평균. 작을수록 시드 간 일관.\n')
    md.write('- 결정당 에코 이웃 수 (개/결정) : 이웃 조회 총횟수 / 라우팅 결정 총횟수, 시드 간 중앙값.\n')
    md.write('- 경로 변경률 (%) : next-hop 변경 총횟수 / 라우팅 결정 총횟수, 시드 간 중앙값. 차이 열은 %p.\n')
    md.write('- R_x (0~1) : 각 노드의 T_est/T_max 를 노드 평균한 혼잡 신호의 전체구간 시간평균.\n')
    md.write('- T_est (틱) : R_x 의 분자. 노드 평균의 전체구간 시간평균.\n')
    md.write('- T_max (틱) : R_x 의 분모. 실행 종료 시점의 값(줄어들지 않는 값이라 평균을 쓰지 않음).\n')
    md.write('- 패킷 전달 통계 : 생성/전달/미전달은 시드 평균 패킷 수, 전달률은 시드 평균과 최고·최저.\n')
    md.write('- 링크 사용량 : 각 링크를 패킷이 지나간 횟수(양방향 합)를 시드 합산.\n\n')
    md.write(f'각 지표의 산정 방법과 주의사항은 `{doc}` 참고.\n\n')


def _write_metric_guide_md_full(md):
    """[보관용] 정의서로 분리하기 전의 상세 안내. 현재는 호출하지 않는다."""
    md.write('## 지표 계산 방식\n\n')
    md.write('공통 : 각 실험은 10개 난수 시드로 반복한다. ADT 는 100 tick 구간마다 그 구간에 도착한 '
             '패킷들의 평균 전달시간이다. 성능 지표는 구간을 나누지 않고 실행 전체를 요약한다. '
             '전달시간 = 도착 tick + 1 − 생성 tick.\n\n')

    md.write('### 성능 비교표\n')
    md.write('- 전체 평균 ADT (중앙값 실선의 시간평균 // 단위 = 틱) : 매 100 tick 시점마다 10개 시드의 중앙값을 구하면 '
             'ADT 그래프의 실선이 되는데, 그 실선을 실행 전체 구간에 걸쳐 시간 평균한 값. '
             '즉 표의 이 숫자는 그래프 실선을 전체 구간 평균낸 것과 같다.\n')
    md.write('- 전체 IQR 폭 (띠 평균 폭, 작을수록 일관 // 단위 = 틱) : 같은 시점들에서 75퍼센타일 − 25퍼센타일 폭을 구하면 '
             'ADT 그래프의 IQR 띠가 되는데, 그 폭을 실행 전체 구간에 걸쳐 평균한 값. '
             '시드 간 산포가 실행 내내 얼마나 넓었는지를 하나의 숫자로 요약한다.\n')
    md.write('- 결정당 에코 이웃 수 (전체 평균 // 단위 = 개/결정) : 각 시드의 (이웃 조회 총횟수 ÷ 라우팅 결정 총횟수) 를 구한 뒤 '
             '시드 간 중앙값. 성능 대비 제어정보 조회 비용. Q-routing 의 단일 조회도 조회 수에 포함.\n')
    md.write('- 경로 변경률 (전체 평균 // 단위 = %) : 각 시드의 (next-hop 변경 총횟수 ÷ 라우팅 결정 총횟수) → 시드 간 중앙값. '
             '같은 목적지에 대해 직전과 다른 다음 홉을 고르면 변경 1회. 낮을수록 선택이 안정적. '
             '표에는 백분율로 적으며, 비교표의 차이 열은 두 백분율의 차이이므로 %p 로 표기한다.\n')
    md.write('- T_est (전체 평균 // 단위 = 틱) / T_max (최종값 // 단위 = 틱) : R_x 의 분자와 분모를 따로 뺀 값. '
             'T_est 는 노드 평균 시계열의 전체구간 시간평균이고, T_max 는 줄어들지 않는 값이라 평균 대신 실행 종료 시점의 값을 쓴다. '
             'R_x 가 높을 때 그것이 지연 추정이 실제로 커서인지(T_est 상승) 아니면 그 노드가 겪은 최댓값이 작아서인지(T_max 낮음)를 '
             '이 두 값으로 구분한다. R_x 는 비율의 평균이므로 T_est ÷ T_max 는 R_x 와 일치하지 않는다.\n')
    md.write('- R_x (전체 평균 // 단위 없음, 0~1 비율) : 각 노드의 T_est/T_max 를 노드 평균한 혼잡 신호를 100 tick 시점마다 시드 중앙값으로'
             '모은 뒤, 그 시계열을 실행 전체 구간에 걸쳐 시간 평균한 값(아래 진단 시계열 표의 전체 평균과 같은 값). '
             'T_max 는 그 노드가 관측한 T_est 의 최댓값을 계속 유지하므로, R_x 는 절대 혼잡도가 아니라 '
             '그 노드 자신의 최악 시점 대비 현재 혼잡도이다. 따라서 알고리즘 간 절대값 비교보다 '
             '부하에 따른 추이와 에코 비용과의 관계를 보는 데 사용한다.\n\n')

    md.write('### 진단 시계열 (구간 평균)\n')
    md.write('- 표의 각 열이 무엇인지 : R_x 는 혼잡 신호(동적 에코 가격의 재료), T_est 와 T_max 는 그 R_x 의 '
             '분자와 분모다. 세 열을 같이 보면 R_x 가 높은 것이 실제로 지연 추정이 커서인지(T_est 상승), '
             '아니면 그 노드가 겪은 최댓값이 작아서인지(T_max 낮음)를 구분할 수 있다.\n')
    md.write('- 결정당 에코 이웃 수 / 경로 변경률 : 그 구간의 (조회수 또는 변경수) ÷ 그 구간 결정수.\n')
    md.write('- 맨 끝 행 : R_x / T_est / 에코 / 변경률은 실행 전체 평균이고, T_max 만 실행 종료 시점의 최종값이다.\n\n')

    md.write('#### 진단 시계열 한 칸이 만들어지는 순서\n')
    md.write('1. 노드가 패킷을 라우팅할 때 그 노드의 T_est 를 다시 계산한다. '
             'T_est = 모든 목적지에 대한 (이웃 중 최소 Q) 의 평균. 계산 직후 기존 T_max 보다 크면 T_max 를 그 값으로 올린다. '
             'T_max 는 줄어들지 않고 윈도우가 바뀌어도 초기화되지 않는다.\n')
    md.write('2. 100 tick 이 되는 시점에만 통계를 찍는다. 이때 값을 새로 계산하지 않고 각 노드에 저장된 '
             'T_est 와 T_max 를 그대로 읽는다. 따라서 읽히는 값은 그 노드가 마지막으로 라우팅했을 때의 값이며, '
             '그 100 tick 동안의 변화는 기록되지 않는다(구간 평균이 아니라 시점 표본).\n')
    md.write('3. 노드마다 자기 T_est 를 자기 T_max 로 나눠 비율을 만든다. T_max 가 0 이하이거나 값이 유한하지 않은 노드는 제외한다.\n')
    md.write('4. 그 시점의 노드들을 평균해 그 윈도우의 값 세 개(R_x, T_est, T_max)를 만든다. 노드 축은 평균이다.\n')
    md.write('5. 2~4 를 반복하면 시드 하나당 (전체 tick ÷ 100) 개 길이의 시계열 세 개가 나온다.\n')
    md.write('6. 시드 10개에 대해 5 까지를 반복해 시계열을 10벌 얻는다.\n')
    md.write('7. 같은 시점끼리 시드 간 중앙값을 취해 대표 시계열을 만든다. 이 단계만 평균이 아니라 중앙값이다.\n')
    md.write('8. 대표 시계열을 시간 순 10구간으로 나눈다(20000 tick 실행이면 한 구간이 2000 tick, 그 안에 시점 표본 20개).\n')
    md.write('9. R_x 와 T_est 는 그 구간 20개 값의 평균을 한 칸으로 쓴다. '
             'T_max 만 평균 대신 구간의 마지막 값을 쓴다. 단조 증가하는 값이라 구간 마지막 값이 곧 구간 최댓값이고, '
             '평균을 쓰면 이미 고정된 뒤에도 계속 오르는 것처럼 보여 고정 시점이 흐려지기 때문이다.\n')
    md.write('10. 맨 끝 행은 R_x·T_est 는 시계열 전체 평균, T_max 는 최종값으로 적는다. '
             '에코와 변경률은 이 경로를 타지 않고, 시드마다 실행 전체 누적 조회수 ÷ 누적 결정수로 값 하나를 만든 뒤 '
             '그 10개의 중앙값을 쓴다. 그래서 이 둘만 10구간 평균과 정확히 일치하지 않는다(결정이 많은 구간이 더 크게 반영됨).\n')
    md.write('- 주의 : R_x 는 비율의 평균이고 T_est·T_max 열은 각각의 평균이므로, '
             'T_est 열을 T_max 열로 나눈 값은 R_x 열과 일치하지 않는다.\n')
    md.write('- 주의 : Q-routing 은 T_est 를 사용하지 않아 update 가 일어나지 않으므로 '
             'T_est=0, T_max=1(초기값), R_x=0 으로 고정 출력된다. 혼잡이 없다는 뜻이 아니다.\n\n')

    md.write('### 링크 사용량 / 패킷 전달 통계\n')
    md.write('- 링크 사용량 : 각 링크를 패킷이 지나간 횟수(양방향 합, 목적지 도착 마지막 홉 포함)를 시드 합산. '
             '그림은 그림 전체 전역 최소~최대를 7레벨로 나눠 레벨별 고정 색·굵기(같은 레벨이면 어느 패널이든 같은 색). '
             'md 표는 통과량 많은 순 정렬.\n')
    md.write('- 패킷 전달 통계 : 생성/전달/미전달은 시드 평균 패킷 수. 성공률 = 전달/생성(시드 평균), '
             '최고·최저는 시드 중 최대/최소 성공률(붕괴 확인용).\n\n')

    md.write('### ADT 그래프\n')
    md.write('- 매 100 tick 시점에서 10개 시드의 중앙값을 실선, 25~75퍼센타일(IQR)을 오차 막대로 그린다. '
             '평균 대신 중앙값·IQR 을 쓰는 이유는 한두 시드가 크게 튀어도 대표 곡선이 휘둘리지 않게 하기 위함.\n\n')


def write_diag_series_md(md, label, rx_med, echo_med, switch_med,
                         echo_scalar, switch_scalar, n_chunks=10,
                         t_est_med=None, t_max_med=None):
    """진단 시계열을 구간 표로 md 에 쓴다.
    t_est_med / t_max_med 를 넘기면 R_x 의 분자·분모 열이 함께 나온다.
    T_max 만 구간 평균이 아니라 구간 마지막 값(=구간 최댓값)을 쓰고, 끝 행도 최종값을 쓴다.
    에코·변경률의 끝 행은 총합 기준 스칼라라 10구간 평균과 다를 수 있다."""
    has_t = t_est_med is not None and t_max_med is not None

    md.write(f"#### 진단 시계열 (구간 평균, {label})\n\n")
    if has_t:
        md.write("| 구간 | R_x | T_est (틱) | T_max (틱, 구간 말) | "
                 "결정당 에코 이웃 수 (개/결정) | 경로 변경률 (%) |\n")
        md.write("|---|---:|---:|---:|---:|---:|\n")
    else:
        md.write("| 구간 | R_x | 결정당 에코 이웃 수 (개/결정) | 경로 변경률 (%) |\n")
        md.write("|---|---:|---:|---:|\n")

    rx_c = chunk_avg(rx_med, n_chunks)
    ec_c = chunk_avg(echo_med, n_chunks)
    sw_c = chunk_avg(switch_med, n_chunks)
    te_c = chunk_avg(t_est_med, n_chunks) if has_t else None
    tm_c = chunk_last(t_max_med, n_chunks) if has_t else None   # 단조 증가 → 구간 말 값

    for i in range(len(rx_c)):
        if has_t:
            md.write(f"| {i + 1} | {rx_c[i]:.3f} | {te_c[i]:.2f} | {tm_c[i]:.2f} | "
                     f"{ec_c[i]:.2f} | {sw_c[i] * 100:.1f}% |\n")
        else:
            md.write(f"| {i + 1} | {rx_c[i]:.3f} | {ec_c[i]:.2f} | {sw_c[i] * 100:.1f}% |\n")

    if has_t:
        # T_max 는 평균이 아니라 실행 종료 시점의 최종값
        md.write(f"| 전체 (T_max 는 최종값) | {series_overall(rx_med):.3f} | "
                 f"{series_overall(t_est_med):.2f} | {series_final(t_max_med):.2f} | "
                 f"{echo_scalar:.2f} | {switch_scalar * 100:.1f}% |\n\n")
    else:
        md.write(f"| 전체 평균 | {series_overall(rx_med):.3f} | "
                 f"{echo_scalar:.2f} | {switch_scalar * 100:.1f}% |\n\n")
