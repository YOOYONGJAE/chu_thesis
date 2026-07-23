# =============================================================================
# [요약] AQPRICE 파라미터 자동 튜닝 — c / pfe_gr / pfe_b_max 를 BO 로 탐색
# - 6x6 grid, λ=3.6, 10 시드 평가, gp_minimize 40회 (random 10 + BO 30)
# - 평가 점수: score_seed = mean(ADT 전반) + mean(ADT 후반) + 0.5·std(ADT 후반),
#   score = 시드 간 median (작을수록 좋음)
# - 현재 BASE_PARAMS 의 c=0.22 / pfe_b_max=0.5 가 이 탐색의 best 조합 출처
# - 필요 라이브러리: scikit-optimize (pip install scikit-optimize)
# - 산출물: 수렴 곡선 + 3D 산점도 dashboard (png) / 평가 이력 md
# =============================================================================

import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트 (그래프 한글 깨짐 방지)
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — projection='3d' 위해 등록
from simulator import Simulator
from topology_grid import NUM_NODES as GRID_NUM_NODES, ADJACENCY as GRID_ADJACENCY

# BO 라이브러리
from skopt import gp_minimize
from skopt.space import Real


# -------------------------------------------------------------------------
# 토폴로지 / 알고리즘 / 부하
# -------------------------------------------------------------------------
TOPOLOGY_GRID = {'num_nodes': GRID_NUM_NODES, 'adjacency': GRID_ADJACENCY}
ALGORITHM = 'aqprice'
LAM = 3.6

# -------------------------------------------------------------------------
# 시뮬레이션 설정
# -------------------------------------------------------------------------
TOTAL_TICKS = 10000
STAT_INTERVAL = 100
N_WINDOWS = TOTAL_TICKS // STAT_INTERVAL                # 100 윈도우
HALF = N_WINDOWS // 2                                    # 50 (수렴 후 절반 시작점)

# -------------------------------------------------------------------------
# 고정 파라미터 (sweep 안 하는 것)
# -------------------------------------------------------------------------
ETA = 0.9
K   = 0.5
L   = 3
PFE_C_FIXED = 0.1   # 풀에코 가격 override 고정. sweep 대상이 아님.

# -------------------------------------------------------------------------
# 평가 시드 (조합당 평가 시 사용)
# -------------------------------------------------------------------------
EVAL_SEEDS = list(range(100, 1001, 100))   # [100, 200, ..., 1000], 10 개

# -------------------------------------------------------------------------
# BO 설정
# -------------------------------------------------------------------------
N_CALLS = 40             # 총 BO 평가 횟수
N_INITIAL = 10           # 초기 무작위 샘플 수
BO_SEED = 42             # BO 내부 무작위성 재현용

# 탐색 공간 (연속)
SPACE = [
    Real(0.1, 0.5, name='c'),
    Real(0.1, 0.5, name='pfe_gr'),
    Real(0.2, 0.5, name='pfe_b_max'),
]

# -------------------------------------------------------------------------
# 출력 파일
# -------------------------------------------------------------------------
PNG_PATH = 'result_compare_AQPRICE_p_params.png'
MD_PATH  = 'result_compare_AQPRICE_p_params.md'


# -------------------------------------------------------------------------
# 진행률 출력용 모듈 전역 상태
# -------------------------------------------------------------------------
_eval_count = 0
_start_time = None


def _fmt_time(seconds):
    """초 단위 시간을 상황에 맞춰 s / min / h 단위로 포맷."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


# -------------------------------------------------------------------------
# 단일 (시드, 파라미터) 시뮬레이션 → ADT 시계열 반환
# -------------------------------------------------------------------------
def run_one_seed(c, pfe_gr, pfe_b_max, seed):
    random.seed(seed)
    np.random.seed(seed)
    params = {
        'eta': ETA,
        'k': K,
        'L': L,
        'c': c,
        'pfe_gr': pfe_gr,
        'pfe_b_max': pfe_b_max,
        'pfe_c': PFE_C_FIXED,
    }
    sim = Simulator(algorithm=ALGORITHM, params=params, seed=seed, topology=TOPOLOGY_GRID)
    adt = sim.run(lam=LAM, total_ticks=TOTAL_TICKS, stat_interval=STAT_INTERVAL)
    return np.asarray(adt, dtype=float)


# -------------------------------------------------------------------------
# 단일 시드의 점수 계산
#   score_seed = mean(0~half) + mean(half:) + 0.5 * std(half:)
# -------------------------------------------------------------------------
def seed_score(adt):
    front = adt[:HALF]
    back  = adt[HALF:]
    return float(np.mean(front) + np.mean(back) + 0.5 * np.std(back))


# -------------------------------------------------------------------------
# 한 조합당 평가 점수 (BO objective)
#   10 시드 점수의 median
# -------------------------------------------------------------------------
def evaluate(params):
    global _eval_count, _start_time
    if _start_time is None:
        _start_time = time.time()
    _eval_count += 1

    # 진행률 헤더 — 현재 평가 번호 / 총 평가 / 백분율 / 경과 / ETA
    pct = _eval_count / N_CALLS * 100
    elapsed = time.time() - _start_time
    # ETA 추정: (현재까지 평균 시간) × (남은 평가 횟수). 첫 평가는 ETA 미정.
    if _eval_count > 1:
        avg_per_eval = elapsed / (_eval_count - 1)
        eta = avg_per_eval * (N_CALLS - _eval_count + 1)
        eta_str = _fmt_time(eta)
    else:
        eta_str = "?"
    print(f"\n[{_eval_count}/{N_CALLS}, {pct:5.1f}%]  "
          f"elapsed={_fmt_time(elapsed)}  ETA={eta_str}")

    c, pfe_gr, pfe_b_max = params
    print(f"  params : c={c:.4f}  gr={pfe_gr:.4f}  b_max={pfe_b_max:.4f}")

    seed_scores = []
    n_seeds = len(EVAL_SEEDS)
    for i, seed in enumerate(EVAL_SEEDS, start=1):
        adt = run_one_seed(c, pfe_gr, pfe_b_max, seed)
        seed_scores.append(seed_score(adt))
        # 시드 진행 표시 (같은 줄 덮어쓰기)
        seed_pct = i / n_seeds * 100
        print(f"\r  seeds  : {i}/{n_seeds} ({seed_pct:3.0f}%)", end='', flush=True)
    print()  # 시드 진행 줄 닫기

    score = float(np.median(seed_scores))
    print(f"  result : score={score:.4f}  "
          f"(seed min={min(seed_scores):.2f}, max={max(seed_scores):.2f})")
    return score


# -------------------------------------------------------------------------
# BO 실행 + 시각화 + 로그
# -------------------------------------------------------------------------
def main():
    print(f"=== BO 시작 ===")
    print(f"알고리즘 : {ALGORITHM}")
    print(f"부하 λ   : {LAM}")
    print(f"total_ticks : {TOTAL_TICKS} (N_WINDOWS={N_WINDOWS}, HALF={HALF})")
    print(f"EVAL_SEEDS : {EVAL_SEEDS}")
    print(f"BO_SEED : {BO_SEED}")
    print(f"N_CALLS={N_CALLS}, N_INITIAL={N_INITIAL}")
    print(f"고정 : eta={ETA}, k={K}, L={L}, pfe_c={PFE_C_FIXED}")
    print()

    # ---- BO 호출 ----
    result = gp_minimize(
        evaluate,
        SPACE,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        random_state=BO_SEED,
        verbose=False,
    )

    # ---- 결과 추출 ----
    best_x = result.x                        # [c, gr, b_max]
    best_score = float(result.fun)
    x_iters = np.array(result.x_iters)       # shape (N_CALLS, 3)
    func_vals = np.array(result.func_vals)   # shape (N_CALLS,)
    convergence = np.minimum.accumulate(func_vals)
    best_idx = int(np.argmin(func_vals))

    print(f"\n=== BO 완료 ===")
    print(f"best 조합 : c={best_x[0]:.4f}, gr={best_x[1]:.4f}, b_max={best_x[2]:.4f}")
    print(f"best score : {best_score:.4f}")
    print(f"best 발견 시점 : 평가 {best_idx + 1} / {N_CALLS}")

    # ---- 시각화 : 1×2 dashboard ----
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        f"PFE_c_PreEcho_Tick BO (λ={LAM}, n_evals={N_CALLS}, n_seeds={len(EVAL_SEEDS)}, "
        f"pfe_c fixed={PFE_C_FIXED}, BO_seed={BO_SEED})"
    )

    # 왼쪽 : 수렴 곡선
    ax1 = fig.add_subplot(1, 2, 1)
    eval_x = np.arange(1, N_CALLS + 1)
    ax1.plot(eval_x, func_vals, 'o-', color='#999999', alpha=0.5,
             markersize=5, linewidth=0.8, label='each eval')
    ax1.plot(eval_x, convergence, '-', color='#0072B2', linewidth=2.2,
             label='best so far')
    ax1.axvline(N_INITIAL + 0.5, color='red', linestyle='--', linewidth=1,
                alpha=0.6, label=f'BO 시작 (after {N_INITIAL} random)')
    ax1.set_xlabel('Evaluation #')
    ax1.set_ylabel('Score (lower is better)')
    ax1.set_title('Convergence Curve')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 오른쪽 : 3D 산점도
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc = ax2.scatter(
        x_iters[:, 0], x_iters[:, 1], x_iters[:, 2],
        c=func_vals, cmap='viridis', s=60, alpha=0.85,
        edgecolors='k', linewidth=0.3,
    )
    # best 점 강조 (빨간 별)
    ax2.scatter(
        best_x[0], best_x[1], best_x[2],
        c='red', marker='*', s=400, edgecolors='black', linewidth=1.2,
        label=f'BEST\nc={best_x[0]:.3f}\ngr={best_x[1]:.3f}\nb_max={best_x[2]:.3f}\nscore={best_score:.2f}',
        zorder=10,
    )
    ax2.set_xlabel('c')
    ax2.set_ylabel('pfe_gr')
    ax2.set_zlabel('pfe_b_max')
    ax2.set_title('Parameter Space Exploration (color = score)')
    ax2.view_init(elev=22, azim=45)
    cbar = fig.colorbar(sc, ax=ax2, shrink=0.6, pad=0.08)
    cbar.set_label('Score (lower better)')
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()
    print(f"\n결과 PNG 저장 : {PNG_PATH}")

    # ---- MD 로그 ----
    with open(MD_PATH, 'w', encoding='utf-8') as md:
        md.write('# PFE_c_PreEcho_Tick BO 결과\n\n')
        md.write('## 설정\n\n')
        md.write(f'- algorithm : {ALGORITHM}\n')
        md.write(f'- λ : {LAM}\n')
        md.write(f'- total_ticks : {TOTAL_TICKS}  (N_WINDOWS={N_WINDOWS}, HALF={HALF})\n')
        md.write(f'- EVAL_SEEDS : {EVAL_SEEDS}\n')
        md.write(f'- BO_SEED : {BO_SEED}\n')
        md.write(f'- N_CALLS : {N_CALLS}  (initial random {N_INITIAL} + BO {N_CALLS - N_INITIAL})\n')
        md.write(f'- 고정 : eta={ETA}, k={K}, L={L}, pfe_c={PFE_C_FIXED}\n')
        md.write(f'- 탐색 범위 : c∈[0.1, 0.5], pfe_gr∈[0.1, 0.5], pfe_b_max∈[0.2, 0.5]\n')
        md.write('\n')

        md.write('## Best 조합\n\n')
        md.write(f'- **c = {best_x[0]:.4f}**\n')
        md.write(f'- **pfe_gr = {best_x[1]:.4f}**\n')
        md.write(f'- **pfe_b_max = {best_x[2]:.4f}**\n')
        md.write(f'- **score = {best_score:.4f}**\n')
        md.write(f'- 발견 시점 : 평가 {best_idx + 1} / {N_CALLS}\n')
        md.write('\n')

        md.write('## 평가 이력 (40 회)\n\n')
        md.write('| eval # | c | pfe_gr | pfe_b_max | score | best so far |\n')
        md.write('|---:|---:|---:|---:|---:|---:|\n')
        for i, (xs, val, conv) in enumerate(zip(x_iters, func_vals, convergence), start=1):
            mark = ' ★' if i == best_idx + 1 else ''
            md.write(f'| {i}{mark} | {xs[0]:.4f} | {xs[1]:.4f} | {xs[2]:.4f} | '
                     f'{val:.4f} | {conv:.4f} |\n')

    print(f"결과 MD 저장 : {MD_PATH}")


if __name__ == '__main__':
    main()
