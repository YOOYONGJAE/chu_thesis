# =============================================================================
# [요약] 야간 일괄 실행 러너 — 5개 비교 스크립트 × (LOW / HIGH) 두 부하 세트
# - 각 조합을 별도 프로세스로 돌려 한 조합이 실패해도 나머지가 이어서 실행된다.
# - 원본 main_*.py 는 수정하지 않는다. import 후 모듈 전역만 덮어써서 돌린다.
#     EXPERIMENTS  : 아래 LOADS 의 부하 세트로 교체
#     RESULT_TAG   : 'LOW' / 'HIGH'
#     TOTAL_TICKS  : 있으면 20000 으로 통일
#     출력 경로     : 기존 _BASENAME 의 접두어를 살려 (토폴로지,부하,틱수) 로 재조립
# - main_c_effect_compare 는 기본이 PQ 15노드라, 격자로 바꿔 (GRID,...) 로 나오게 한다.
# - 이미 결과 md 가 있는 조합은 건너뛴다 (--force 로 무시). 중간에 죽어도 이어서 돌릴 수 있다.
#
# 사용법:
#   python run_overnight.py                 # 전부 실행 (기존 결과는 건너뜀)
#   python run_overnight.py --force         # 기존 결과가 있어도 다시 실행
#   python run_overnight.py --only lineage  # 이름에 lineage 가 들어간 것만
# =============================================================================
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
TOTAL_TICKS = 20000

# 부하 세트 — 저 / 중저 / 중고 / 고 네 단계를 한 세트로 묶어 모든 실험에 동일 적용.
# 고부하 상한이 3.8 인 이유 : AQRERM 은 발산하고 AQPRICE 는 정착하는 경계 지점.
LOADS = {
    '4LVL': [
        {'lam': 2.5, 'total_ticks': TOTAL_TICKS},   # 저부하
        {'lam': 3.0, 'total_ticks': TOTAL_TICKS},   # 중저부하
        {'lam': 3.5, 'total_ticks': TOTAL_TICKS},   # 중고부하
        {'lam': 3.8, 'total_ticks': TOTAL_TICKS},   # 고부하
    ],
}

MODULES = [
    'main_lineage',
    'main_aqrerm_vs_aqprice',
    'main_aqprice_route_memory_L_changed',
    'main_aqprice_inline_compare',
    'main_c_effect_compare',
]

LOG_PATH = BASE / 'run_overnight.log'


# -------------------------------------------------------------------------
# 자식 프로세스: 조합 하나만 실행
# -------------------------------------------------------------------------
def force_grid(mod, name):
    """어느 스크립트든 6x6 격자로 맞춘다.
    원본 파일의 토폴로지 주석 토글은 건드리지 않고, import 된 모듈의 전역만 갈아끼운다.
    스크립트마다 전역 이름이 GRID_* / PQ_* 로 달라서 있는 쪽만 덮어쓴다.
    (c_effect 는 PQ, aqrerm_vs_aqprice 는 NSFNET 으로 두고 쓰던 이력이 있어 필요한 처리)"""
    import topology_grid as tg
    if hasattr(mod, 'GRID_NUM_NODES'):
        mod.GRID_NUM_NODES = tg.NUM_NODES
        mod.GRID_ADJACENCY = tg.ADJACENCY
        mod.TOPOLOGY_GRID = {'num_nodes': tg.NUM_NODES, 'adjacency': tg.ADJACENCY}
    if hasattr(mod, 'PQ_NUM_NODES'):
        mod.PQ_NUM_NODES = tg.NUM_NODES
        mod.PQ_ADJACENCY = tg.ADJACENCY
        mod.TOPOLOGY_PQ = {'num_nodes': tg.NUM_NODES, 'adjacency': tg.ADJACENCY}
    if hasattr(mod, 'TOPO_NAME'):
        mod.TOPO_NAME = '6x6 Grid'


def result_paths(mod, load):
    """기존 _BASENAME 의 접두어(result_xxx)를 살려 새 파일명을 만든다."""
    prefix = mod._BASENAME.split('(')[0]          # 'result_lineage' 등
    tag = getattr(mod, 'TOPO_TAG', 'GRID')
    base = f'{prefix}({tag},{load},{TOTAL_TICKS})'
    return base, f'{base}.md', f'{base}.png', f'{base}_linkuse.png'


def run_one(name, load):
    import importlib
    sys.path.insert(0, str(BASE))
    mod = importlib.import_module(name)

    # 1) 토폴로지 (c_effect 만 해당) → TOPO_TAG 재계산의 근거가 되므로 먼저
    force_grid(mod, name)
    n_nodes = getattr(mod, 'GRID_NUM_NODES', None) or getattr(mod, 'PQ_NUM_NODES', None)
    mod.TOPO_TAG = {36: 'GRID', 14: 'NSFNET', 15: 'PQ'}.get(n_nodes, f'{n_nodes}N')

    # 2) 부하 세트와 실행 길이
    mod.EXPERIMENTS = [dict(e) for e in LOADS[load]]
    mod.RESULT_TAG = load
    if hasattr(mod, 'TOTAL_TICKS'):
        mod.TOTAL_TICKS = TOTAL_TICKS

    # 3) 출력 경로 재조립
    base, md, png, link_png = result_paths(mod, load)
    mod._BASENAME = base
    mod.MD_PATH = md
    mod.PNG_PATH = png
    mod.LINK_PNG_PATH = link_png

    print(f'[run_one] {name} / {load} → {md}', flush=True)
    mod.run_all()


# -------------------------------------------------------------------------
# 부모 프로세스: 조합을 순서대로 자식에게 던진다
# -------------------------------------------------------------------------
def expected_md(name, load):
    """건너뛰기 판정용 — 모듈을 import 하지 않고 파일명만 추정한다."""
    prefix = name.replace('main_', 'result_')
    return BASE / f'{prefix}(GRID,{load},{TOTAL_TICKS}).md'


def log(msg):
    line = f'[{datetime.now():%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--one', nargs=2, metavar=('MODULE', 'LOAD'),
                    help='자식 프로세스 전용 — 조합 하나만 실행')
    ap.add_argument('--force', action='store_true', help='기존 결과가 있어도 다시 실행')
    ap.add_argument('--only', default=None, help='모듈 이름에 이 문자열이 들어간 것만')
    args = ap.parse_args()

    if args.one:
        run_one(args.one[0], args.one[1])
        return

    combos = [(m, l) for m in MODULES for l in LOADS
              if args.only is None or args.only in m]

    n_lam = len(next(iter(LOADS.values())))
    log(f'=== 야간 실행 시작 : {len(combos)} 조합, '
        f'각 {TOTAL_TICKS} tick × {n_lam} 부하 × 10 시드 ===')
    started = time.time()
    done, skipped, failed = [], [], []

    for i, (name, load) in enumerate(combos, 1):
        md = expected_md(name, load)
        if md.exists() and not args.force:
            log(f'({i}/{len(combos)}) 건너뜀 — 이미 있음 : {md.name}')
            skipped.append((name, load))
            continue

        log(f'({i}/{len(combos)}) 시작 : {name} / {load}')
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--one', name, load],
            cwd=str(BASE),
        )
        mins = (time.time() - t0) / 60
        if proc.returncode == 0:
            log(f'({i}/{len(combos)}) 완료 : {name} / {load}  ({mins:.1f}분)')
            done.append((name, load))
        else:
            log(f'({i}/{len(combos)}) 실패 : {name} / {load}  '
                f'(exit {proc.returncode}, {mins:.1f}분) — 다음으로 넘어갑니다')
            failed.append((name, load))

    total = (time.time() - started) / 60
    log(f'=== 종료 : 완료 {len(done)} / 건너뜀 {len(skipped)} / 실패 {len(failed)} '
        f'/ 총 {total:.1f}분 ===')
    for name, load in failed:
        log(f'  실패 목록 : {name} / {load}')


if __name__ == '__main__':
    main()
