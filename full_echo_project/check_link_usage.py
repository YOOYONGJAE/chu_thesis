from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A
from simulator import Simulator

# =====================================================================
# 공통 설정 — main_link_cut.py 와 동일
# =====================================================================
BASE_PARAMS   = {'eta': 0.9, 'k': 0.556, 'L': 3}
TOTAL_TICKS   = 13000
STAT_INTERVAL = 100
CUT_TICK      = 7000
SEED          = 800
c             = 0.5
topo          = {'num_nodes': G_N, 'adjacency': G_A}

MD_PATH       = 'result_link_usage.md'

# =====================================================================
# 실험 매트릭스 — main_link_cut.py 활성 변형과 동일
# =====================================================================
ALGORITHMS = [
    'aqrerm',
    'aqrerm_l0',         # AQRERM + 7000 이후 L=0
    'pfe_c_ade',         # L=0 전환 없음 (L=3 유지)
    'pfe_c_ade_l0',      # L=0 + c=0.5
]

ALGO_LABELS = {
    'aqrerm':       'AQRERM',
    'aqrerm_l0':    'AQRERM_L=0',
    'pfe_c_ade':    'PFE_c=0.5',
    'pfe_c_ade_l0': 'PFE_c=0.5_L=0',
}

# 절단 시나리오 — main_link_cut.py 와 동일 (두 다리)
CUT_SCENARIOS = [
    {'name': '(14,15) [top bridge]',  'link': (14, 15)},
    {'name': '(2,3) [bottom bridge]', 'link': (2, 3)},
]

# 부하 — main_link_cut.py 와 동일
LAMBDAS = [1.5, 2]


def top10(usage_dict):
    """usage_dict 에서 사용량 상위 10 개 (link, count) 튜플 리스트 반환."""
    return sorted(usage_dict.items(), key=lambda x: -x[1])[:10]


def print_table_stdout(label, items):
    print(f'\n  {label}:')
    for link, count in items:
        print(f'    {link}: {count}')


def write_table_md(md, header, items):
    md.write(f'#### {header}\n\n')
    md.write('| 링크 | 사용량 |\n|---|---|\n')
    for link, count in items:
        md.write(f'| {link} | {count} |\n')
    md.write('\n')


# =====================================================================
# 실행 — scenario × lambda × algorithm 매트릭스
# =====================================================================
with open(MD_PATH, 'w', encoding='utf-8') as md:
    # 공통 설정 헤더
    md.write('# 링크 사용량 분석 (Top 10) — main_link_cut.py 동일 설정\n\n')
    md.write('## 공통 설정\n\n')
    md.write(f'- TOTAL_TICKS: {TOTAL_TICKS}\n')
    md.write(f'- STAT_INTERVAL: {STAT_INTERVAL}\n')
    md.write(f'- CUT_TICK: {CUT_TICK}\n')
    md.write(f'- SEED: {SEED}\n')
    md.write(f'- c: {c}\n')
    md.write(f'- BASE_PARAMS: {BASE_PARAMS}\n')
    md.write(f'- ALGORITHMS: {ALGORITHMS}\n')
    md.write(f'- LAMBDAS: {LAMBDAS}\n')
    md.write(f'- CUT_SCENARIOS: {[s["name"] for s in CUT_SCENARIOS]}\n\n')
    md.write('---\n\n')

    for scenario in CUT_SCENARIOS:
        md.write(f'## Cut {scenario["name"]}\n\n')
        for lam in LAMBDAS:
            md.write(f'### λ={lam}\n\n')
            print(f'\n========== Cut {scenario["name"]} | λ={lam} ==========')

            for algo in ALGORITHMS:
                label = ALGO_LABELS[algo]
                print(f'\n  --- {label} ---')

                params = {**BASE_PARAMS, 'c': c, 'memory_cut_tick': CUT_TICK}
                link_cuts = [(CUT_TICK, *scenario['link'])]

                sim = Simulator(algo, params, seed=SEED, topology=topo)
                sim.run(lam=lam, total_ticks=TOTAL_TICKS,
                        stat_interval=STAT_INTERVAL, link_cuts=link_cuts)

                gen, dlv, und = sim.total_generated, sim.total_delivered, sim.undelivered_count
                rate = (dlv / gen * 100) if gen > 0 else 0.0
                stats_line = (f'generated={gen}  delivered={dlv}  '
                              f'undelivered={und}  delivery_rate={rate:.1f}%')
                print(f'    {stats_line}')

                # MD 섹션 시작
                md.write(f'#### {label}\n\n')
                md.write(f'- algorithm: `{algo}`\n')
                md.write(f'- {stats_line}\n\n')

                # 절단 전/후 분리 출력
                pre  = top10(sim.link_usage_pre_cut)
                post = top10(sim.link_usage_post_cut)
                print_table_stdout(f'절단 전 (tick 0~{CUT_TICK-1})', pre)
                print_table_stdout(f'절단 후 (tick {CUT_TICK}~{TOTAL_TICKS-1})', post)
                write_table_md(md, f'절단 전 (tick 0~{CUT_TICK-1})', pre)
                write_table_md(md, f'절단 후 (tick {CUT_TICK}~{TOTAL_TICKS-1})', post)

            md.write('---\n\n')

print(f'\n모든 실험 완료. 결과 로그: {MD_PATH}')
