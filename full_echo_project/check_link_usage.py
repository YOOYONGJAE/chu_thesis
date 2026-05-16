from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A
from simulator import Simulator

BASE_PARAMS = {'eta': 0.9, 'k': 0.556, 'L': 3}
CUT_TICK = 7000
c= 0.5
params = {**BASE_PARAMS, 'c': c, 'memory_cut_tick': CUT_TICK}
topo = {'num_nodes': G_N, 'adjacency': G_A}

# sim = Simulator('aqlrerm', params, seed=800, topology=topo)
sim = Simulator('aqrerm', params, seed=800, topology=topo)
sim.run(lam=3, total_ticks=7000, stat_interval=100)

top10 = sorted(sim.link_usage.items(), key=lambda x: -x[1])[:10]
print('Top 10 링크 사용량:')
for link, count in top10:
    print(f'  {link}: {count}')
