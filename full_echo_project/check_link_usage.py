from topology_grid import NUM_NODES as G_N, ADJACENCY as G_A
from simulator import Simulator

PARAMS = {'eta': 0.9, 'k': 0.556, 'L': 3}
topo = {'num_nodes': G_N, 'adjacency': G_A}

sim = Simulator('aqrerm', PARAMS, seed=800, topology=topo)
sim.run(lam=3, total_ticks=7000, stat_interval=100)

top10 = sorted(sim.link_usage.items(), key=lambda x: -x[1])[:10]
print('Top 10 링크 사용량:')
for link, count in top10:
    print(f'  {link}: {count}')
