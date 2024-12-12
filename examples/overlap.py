import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import randium as rd

L = 64
M = 512
beta = 1.32
lat = rd.Lattice(L, M, D=2, beta=beta)
print(f'{lat.N=} {lat.M=} {lat.N_m=} {lat.beta=}')

## Equilibrate system
steps_eq = lat.N*1024*2
lat.simulation_monte_carlo_global(steps_eq)

steps = steps_eq
print_stride = steps//512
times = [(n*print_stride/lat.N) for n in range(steps//print_stride)]
enr, acc = lat.simulation_monte_carlo_local(steps, print_stride)
print(f'Acceptance ratio: {acc:.6f}')

plt.figure()
plt.title(f'{lat.L=} {lat.N=} {lat.M=} {lat.N_m=} {lat.beta=}')
plt.plot(times, enr)
plt.xlabel('Time')
plt.ylabel('Total Energy, $U$')
plt.show()

# Compute overlap data
timeblocks = 8
times_in_timeblocks = np.logspace(1, 6, num=64, base=10)
overlap_data = []
for b in range(timeblocks):
    lat.simulation_monte_carlo_global(steps_eq)
    t_now = 0.0
    lat_ref = lat.copy()
    this_overlaps = []
    print(f'{b:<6}', end='', flush=True)
    for t_next in times_in_timeblocks:
        steps = int((t_next-t_now)*lat.N)
        enr, acc = lat.simulation_monte_carlo_local(steps)
        t_now += steps/lat.N
        Q, Q_arr = lat.overlap(lat_ref)
        this_overlaps.append(Q)
        print('.', end='', flush=True)
    print('')
    overlap_data.append(this_overlaps)

# Plot mean of overlap data:
overlap_data = np.array(overlap_data)
overlap_mean = np.mean(overlap_data, axis=0)
plt.figure()
plt.plot(times_in_timeblocks, overlap_mean, 'o')
plt.plot([min(times_in_timeblocks), max(times_in_timeblocks)], [0,0], 'k--')
plt.xscale('log')
plt.ylim(-0.1, 1)
plt.show()

# Save to CSV file
pd.DataFrame({
    'times': times_in_timeblocks,
    'overlap': overlap_mean,
}).to_csv(f'data/overlap_{beta:.2f}_new.csv', index=False)
