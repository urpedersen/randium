import time
from time import perf_counter
import toml
import numpy as np
import matplotlib.pyplot as plt

import randium as rd



for beta in 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8:
    print(f'beta = {beta}, {time.ctime() =}')
    #beta = 0.1
    L = 96
    M = 96*96
    D = 2

    # Pre-investigations
    tic = perf_counter()
    lat = rd.Lattice(L, M, D, beta=beta)
    print(f'Time to allocate: {perf_counter()-tic}')
    print(perf_counter()-tic)
    eq_steps_per_blocks = 1024*8
    eq_blocks = 512
    eq_steps = eq_steps_per_blocks*eq_blocks
    tic = perf_counter()
    energies = []
    for _ in range(eq_blocks):
        lat.simulation_monte_carlo_local(eq_steps_per_blocks)
        energies.append(lat.get_total_energy()/lat.N)
    print(f'Time to run equbriliation: {perf_counter()-tic}')

    plt.figure()
    plt.title(f'{lat.L=} {lat.N=} {lat.M=} {lat.N_m=} {lat.beta=}')
    plt.plot(energies)
    plt.plot([len(energies)], [-2*beta], 'o')
    plt.show()

    # Compute overlap data
    timeblocks = 16
    #times_in_timeblocks = np.logspace(-0.5, 5, num=128, base=10)
    inner_steps = [int(1.4 ** x) for x in range(1, 46)]
    print(f'{sum(inner_steps) = }')
    energy_data = []
    overlap_data = []
    times_in_timeblocks = np.cumsum(inner_steps)/lat.N
    for b in range(timeblocks):
        lat = rd.Lattice(L, M, D, beta=beta)
        lat.simulation_monte_carlo_global(eq_steps)
        t_now = 0.0
        lat_ref = lat.copy()
        this_overlaps = []
        this_energies = []
        print(f'{b:<6}', end='', flush=True)
        for idx, steps in enumerate(inner_steps):
            enr, acc = lat.simulation_monte_carlo_local(steps)
            this_energies.append(enr)
            t_now += steps/lat.N
            Q, Q_arr = lat.overlap(lat_ref)
            this_overlaps.append(Q)
            if idx%4==0:
                print('.', end='', flush=True)
        print('')
        energy_data.append(this_energies)
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

    energy_data = np.array(energy_data)
    toml.dump(dict(
        rows=lat.L,
        cols=lat.L,
        N = lat.N,
        M = lat.M,
        N_m = lat.N_m,
        #N_M = lat.N_M,
        #**rdm.meta_info(),
        beta=beta,
        times=[float(x) for x in times_in_timeblocks],
        overlaps=[float(x) for x in overlap_mean],
        energy = float(np.mean(energy_data/lat.N)),
        energy_std = float(np.std(energy_data/lat.N))
    ), open(f'data/local_{lat.L}x{lat.L}M{lat.M}beta{beta:.4f}_cpu.toml', 'w'))


