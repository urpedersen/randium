import time

import numpy as np
import matplotlib.pyplot as plt

import randium_2d_gpu as rd2

beta_old = 1.40
beta_new = 1.30
plt.figure()
plt.title(f'Ageing from {beta_old} to {beta_new}')
energy_data = []
num_frames = 16
tic = time.perf_counter()
for frame_to_load in range(num_frames):
    print(f'  Frame {frame_to_load}   {time.ctime()}')
    fname=f'local_192x192M36864beta{beta_old:.4f}'
    lattices = np.load(f'data/{fname}.npy')
    print(f'{lattices.shape = }')
    lattice = lattices[frame_to_load]

    rdm = rd2.Randium_2d_gpu(threads_per_block=(4, 4), blocks=(12, 12), tiles=(4, 4), num_of_each_type=1)
    rdm.set_lattice(lattice)
    rdm.run(beta=1.6, steps=0)

    num_points = 28  # 34
    inner_steps = [int(1.4 ** x) for x in range(1, num_points)]
    energies=[]
    sim_time = 0.0
    sim_times=[]
    for idx, delta_steps in enumerate(inner_steps):
        rdm.run(beta=beta_new, steps=delta_steps)
        sim_time += delta_steps * 4
        sim_times.append(sim_time)
        energy = rdm.energy()
        print(f'{idx}/{num_points}: {sim_time = :.4} {delta_steps = }, {energy = :.4} {-energy/(2*beta_new) = :.4}')
        energies.append(energy)
    plt.plot(sim_times, energies, 'o')
    energy_data.append(energies)
    toc = time.perf_counter()
    print(f'Time elapsed: {toc - tic:0.1f} seconds')
    runtime_estimate = (toc - tic) * num_frames / (frame_to_load+1) / 60
    print(f'Percent done: {frame_to_load/num_frames*100:0.1f}%')
    print(f'Estimated total runtime: {runtime_estimate:0.1f} minutes')
    print(f'                       = {runtime_estimate / 60:0.1f} hours')
    time_left = runtime_estimate - (toc - tic) / 60
    print(f'Estimated time left: {time_left:0.1f} minutes')
    print(f'                   = {time_left / 60:0.1f} hours')
plt.xlabel(r'Time, $t$ [swap attempts per particle]')
plt.ylabel(r'Energy, $u=E/N$')
plt.xscale('log')
plt.show()

fname=f'data/ageing_{beta_old:.4f}_to_{beta_new:.4f}.npz'
np.savez(fname, times=sim_times, energy_data=energy_data)
