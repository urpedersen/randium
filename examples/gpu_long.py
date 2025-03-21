from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import randium_2d_gpu as rd2

import toml

beta = 1.0
rdm = rd2.Randium_2d_gpu(threads_per_block=(8, 8), blocks=(16, 16), tiles=(6, 6), num_of_each_type=16)
print(rdm)

lat_ref = rdm.lattice.copy()
wc = rdm.run()  # Compile
print(f'Compile (local GPU): {wc} ms')
wc = rdm.run(beta=beta, steps=128)
print(f'Equilibrate (local): {wc} ms')
print(f'MC attempts per second (GPU): {rdm.get_benchmark()['mc_attempts_per_sec']:0.2e}')
tic = perf_counter()
rdm.run_global(beta=beta)  # Compile
toc = perf_counter()
print(f'Compile (global CPU): {(toc - tic) * 1000} ms')
steps = 8
tic = perf_counter()
rdm.run_global(beta=beta, steps=steps)
toc = perf_counter()
print(f'Equilibrate (global): {(toc - tic) * 1000} ms')
mc_attempts_per_sec = rdm.N * steps / (toc - tic)
print(f'MC attempts per second (global CPU): {mc_attempts_per_sec:0.2e}')
print(f'Speed-up: {rdm.get_benchmark()['mc_attempts_per_sec'] / mc_attempts_per_sec}')
wc = rdm.run(beta=beta, steps=512)
print(f'Equilibrate (local, extra): {wc} ms')


for _ in range(4):
    tic = perf_counter()
    u = rdm.energy()
    toc = perf_counter()
    print(f'Energy: {u} (in {(toc - tic) * 1000:0.2f} ms)')

time_blocks = 4
block_energies = []
block_times = []
inner_steps = [int(1.3 ** x) for x in range(1, 20)]
print(f'{inner_steps = }')
overlap_table = []
for _ in range(time_blocks):
    lat_ref = rdm.lattice.copy()
    time = 0
    times = []
    this_overlaps = []
    for delta_steps in inner_steps:
        rdm.run(beta=beta, steps=delta_steps)
        time += delta_steps * 4
        times.append(time)
        overlap = float(np.sum((rdm.lattice - lat_ref) == 0) / rdm.N)
        this_overlaps.append(overlap)
        energy = rdm.energy()
        print(f'{time = }, {overlap = :.3} {energy = :.3}')
        block_energies.append(energy)
        block_times.append(time)
    overlap_table.append(this_overlaps)

plt.figure()
plt.plot(block_times, block_energies, '-')
plt.xlabel(r'Time, $t$ [swap attempts per particle]')
plt.ylabel(r'Energy, $E/N$')
plt.show()

times = np.array(times)
overlaps = np.array(overlap_table).mean(axis=0)
overlaps = (overlaps - 1 / rdm.M) / (1 - 1 / rdm.M)  # Normalize
print(f'{overlaps = }  ({1/rdm.M = })')
print(f'{times = }')
plt.figure()
plt.plot(times, overlaps, 'o-')
plt.title(f'{rdm.cols}×{rdm.cols}, M={rdm.M}, N_m={rdm.N_m}, {beta = }')
plt.xlabel(r'Time, $t$ [swap attempts per particle]')
plt.ylabel(r'Overlap, $Q(t)$')
plt.ylim(0.0, 1.0)
plt.xscale('log')
plt.xlim(1e-1, 1e8)
plt.show()

# Save overlap to data folder
toml.dump(dict(
    **rdm.meta_info(),
    times=[int(x) for x in times],
    overlaps=[float(x) for x in overlaps],
), open(f'data/local_{rdm.rows}x{rdm.cols}M{rdm.M}beta{beta:.4f}.toml', 'w'))
