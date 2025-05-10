from itertools import product
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import randium_2d_gpu as rd2

import toml

from numba import cuda

device = cuda.get_current_device()
print(f"Device name: {device.name.decode('utf-8')}")

beta = 1.20
rdm = rd2.Randium_2d_gpu(threads_per_block=(4, 4), blocks=(12, 12), tiles=(4, 4), num_of_each_type=1)
print(rdm)
num_threads = int(np.prod(rdm.threads_per_block + rdm.blocks))
print(f'{num_threads = }')

lat_ref = rdm.lattice.copy()
wc = rdm.run()  # Compile
print(f'Compile (local GPU): {wc} ms')
wc = rdm.run(beta=beta, steps=512)
print(f'Equilibrate (local): {wc} ms')
print(f"MC attempts per second (GPU): {rdm.benchmark()['mc_attempts_per_sec']:0.2e}")
tic = perf_counter()
rdm.run_global(beta=beta)  # Compile
toc = perf_counter()
print(f'Compile (global CPU): {(toc - tic) * 1000} ms')

steps = 16
tic = perf_counter()
rdm.run_global(beta=beta, steps=steps)
toc = perf_counter()
print(f'Equilibrate (global): {(toc - tic) * 1000:.1f} ms = {toc - tic:.1f} s = {(toc - tic)/60:.1f} minutes ')
mc_attempts_per_sec = rdm.N * steps / (toc - tic)
print(f'MC attempts per second (global CPU): {mc_attempts_per_sec:0.2e}')
print(f'Speed-up: {rdm.benchmark()["mc_attempts_per_sec"] / mc_attempts_per_sec}')

steps = 2048*8
mc_attempts = steps*rdm.N*4
run_est = mc_attempts/rdm.benchmark()["mc_attempts_per_sec"]
print(f'Estimated equbriliation runtime: {run_est:0.1f} s = {run_est / 60:0.1f} minutes ')
print(f'                                 = {run_est / 3600:0.1f} hours = {run_est / 3600 / 24 :0.1f} days')
wc = rdm.run(beta=beta, steps=steps)
print(f'Equilibrate (local, extra): {wc} ms = {wc/1000:.1f} s = {wc/1000/60:.1f} minutes = {wc/1000/3600:.1f} hours')


for _ in range(4):
    tic = perf_counter()
    u = rdm.energy()
    toc = perf_counter()
    print(f'Energy: {u} (in {(toc - tic) * 1000:0.2f} ms)')

time_blocks = 16
block_energies = []
block_times = []
inner_steps = [int(1.4 ** x) for x in range(1, 22)]  #
print(f'{inner_steps = }')
overlap_table = []
store_lattices = []
mc_attempts = time_blocks*sum(inner_steps)*rdm.N*4
run_est = mc_attempts / rdm.benchmark()["mc_attempts_per_sec"]
print(f'Estimated runtime: {run_est:0.1f} s = {run_est / 60:0.1f} minutes ')
print(f'                   = {run_est / 3600:0.1f} hours = {run_est / 3600 / 24 :0.1f} days')

for time_block in range(time_blocks):
    print(f'  Time block: {time_block}')
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
        print(f'{time = }, {overlap = :.3}, {energy = :.4},  {-0.5*energy/beta = :.4}')
        block_energies.append(energy)
        block_times.append(time)
    overlap_table.append(this_overlaps)
    store_lattices.append(rdm.lattice.copy())
print(f"MC attempts per second (GPU): {rdm.benchmark()['mc_attempts_per_sec']:0.2e}")

plt.figure()
plt.plot(block_times, block_energies, '-')
plt.xlabel(r'Time, $t$ [swap attempts per particle]')
plt.ylabel(r'Energy, $E/N$')
plt.show()

times = np.array(times)
overlaps = np.array(overlap_table).mean(axis=0)
overlaps = (overlaps - 1 / rdm.M) / (1 - 1 / rdm.M)  # Normalize
print(f'{overlaps = }  ({1/rdm.M = :0.2e})')
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
    energies=[float(x) for x in block_energies]
), open(f'data/local_{rdm.rows}x{rdm.cols}M{rdm.M}beta{beta:.4f}.toml', 'w'))

# Save lattice configurations
store_lattices = np.array(store_lattices, dtype=np.uint32)
fname = f'data/local_{rdm.rows}x{rdm.cols}M{rdm.M}beta{beta:.4f}.npy'
np.save(fname, store_lattices)
