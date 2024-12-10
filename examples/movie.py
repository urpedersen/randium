import matplotlib.pyplot as plt
import randium as rd
import numpy as np
import time

L = 64
M = 2048//4
beta = 1.6
lat = rd.Lattice(L, M, D=2, beta=beta)
print(f'{lat.N=} {lat.M=} {lat.N_m=} {lat.beta=}')

steps_eq = lat.N*2048*128
steps = lat.N*2048*128
print_stride = steps

print('Equilibrating...')
for _ in range(2):
    tic = time.perf_counter()
    energies, acceptance_rate = lat.simulation_monte_carlo_global(
        steps=steps_eq, print_stride=steps_eq//1024)
    toc = time.perf_counter()
# Plot energy trajectory
print(f'Total energy: {lat.get_total_energy()}')
print(f'Time elapsed: {toc - tic:0.4f} seconds, acceptance rate: {acceptance_rate:.4f}')
plt.figure()
plt.plot(energies)
plt.show()

print('Generating movie...')
frames = 240  # *5
image_folder = 'movie'
for frame in range(frames):
    if (frame)%100 == 0 and frame != 0:
        print(f'| {frame}\n|', end='', flush=True)
    elif frame%20 ==0:
        print('|', end='', flush=True)
    elif frame%10 ==0:
        print(':', end='', flush=True)
    else:
        print('.', end='', flush=True)
    _ = lat.simulation_monte_carlo_local(steps=steps)
    lat.to_png(f'{image_folder}/{frame:04d}.png')
print('')

# Convert to movie with
print('  Generate movie with someething like')
print('ffmpeg -framerate 24 -i movie/%04d.png -c:v mpeg1video movie.mpg')
