import math

import numpy as np

from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float32
from numba import cuda

@cuda.jit(device=True)
def d_get_particle_energy(lattice, interactions, num_types, xx, yy):
    rows, columns = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        xx1 = (xx + dx) % rows
        yy1 = (yy + dy) % columns
        that_type = lattice[xx1, yy1]
        if this_type == that_type:
            energy += np.float32(np.inf)
        i = max(this_type, that_type)
        j = min(this_type, that_type)
        n = num_types
        idx = j*n - j*(j+1)//2 + (i - j - 1)
        energy += np.float32(interactions[idx])
    return energy

@cuda.jit(device=True)
def d_update(lattice, interactions, num_types, beta, tiles, rng_states, step, x, y):

    # Helper variables
    rows, columns = lattice.shape
    block_rows = rows // tiles[0]
    thread_id = x + y * block_rows
    xx = x * tiles[0]  # My upper-left tile
    yy = y * tiles[1]

    # Find cell where I'm allowed try neighbour swaps
    tile_size = tiles[0] * tiles[1]
    tile_idx = step % tile_size
    tx = tile_idx % tiles[0]
    ty = tile_idx // tiles[0]
    xx0 = xx + tx
    yy0 = yy + ty

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        energy0_old = d_get_particle_energy(lattice, interactions, num_types, xx0, yy0)
        xx1 = (xx0 + dx) % rows
        yy1 = (yy0 + dy) % columns
        energy1_old = d_get_particle_energy(lattice, interactions, num_types, xx1, yy1)
        this_type = lattice[xx0, yy0]
        that_type = lattice[xx1, yy1]
        lattice[xx0, yy0] = that_type
        lattice[xx1, yy1] = this_type
        energy0_new = d_get_particle_energy(lattice, interactions, num_types, xx0, yy0)
        energy1_new = d_get_particle_energy(lattice, interactions, num_types, xx1, yy1)
        delta = (energy0_new + energy1_new) - (energy0_old + energy1_old)
        rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if rnd < math.exp(-beta * delta):
            pass  # Accept move
        else:
            lattice[xx0, yy0] = this_type
            lattice[xx1, yy1] = that_type


@cuda.jit
def kernel_run_simulation(lattice, interactions, num_types, beta, tiles, rng_states, steps):
    x, y = cuda.grid(2)
    grid = cuda.cg.this_grid()
    tile_size = tiles[0] * tiles[1]

    for _ in range(steps):
        for tile_step in range(tile_size):
            d_update(lattice, interactions, num_types, beta, tiles, rng_states, tile_step, x, y)
            grid.sync()

def main():
    threads_per_block = (8, 8)
    blocks = (16, 16)
    tiles = (8, 8)

    rows = tiles[0] * blocks[0] * threads_per_block[0]
    cols = tiles[1] * blocks[1] * threads_per_block[1]
    N = rows * cols

    num_of_each_type = N_m = 512
    num_types = M = N // num_of_each_type
    num_unique_pairs = N_M = M*(M-1)//2
    print(f'{rows}x{cols} = {N}, {num_of_each_type = }, {num_types = }, {num_unique_pairs = }')

    # Setup Lattice
    lattice = np.array([[t] * num_of_each_type for t in range(num_types)], dtype=np.int32).flatten()
    np.random.shuffle(lattice)
    lattice = lattice.reshape((rows, cols))
    d_lattice = cuda.to_device(lattice)

    # Setup interaction matrix elements
    interactions = np.array(np.random.randn(N_M), dtype=np.float32)
    d_interactions = cuda.to_device(interactions)

    # Setup random number generator
    tile_size = tiles[0] * tiles[1]
    n_threads = N // tile_size
    rng_states = create_xoroshiro128p_states(n_threads, seed=2025)

    # For timing of device code
    start = cuda.event()
    end = cuda.event()

    # Run simulation
    beta = 1.0
    steps_per_timeblock = 128
    mc_steps = 4 * N * steps_per_timeblock
    for _ in range(8):
        start.record()
        kernel_run_simulation[blocks, threads_per_block](d_lattice, d_interactions, num_types, beta, tiles, rng_states, steps_per_timeblock)
        end.record()
        end.synchronize()
        wall_clock = start.elapsed_time(end)  # ms
        mc_steps_per_second = mc_steps / (wall_clock / 1000.0)
        print(f'{wall_clock = :.1f} ms, {mc_steps_per_second:.2e} mc steps per second')


if __name__ == '__main__':
    main()
