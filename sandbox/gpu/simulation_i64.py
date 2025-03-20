""" Randium with Local Moves on the GPU

- This code implements the 2D Randium model with local (Monte-Carlo) particle swaps.
- Each particle is assigned a specific type.
- The interaction energy of a type-pair is determined
by a (unique) pseudo-random function that approximates the normal distribution.
- The main function makes a simulation, calculate the total energy and report performance.

Implementation Details:
-----------------------

Energy Computation:

- The pair interaction energy is computed generating a shuffled index of an interaction matrix
via an affine transformation.
- This index is then mapped to a value between -1 and 1, which is converted into an
energy value using an approximation of the inverse error function.

Local Moves  and Monte Carlo updates:

- The simulation applies local moves by attempting to swap neighboring particles.
- The move is accepted or rejected based on the Metropolis criterion

GPU Acceleration:

- Using Numba’s CUDA JIT to executes the local updates in parallel in a GPU.
- Dividing the lattice into tiles to avoid race conditions of local type swaps.
- Functions are provided for host (CPU) and device (GPU).

Testing:

- Set `TESTING = True` below to execute tests.

Author: Ulf R. Pedersen
"""

import math

import numba
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float32

TESTING = False


def inverf(x):
    """ S. Winitzki’s (2008)
    A handy approximation for the error function and its inverse
    Lecture Notes in Computer Science series, volume 2667
    https://link.springer.com/chapter/10.1007/3-540-44839-X_82
    See also: Approximations to inverse error functions, Stephen Dyer
    See also: https://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
    See also: https://www.scribd.com/document/82414963/Winitzki-Approximation-to-Error-Function
    """
    one = np.float32(1.0)
    two = np.float32(2.0)
    one_half = np.float32(0.5)
    a = np.float32(0.147)  # 0.1400122886866665
    s = math.copysign(one, x)
    xx = one - x * x
    log_xx = math.log(xx)
    t = two / (math.pi * a) + one_half * log_xx
    inner = t * t - (one / a) * log_xx
    result = s * math.sqrt(math.sqrt(inner) - t)  # Eq. (7) in "A handy approximation ..."
    return result


d_inverf = cuda.jit(device=True)(inverf)  # Device function
h_inverf = numba.jit(inverf)  # Host function


def test_inverf():
    """ Test the inverf function """
    import scipy
    xs = [-0.99999, -0.9999, -0.999, -0.99, -0.9, 0.0, 0.999, 0.99, 0.9, 0.9999, 0.99999]
    for x in xs:
        print(f'Test inverf: {x = }: {inverf(x) = }, {h_inverf(x) = }, {scipy.special.erfinv(x) = },')
        assert np.isclose(h_inverf(x), scipy.special.erfinv(x), atol=1e-2)


if TESTING:
    test_inverf()


def test_affine_shuffle():
    """ This function can be used to investigate affine shuffling of integers """

    def get_affine_shuffle(x, n, a, b, c=1):
        for _ in range(c):
            x = (a * x + b) % n
        return x

    M = 16384
    n = np.int32(M * (M - 1) // 2)  # np.int32(2**20)
    a = np.int32(7)  # Integer overflow with: np.int32(733*13)
    b = np.int32(0)
    c = np.int32(8)
    print(f'TEST affine shuffle: {n = }, {a = }, {b = }, {c = }, {math.gcd(n, a) = } (should be 1)')
    assert math.gcd(n, a) == 1, "Ensure that gcd(a, n) = 1"
    xs = [0, 1, 2, 3, 5, M - 4, M - 3, M - 2, M - 1, n - 4, n - 3, n - 2, n - 1]
    for x in xs:
        print(f'TEST affine shuffle: {x = }: {(a*x + b)%n = }, {get_affine_shuffle(x, n, a, b, c) = }')


if TESTING:
    test_affine_shuffle()


def get_shuffle_idx(type0, type1, num_types):
    """ Return a shuffled index in a Strictly Upper Triangular Matrix """
    if type0 == type1:  # Handle special diagonal
        return np.uint32(np.nan)

    # Parameters, N.B. be aware of integer overflow (see error by making 'a' large)
    a = np.uint64(7)
    b = np.uint64(0)
    c = np.uint64(8)
    M = np.uint64(num_types)

    i = np.uint64(min(type0, type1))  # j > i
    j = np.uint64(max(type0, type1))
    idx = np.uint64(i * (M - 1) - i * (i - 1) // 2 + (j - i - 1))  # Index in Strictly Upper Triangular Matrix
    N_M = np.uint64(M * (M - 1) // 2)  # Number of unique indexes
    for _ in range(c):
        idx = np.uint64((a * idx + b) % N_M)  # Shuffle index
    return np.uint32(idx)


d_get_shuffle_idx = cuda.jit(device=True)(get_shuffle_idx)  # Device function
h_get_shuffle_idx = numba.jit(get_shuffle_idx)  # Host function


def test_get_shuffle_idx():
    # Test small matrix
    M = 1024
    xs = range(M * M)
    N_M = M * (M - 1) // 2
    x_shuffle = {h_get_shuffle_idx(x % M, x // M, M) for x in xs}
    x_shuffle.remove(-1)  # Remove placeholder of diagonal
    assert len(x_shuffle) == N_M, "The get_shuffle_idx is NOT a bijection"

    # Test large matrix
    M = 16384
    for i, j in ((0, 1), (0, 2), (5, 5), (5, 6), (6, 5), (102, 0)):
        print(f'TEST get_shuffle_idx: {i = }, {j = }, {M = }, {get_shuffle_idx(i, j, M) = }')
    columns = [0, 1, 2, M // 3 + 1, M // 3 + 2, M // 3 + 3, M // 3 + 4, M // 2 + 1, M // 2 + 2, M // 2 + 3, M // 2 + 4,
               M - 3, M - 2, M - 1]
    plt.figure()
    plt.title('test of get_shuffle_idx')
    for j in columns:
        idx_list = [get_shuffle_idx(i, j, M) for i in range(M)]
        plt.plot(idx_list, '+', label=f'column {j}', alpha=0.3)
    plt.xlabel(r'row $i$')
    plt.ylabel(r'Shuffled index')
    plt.legend()
    plt.show()


if TESTING:
    test_get_shuffle_idx()


@numba.njit
def h_get_pair_energy(type0, type1, num_types):
    """ Host function: Return energy of the pair of types i and j. """
    M = num_types
    if type0 == type1:  # Handle special diagonal
        return np.inf
    idx_shuffle = h_get_shuffle_idx(type0, type1, num_types)
    N_M = M * (M - 1) // 2
    x_k = (2 * idx_shuffle + 1) / N_M - 1  # -1 < x_k < +1
    energy = h_inverf(x_k)
    return energy


def test_h_get_pair_energy():
    M = 16384
    this_type = 0
    for that_type in range(10):
        print(
            f'TEST h_get_pair_energy: {M = }, {this_type = }, {that_type = }, {h_get_pair_energy(this_type, that_type, M)}')
    plt.figure()
    plt.title('test of h_get_pair_energy')
    for i in 0, 100, 16384 // 3, 16384 // 2, 16384:
        plt.plot([h_get_pair_energy(i, j, M) for j in range(512)], '-', label=f'Type {i = }')
    plt.xlabel(r'Other particle type, j')
    plt.legend()
    plt.show()


if TESTING:
    test_h_get_pair_energy()


@cuda.jit(device=True)
def d_get_pair_energy(type0, type1, num_types):
    """ Device function: Return energy of the pair of types i and j. """
    one = np.float32(1.0)
    two = np.float32(2.0)
    M = num_types
    if type0 == type1:  # Handle special diagonal
        return np.inf
    idx_shuffle = d_get_shuffle_idx(type0, type1, num_types)
    N_M = M * (M - 1) // 2
    x_k = (two * idx_shuffle + one) / N_M - one  # -1 < x_k < +1
    energy = d_inverf(x_k)
    return energy


@cuda.jit(device=True)
def d_get_particle_energy(lattice, num_types, xx, yy):
    """ Device function: Return energy of the particle located at (xx, yy) in lattice. """
    rows, columns = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        xx1 = (xx + dx) % rows
        yy1 = (yy + dy) % columns
        that_type = lattice[xx1, yy1]
        energy += d_get_pair_energy(this_type, that_type, num_types)
    return energy


@cuda.jit(device=True)
def d_update(lattice, num_types, beta, tiles, rng_states, step, x, y):
    """ Device function: updates state """
    # Parameters
    beta = numba.float32(beta)  # Inverse temperature

    # Helper variables
    rows, columns = lattice.shape
    block_rows = rows // tiles[0]
    thread_id = x + y * block_rows
    xx = x * tiles[0]   # My upper-left tile
    yy = y * tiles[1]

    # Find cell where I'm allowed try neighbour swaps
    tile_size = tiles[0] * tiles[1]
    tile_idx = step % tile_size
    tx = tile_idx % tiles[0]
    ty = tile_idx // tiles[0]
    xx0 = xx + tx
    yy0 = yy + ty

    # Try swaps on neighbours
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        energy0_old = d_get_particle_energy(lattice, num_types, xx0, yy0)
        xx1 = (xx0 + dx) % rows
        yy1 = (yy0 + dy) % columns
        energy1_old = d_get_particle_energy(lattice, num_types, xx1, yy1)
        this_type = lattice[xx0, yy0]
        that_type = lattice[xx1, yy1]
        lattice[xx0, yy0] = that_type
        lattice[xx1, yy1] = this_type
        energy0_new = d_get_particle_energy(lattice, num_types, xx0, yy0)
        energy1_new = d_get_particle_energy(lattice, num_types, xx1, yy1)
        delta = (energy0_new + energy1_new) - (energy0_old + energy1_old)
        rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if rnd < math.exp(-beta * delta):
            ...  # Accept move
        else:
            lattice[xx0, yy0] = this_type
            lattice[xx1, yy1] = that_type

        # if x == 0 and y == 0:
        #     print(dx, dy, rnd)
        #     print(this_type, that_type, d_get_pair_energy(this_type, that_type, num_types))


@cuda.jit
def run_simulation(lattice, num_types, beta, tiles, rng_states, steps):
    """ Kernel than run simulation for one spin step on the device """
    x, y = cuda.grid(2)
    grid = cuda.cg.this_grid()
    tile_size = tiles[0]*tiles[1]

    for step in range(steps):
        for tile_step in range(tile_size):
            d_update(lattice, num_types, beta, tiles, rng_states, tile_step, x, y)
            grid.sync()


@numba.njit
def get_lattice_energy(lattice, num_types):
    """ Host function. Return energy of the lattice. """
    rows, columns = lattice.shape
    energy = 0.0
    M = np.int32(num_types)
    for row in range(rows):
        for col in range(columns):
            this_type = lattice[row, col]
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                that_type = lattice[(row + dx) % rows, (col + dy) % columns]
                energy += 0.5 * h_get_pair_energy(this_type, that_type, M)
    return energy / lattice.size


def main():
    device = cuda.get_current_device()
    print(f"Device name: {device.name.decode('utf-8')}")
    cc = device.compute_capability
    print(f"Compute capability: {cc[0]}.{cc[1]}")

    # For timing of device code
    start = cuda.event()
    end = cuda.event()
    start_block = cuda.event()
    end_block = cuda.event()
    start.record()

    # Setup system size
    threads_per_block = (8, 8)
    blocks = (16, 16)  # (24, 24)
    tiles = (8, 8)  # (24, 24)
    if tiles[0] < 6 or tiles[1] < 6:
        raise ValueError("Increase tile size to avoid race condition")
    rows = tiles[0] * blocks[0] * threads_per_block[0]
    cols = tiles[1] * blocks[1] * threads_per_block[1]
    N = rows * cols
    print(f"Lattice size: N = {rows} x {cols} = {N}")
    print(f"with {tiles = }, {blocks = }, and {threads_per_block = } ")

    # Setup initial lattice
    num_of_each_type = 1024 // 16
    # Assert that number of types is deducible by number of particles
    num_types = N // num_of_each_type
    if num_of_each_type * num_types != N:
        raise ValueError("Number of types is not a multiple of number of types")

    # Setup Lattice
    print(f"M = {num_types = }, {num_of_each_type = }, (N={num_of_each_type * num_types = })")
    lattice = np.array([[t] * num_of_each_type for t in range(num_types)], dtype=np.int32).flatten()
    np.random.shuffle(lattice)
    lattice = lattice.reshape((rows, cols))

    # Setup random number generator
    tile_size = tiles[0] * tiles[1]
    n_threads = N // tile_size
    rng_states = create_xoroshiro128p_states(n_threads, seed=2025)

    # Copy data to device
    d_lattice = cuda.to_device(lattice)

    # Run simulation on device
    wallclock_times = []
    time_blocks = 8
    steps_per_time_block = 16
    mc_attempts_per_step = rows * cols * 4
    beta = 2.0
    print(f'{time_blocks=}, {steps_per_time_block=}')
    print(f'Total number of time steps: {time_blocks * steps_per_time_block}')
    print(f'MC attempt per step: {mc_attempts_per_step}')
    print(f'Total number of mc attempts: {mc_attempts_per_step*time_blocks*steps_per_time_block}')
    for time_block in range(time_blocks):
        start_block.record()
        run_simulation[blocks, threads_per_block](d_lattice, num_types, beta, tiles, rng_states, steps_per_time_block)
        end_block.record()
        end_block.synchronize()
        wallclock_times.append(start_block.elapsed_time(end_block))
        lattice = d_lattice.copy_to_host()
        ptype = int(lattice[0][0])
        energy = get_lattice_energy(lattice, num_types)
        print(f'{time_block:>4}: lattice[0][0] = {ptype}, {energy = }, {wallclock_times[-1] = } ms')
    end.record()
    end.synchronize()
    total_wallclock_time = cuda.event_elapsed_time(start, end)
    print(f'{total_wallclock_time = :0.1f} ms')

    # Print benchmark data
    print(f"First, wallclock time (compile): {wallclock_times[0] = :0.2f} ms")
    print(f"Other avg. wallclock time: {np.mean(wallclock_times[1:]):0.1f} ms +- {np.std(wallclock_times[1:]):0.1f} ms")
    delta_t = np.mean(wallclock_times[1:]) / 1000  # Seconds
    steps_per_second = steps_per_time_block / delta_t
    print(f"{steps_per_second = :0.2e}")
    mc_attempts_per_second = steps_per_second * mc_attempts_per_step
    print(f"{mc_attempts_per_second = :0.2e}")


if __name__ == "__main__":
    main()
