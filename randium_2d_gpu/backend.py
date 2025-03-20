import math
import numpy as np
import numba
from numba import cuda

import functools

from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_uniform_float32
from numba import cuda, uint32, uint64

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


h_inverf = numba.jit(inverf)  # Host function
d_inverf = cuda.jit(device=True)(inverf)  # Device function


def get_shuffle_idx_ulf(type0, type1, num_types):
    """ Return a shuffled index in a Strictly Upper Triangular Matrix.
    Note: Ensure that gcd(a, N_M) = 1 where N_M = M * (M - 1) // 2 and M is num_types.
     """
    if type0 == type1:  # Handle special diagonal
        return np.uint32(np.nan)

    # Parameters, N.B. be aware of integer overflow (see error by making 'a' large)
    M = np.uint64(num_types)

    i = np.uint64(min(type0, type1))  # j > i
    j = np.uint64(max(type0, type1))
    idx = np.uint64(i * (M - 1) - i * (i - 1) // 2 + (j - i - 1))  # Index in Strictly Upper Triangular Matrix
    N_M = np.uint64(M * (M - 1) // 2)  # Number of unique indexes
    # 5**2 * 139 * 479 = 166452, Note: gcd(1664525, N_M) should be one
    for _ in range(8):
        idx = np.uint64((1664525 * idx + 1013904223) % N_M)  # Shuffle index
        #idx = np.uint64((7919 * idx + 123456) % N_M)  # Shuffle index
    return np.uint32(idx)




def get_shuffle_idx_wang(type0, type1, num_types):
    """
    Returns a shuffled index in a strictly upper triangular matrix using an improved hash.

    Parameters:
      type0 (uint32): First type index.
      type1 (uint32): Second type index.
      num_types (uint32): Total number of types.

    Returns:
      uint32: A shuffled index corresponding to the (type0, type1) pair when type0 != type1.
              Returns 0xFFFFFFFF as a marker if type0 equals type1.

    This device function computes the linear index into the strictly upper triangular
    part of a matrix and then scrambles it using a variant of Wang’s 32-bit integer hash.

    N.B this algorithm does not preserve the number of pairs
    """
    # Handle the special diagonal case.
    if type0 == type1:
        return uint32(0xFFFFFFFF)  # Marker for diagonal elements

    # Cast parameters to 64-bit to prevent overflow
    M = uint64(num_types)
    # Ensure that i < j for a strictly upper triangular matrix.
    if type0 < type1:
        i = uint64(type0)
        j = uint64(type1)
    else:
        i = uint64(type1)
        j = uint64(type0)

    # Compute linear index for the strictly upper triangular matrix.
    # The formula below maps the (i, j) pair to an index in [0, N_M-1],
    # where N_M = M * (M - 1) // 2 is the number of unique (i, j) pairs.
    idx = i * (M - 1) - i * (i - 1) // 2 + (j - i - 1)
    N_M = M * (M - 1) // 2  # Total number of unique indexes

    # Linear congruential generator (LCG) pre-shuffle avoid (0, 1) is mapped to 0
    # idx = (idx*1664525+1013904223)%N_M

    # Apply a mixing function (Wang’s 32-bit hash variant)
    idx = (~idx) + (idx << 15)
    idx = idx ^ (idx >> 12)
    idx = idx + (idx << 2)
    idx = idx ^ (idx >> 4)
    idx = idx * uint64(2057)
    idx = idx ^ (idx >> 16)

    # Map back into the valid range [0, N_M)
    idx = idx % N_M

    return uint32(idx)

get_shuffle_idx = get_shuffle_idx_ulf
h_get_shuffle_idx = numba.jit(get_shuffle_idx)  # Host function
d_get_shuffle_idx = cuda.jit(device=True)(get_shuffle_idx)  # Device function


@numba.njit
def h_get_pair_energy(type0, type1, num_types):
    """ Host function: Return energy of the pair of types i and j. """
    M = num_types
    if type0 == type1:  # Handle special diagonal
        return np.float32(np.inf)
    idx_shuffle = h_get_shuffle_idx(type0, type1, num_types)
    N_M = M * (M - 1) // 2
    x_k = (2 * idx_shuffle + 1) / N_M - 1  # -1 < x_k < +1
    energy = math.sqrt(2.0)*h_inverf(x_k)
    return energy


@cuda.jit(device=True)
def d_get_pair_energy(type0, type1, num_types):
    """ Device function: Return energy of the pair of types i and j. """
    one = np.float32(1.0)
    two = np.float32(2.0)
    M = num_types
    if type0 == type1:  # Handle special diagonal
        return np.float32(np.inf)
    idx_shuffle = d_get_shuffle_idx(type0, type1, num_types)
    N_M = M * (M - 1) // 2
    x_k = (two * idx_shuffle + one) / N_M - one  # -1 < x_k < +1
    energy = d_inverf(x_k)
    return energy


@numba.njit
def h_get_particle_energy(lattice, num_types, xx, yy):
    """ Host function: Return energy of the particle located at (xx, yy) in lattice. """
    rows, columns = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        xx1 = (xx + dx) % rows
        yy1 = (yy + dy) % columns
        that_type = lattice[xx1, yy1]
        energy += h_get_pair_energy(this_type, that_type, num_types)
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

@numba.njit(parallel=True)
def h_lattice_energy(lattice, num_types):
    rows, columns = lattice.shape
    energy = 0.0
    for xx in numba.prange(0, rows):
        for yy in range(0, columns):
            energy += h_get_particle_energy(lattice, num_types, xx, yy)
    return energy

@numba.njit
def h_global_mc(lattice, num_types, beta=1.0, steps_per_particle=1):
    rows, columns = lattice.shape
    for _ in range(steps_per_particle):
        for xx1 in range(0, rows):
            for yy1 in range(0, columns):
                t1 = lattice[xx1, yy1]
                xx2 = np.random.randint(rows)
                yy2 = np.random.randint(columns)
                t2 = lattice[xx2, yy2]
                u1_old = h_get_particle_energy(lattice, num_types, xx1, yy1)
                u2_old = h_get_particle_energy(lattice, num_types, xx2, yy2)
                lattice[xx1, yy1] = t2
                lattice[xx2, yy2] = t1
                u1_new = h_get_particle_energy(lattice, num_types, xx1, yy1)
                u2_new = h_get_particle_energy(lattice, num_types, xx2, yy2)
                delta = u2_new + u1_new - (u2_old + u1_old)
                rnd = np.random.random()
                if rnd < np.exp(-beta*delta):
                    pass  # Accept move
                else:
                    lattice[xx1, yy1] = t1
                    lattice[xx2, yy2] = t2
    return lattice

@functools.lru_cache(maxsize=1024)
def c_get_pair_energy(type0, type1, num_types):
    return h_get_pair_energy(type0, type1, num_types)

def c_get_particle_energy(lattice, num_types, xx, yy):
    rows, columns = lattice.shape
    energy = 0.0
    this_type = lattice[xx, yy]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        xx1 = (xx + dx) % rows
        yy1 = (yy + dy) % columns
        that_type = lattice[xx1, yy1]
        energy += c_get_pair_energy(this_type, that_type, num_types)
    return energy

def c_global_mc(lattice, num_types, beta=1.0, steps_per_particle=1):
    rows, columns = lattice.shape
    for _ in range(steps_per_particle):
        for xx1 in range(0, rows):
            for yy1 in range(0, columns):
                t1 = lattice[xx1, yy1]
                xx2 = np.random.randint(rows)
                yy2 = np.random.randint(columns)
                t2 = lattice[xx2, yy2]
                u1_old = c_get_particle_energy(lattice, num_types, xx1, yy1)
                u2_old = c_get_particle_energy(lattice, num_types, xx2, yy2)
                lattice[xx1, yy1] = t2
                lattice[xx2, yy2] = t1
                u1_new = c_get_particle_energy(lattice, num_types, xx1, yy1)
                u2_new = c_get_particle_energy(lattice, num_types, xx2, yy2)
                delta = u2_new + u1_new - (u2_old + u1_old)
                rnd = np.random.random()
                if rnd < np.exp(-beta*delta):
                    pass  # Accept move
                else:
                    lattice[xx1, yy1] = t1
                    lattice[xx2, yy2] = t2
    return lattice


@cuda.jit(device=True)
def d_update(lattice, num_types, beta, tiles, rng_states, step, x, y):
    """ Device function: updates state """
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
            pass  # Accept move
        else:
            lattice[xx0, yy0] = this_type
            lattice[xx1, yy1] = that_type


@cuda.jit
def kernel_run_simulation(lattice, num_types, beta, tiles, rng_states, steps):
    """ GPU Kernel than run simulation.
    All sites in tile is attempted to swapped left, right, up and down """
    x, y = cuda.grid(2)
    grid = cuda.cg.this_grid()
    tile_size = tiles[0] * tiles[1]

    for _ in range(steps):
        for tile_step in range(tile_size):
            d_update(lattice, num_types, beta, tiles, rng_states, tile_step, x, y)
            grid.sync()


@numba.njit
def h_get_lattice_energy(lattice, num_types):
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
    threads_per_block = (8, 8)
    blocks = (16, 16)
    tiles = (8, 8)

    rows = tiles[0] * blocks[0] * threads_per_block[0]
    cols = tiles[1] * blocks[1] * threads_per_block[1]
    N = rows * cols

    num_of_each_type = 1024 // 16
    num_types = N // num_of_each_type

    # Setup Lattice
    lattice = np.array([[t] * num_of_each_type for t in range(num_types)], dtype=np.int32).flatten()
    np.random.shuffle(lattice)
    lattice = lattice.reshape((rows, cols))
    d_lattice = cuda.to_device(lattice)

    tile_size = tiles[0] * tiles[1]
    n_threads = N // tile_size
    rng_states = create_xoroshiro128p_states(n_threads, seed=2025)

    kernel_run_simulation[blocks, threads_per_block](d_lattice, num_types, 1.0, tiles, rng_states, 10)


if __name__ == "__main__":
    main()


