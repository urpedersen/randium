import math
import numpy as np
import numba
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


def inverf(x: np.float32) -> np.float32:
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


def get_shuffle_idx(
        type0: np.uint32,
        type1: np.uint32,
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32
) -> np.uint32:
    """ Return a shuffled index in a Strictly Upper Triangular Matrix.
    Note: Ensure that gcd(a, N_M) = 1 where N_M = M * (M - 1) // 2 and M is num_types.
     """
    if type0 == type1:  # Handle special diagonal
        return np.nan

    # Parameters, N.B. be aware of integer overflow (see error by making 'a' large)
    M = np.uint32(num_types)

    i = min(type0, type1)  # j > i
    j = max(type0, type1)
    idx = i * (M - 1) - i * (i - 1) // 2 + (j - i - 1)  # Index in Strictly Upper Triangular Matrix
    N_M = M * (M - 1) // 2  # Number of unique indexes
    for _ in range(c):
        idx = (a * idx + b) % N_M  # Shuffle index
    return idx


h_get_shuffle_idx = numba.jit(get_shuffle_idx)  # Host function
d_get_shuffle_idx = cuda.jit(device=True)(get_shuffle_idx)  # Device function


@numba.njit
def h_get_pair_energy(
        type0: np.uint32,
        type1: np.uint32,
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32
) -> np.float32:
    """ Host function: Return energy of the pair of types i and j. """
    M = num_types
    if type0 == type1:  # Handle special diagonal
        return np.float32(np.inf)
    idx_shuffle = h_get_shuffle_idx(type0, type1, num_types, a, b, c)
    N_M = M * (M - 1) // 2
    x_k = (2 * idx_shuffle + 1) / N_M - 1  # -1 < x_k < +1
    energy = h_inverf(x_k)
    return energy


@cuda.jit(device=True)
def d_get_pair_energy(
        type0: np.uint32,
        type1: np.uint32,
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32
) -> np.float32:
    """ Device function: Return energy of the pair of types i and j. """
    one = np.float32(1.0)
    two = np.float32(2.0)
    M = num_types
    if type0 == type1:  # Handle special diagonal
        return np.float32(np.inf)
    idx_shuffle = d_get_shuffle_idx(type0, type1, num_types, a, b, c)
    N_M = M * (M - 1) // 2
    x_k = (two * idx_shuffle + one) / N_M - one  # -1 < x_k < +1
    energy = d_inverf(x_k)
    return energy


@numba.njit
def h_get_particle_energy(
        lattice: np.ndarray[tuple[np.uint32, np.uint32]],
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32,
        xx: np.uint32,
        yy: np.uint32
) -> np.float32:
    """ Host function: Return energy of the particle located at (xx, yy) in lattice. """
    rows, columns = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        xx1 = (xx + dx) % rows
        yy1 = (yy + dy) % columns
        that_type = lattice[xx1, yy1]
        energy += h_get_pair_energy(this_type, that_type, num_types, a, b, c)
    return energy


@cuda.jit(device=True)
def d_get_particle_energy(
        lattice: np.ndarray[tuple[np.uint32, np.uint32]],
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32,
        xx: np.uint32,
        yy: np.uint32
) -> np.float32:
    """ Device function: Return energy of the particle located at (xx, yy) in lattice. """
    rows, columns = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy]
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        xx1 = (xx + dx) % rows
        yy1 = (yy + dy) % columns
        that_type = lattice[xx1, yy1]
        energy += d_get_pair_energy(this_type, that_type, num_types, a, b, c)
    return energy


cuda.jit(device=True)
def d_update(
        lattice: np.ndarray[tuple[np.uint32, np.uint32]],
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32,
        beta: np.float32,
        tiles: tuple[np.uint32, np.uint32],
        rng_states,
        step: np.uint32,
        x: np.uint32,
        y: np.uint32
):
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
        energy0_old = d_get_particle_energy(lattice, num_types, a, b, c, xx0, yy0)
        xx1 = (xx0 + dx) % rows
        yy1 = (yy0 + dy) % columns
        energy1_old = d_get_particle_energy(lattice, num_types, a, b, c, xx1, yy1)
        this_type = lattice[xx0, yy0]
        that_type = lattice[xx1, yy1]
        lattice[xx0, yy0] = that_type
        lattice[xx1, yy1] = this_type
        energy0_new = d_get_particle_energy(lattice, num_types, a, b, c, xx0, yy0)
        energy1_new = d_get_particle_energy(lattice, num_types, a, b, c, xx1, yy1)
        delta = (energy0_new + energy1_new) - (energy0_old + energy1_old)
        rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if rnd < math.exp(-beta * delta):
            ...  # Accept move
        else:
            lattice[xx0, yy0] = this_type
            lattice[xx1, yy1] = that_type

@cuda.jit
def kernel_run_simulation(
        lattice: np.ndarray[tuple[np.uint32, np.uint32]],
        num_types: np.uint32,
        a: np.uint32,
        b: np.uint32,
        c: np.uint32,
        beta: np.float32,
        tiles: tuple[np.uint32, np.uint32],
        rng_states,
        steps: np.uint32
) -> None:
    """ GPU Kernel than run simulation.
    All sites in tile is attempted to swapped left, right, up and down """
    x, y = cuda.grid(2)
    grid = cuda.cg.this_grid()
    tile_size = tiles[0]*tiles[1]

    for _ in np.arange(steps, dtype=np.uint32):
        for tile_step in np.arange(tile_size, dtype=np.uint32):
            d_update(lattice, num_types, a, b, c, beta, tiles, rng_states, tile_step, x, y)
            grid.sync()


@numba.njit
def h_get_lattice_energy(
        lattice: np.ndarray[tuple[np.uint32, np.uint32]],
        num_types: np.uint32,
) -> np.float32:
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
