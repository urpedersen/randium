import time

import numpy as np
from scipy.special import erfinv
import numba

def get_interactions(M=5):
    """ Create a symmetric interaction matrix """
    I = np.zeros((M, M))

    # Put infinity in the diagonal
    # np.fill_diagonal(I, np.inf)
    np.fill_diagonal(I, 100)

    # Number if unique elements in the upper triangle of a symmetric matrix (minus diagonal)
    N_M = M * (M + 1) // 2 - M

    # Create values from standard normal distribution
    k = np.arange(1, N_M + 1)
    x_k = (2 * k - 1) / N_M - 1
    y_k = np.sqrt(2) * erfinv(x_k)
    y_k_shuffled = y_k.copy()
    np.random.shuffle(y_k_shuffled)

    # Fill elements into matrix
    upper_indices = np.triu_indices(M, k=1)
    I[upper_indices] = y_k_shuffled
    I = I + I.T  # Make the matrix symmetric

    return I

def get_types(M, N_m):
    """ Initialize the types array with N particles """
    types = np.repeat(np.arange(M), N_m)
    np.random.shuffle(types)
    return types

def get_neighbours(L, D):
    """ Precomputed neighbors indices for each particle (to speed up computations) """
    N = L ** D
    neighbors = np.zeros((N, 2 * D), dtype=np.int32)
    for n in range(N):
        pos = np.zeros(D, dtype=np.int32)
        temp = n
        for k in range(D - 1, -1, -1):
            pos[k] = temp % L
            temp //= L
        neighbor_idx = 0
        for k in range(D):
            for direction in [-1, 1]:
                neighbor_pos = pos.copy()
                neighbor_pos[k] = (neighbor_pos[k] + direction) % L
                neighbor_n = 0
                for dim in neighbor_pos:
                    neighbor_n = neighbor_n * L + dim
                neighbors[n, neighbor_idx] = neighbor_n
                neighbor_idx += 1
    return neighbors


# Function to calculate the total energy of the system
@numba.njit
def total_energy(types, I, neighbors):
    E = 0.0
    N = types.size
    for n in range(N):
        type_n = types[n]
        for neighbor in neighbors[n]:
            if neighbor > n:  # Avoid double counting
                type_neighbor = types[neighbor]
                E += I[type_n, type_neighbor]
    return E


@numba.njit
def _monte_carlo_simulation(I, types, neighbors, beta, num_steps, equilibration_steps, print_stride):
    """ Perform the Monte Carlo simulation with global particle swops """
    N = types.size
    E = total_energy(types, I, neighbors)
    accepted_moves = 0
    attempted_moves = 0

    num_records = (num_steps - equilibration_steps) // print_stride
    energies = np.zeros(num_records)
    record_index = 0

    for step in range(1, num_steps + 1):
        # Choose two random particles to swap
        n1 = np.random.randint(0, N)
        n2 = np.random.randint(0, N)
        while n2 == n1:
            n2 = np.random.randint(0, N)

        type1 = types[n1]
        type2 = types[n2]

        # Calculate energy change due to swapping
        dE = 0.0
        for neighbor in neighbors[n1]:
            if neighbor != n2:
                type_neighbor = types[neighbor]
                dE -= I[type1, type_neighbor]
                dE += I[type2, type_neighbor]
        for neighbor in neighbors[n2]:
            if neighbor != n1:
                type_neighbor = types[neighbor]
                dE -= I[type2, type_neighbor]
                dE += I[type1, type_neighbor]

        attempted_moves += 1

        # Metropolis criterion
        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            # Accept the swap
            types[n1], types[n2] = types[n2], types[n1]
            E += dE
            accepted_moves += 1

        # Record energy every PRINT_STRIDE steps after equilibration
        if step >= equilibration_steps and (step - equilibration_steps) % print_stride == 0:
            if record_index < num_records:
                energies[record_index] = E
                record_index += 1

    return energies, float(accepted_moves / attempted_moves)

def monte_carlo_simulation(L, M, D, beta, num_steps, equilibration_steps, print_stride):
    I = get_interactions(M)
    N = L ** D
    N_m = N // M
    if N_m * M != N:
        raise ValueError("N_m does not divide N evenly. Please adjust N_m or L.")
    types = get_types(M, N_m)
    neighbors = get_neighbours(L, D)
    return _monte_carlo_simulation(I, types, neighbors, beta, num_steps, equilibration_steps, print_stride)

def main():
    tic = time.perf_counter()
    monte_carlo_simulation(16, 2, 2, 2, 64, 16, 4)
    toc = time.perf_counter()
    print(f"Time elapsed: {toc - tic:0.4f} seconds for JIT compilation")
    tic = time.perf_counter()
    print(f"""{monte_carlo_simulation(
        L=128,
        M=128,
        D=2,
        beta=2,
        num_steps=128_000,
        equilibration_steps=64_000,
        print_stride=8_000) = 
    }""")
    toc = time.perf_counter()
    print(f"Time elapsed: {toc - tic:0.4f} seconds to run simulation")

if __name__ == "__main__":
    main()
