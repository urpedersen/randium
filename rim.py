from cProfile import label

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

def get_interactions(M):
    """ Initialize the interaction matrix I[M][M] with elements drawn from a standard normal distribution """
    np.random.seed(int(time.time()))
    I = np.random.normal(0, 1, size=(M, M))
    I = (I + I.T) / 2  # Make the interaction matrix symmetric
    return I


def get_types(M, N_m):
    """ Initialize the types array with N particles """
    types = np.repeat(np.arange(M), N_m)
    np.random.shuffle(types)
    return types

@njit
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
@njit
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


# Function to perform the Monte Carlo simulation
@njit
def monte_carlo_simulation(I, types, neighbors, beta, num_steps, equilibration_steps, print_stride):
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

    return energies, float(accepted_moves/attempted_moves)


def run_betas(plot=True):
    # Define model parameters
    L = 16  # Lattice size in each dimension
    d = 2  # Dimensionality of the lattice
    N_m = 16  # Number of particles of each type
    num_steps = 16_000_000  # Total number of Monte Carlo steps
    equilibration_steps = 8_000_000  # Number of equilibration steps
    energy_stride = 64  # Store energy every energy_stride steps

    N = L ** d  # Total number of particles

    # Ensure that N is divisible by N_m
    if N % N_m != 0:
        raise ValueError("N_m does not divide N evenly. Please adjust N_m or L.")

    M = N // N_m  # Number of types

    # Range of beta values from 0 to 2 in steps of 0.2
    num_replicas = 8*32
    betas = np.arange(0.0, 1.6, 0.2)
    average_energies = []
    sig_energies = []

    # Run simulations for each beta value
    neighbors = get_neighbours(L, d)
    for beta in betas:
        print(f"Running simulation for beta = {beta:.2f}")

        # Reset arras for each beta
        replica_energies = []
        for replica in range(num_replicas):
            I = get_interactions(M)
            types = get_types(M, N_m)

            # Run the simulation
            energies, acc_ratio = monte_carlo_simulation(I, types, neighbors, beta, num_steps, equilibration_steps, energy_stride)
            this_avg_energy = np.mean(energies)
            replica_energies.append(this_avg_energy)
            print(f"  Average Energy = {this_avg_energy}  (Acceptance ratio = {acc_ratio})")

        average_energies.append(np.mean(replica_energies))
        sig_energies.append(np.var(replica_energies)**0.5)

    if plot:
        # Plot energies of lowers temperature
        times = np.arange(0, len(energies) * energy_stride, energy_stride)/N
        plt.figure(figsize=(8, 6))
        plt.plot(energies)
        plt.title(f'RIM: {d=}, {L=}, {N=}, {M=} {beta=:0.2f}')
        plt.xlabel('Measurement number')
        plt.ylabel('Total Energy, $U$')
        plt.grid(True)
        plt.savefig(f'./figures/rim_enr_{d}_{L}_{N}_{N_m}_{beta:0.2f}.pdf', dpi=300, bbox_inches='tight')
        plt.show()


    # Plot average energy versus beta
    if plot:
        plt.figure(figsize=(8, 6))
        plt.errorbar(betas, average_energies, sig_energies, marker='o', capsize=5)
        beta_max = max(betas)
        plt.plot([0, beta_max], [0, -1 * N * beta_max], 'r--', label='High temperature approx.')
        plt.xlabel(r'Inverse temperature, $\beta$')
        plt.ylabel(r'Average Total Energy, $U$')
        plt.title(f'RIM: {d=}, {L=}, {N=}, {M=} {N_m=} {num_replicas=}')
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.savefig(f'./figures/rim_{d}_{L}_{N}_{N_m}.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    return betas, average_energies

if __name__ == "__main__":
    run_betas()
