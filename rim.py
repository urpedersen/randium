import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

# Define model parameters
L = 64                 # Lattice size in each dimension
D = 2                  # Dimensionality of the lattice
N = L ** D             # Total number of particles
N_m = 4                # Number of particles of each type
NUM_STEPS = 2_000_000     # Total number of Monte Carlo steps
EQUILIBRATION_STEPS = 2_000_000  # Number of equilibration steps
PRINT_STRIDE = 64     # Print energy every PRINT_STRIDE steps

# Ensure that N is divisible by N_m
if N % N_m != 0:
    raise ValueError("N_m does not divide N evenly. Please adjust N_m or L.")

M = N // N_m           # Number of types

# Seed the random number generator for reproducibility
np.random.seed(int(time.time()))

# Initialize the interaction matrix I[M][M] with elements drawn from a standard normal distribution
I = np.random.normal(0, 1, size=(M, M))
I = (I + I.T) / 2  # Make the interaction matrix symmetric

# Initialize the types array with N particles
types = np.repeat(np.arange(M), N_m)
np.random.shuffle(types)

# Precompute the neighbor indices for each particle to speed up computations
@njit
def compute_neighbors(L, D):
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

neighbors = compute_neighbors(L, D)

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
def monte_carlo_simulation(types, I, neighbors, beta, NUM_STEPS, EQUILIBRATION_STEPS, PRINT_STRIDE):
    N = types.size
    E = total_energy(types, I, neighbors)
    accepted_moves = 0
    attempted_moves = 0

    num_records = (NUM_STEPS - EQUILIBRATION_STEPS) // PRINT_STRIDE
    energies = np.zeros(num_records)
    record_index = 0

    for step in range(1, NUM_STEPS + 1):
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
        if step >= EQUILIBRATION_STEPS and (step - EQUILIBRATION_STEPS) % PRINT_STRIDE == 0:
            if record_index < num_records:
                energies[record_index] = E
                record_index += 1

    # Calculate average energy after equilibration
    if record_index > 0:
        average_energy = np.mean(energies[:record_index])
    else:
        average_energy = E

    return average_energy

# Range of beta values from 0 to 2 in steps of 0.2
beta_values = np.arange(0.0, 2.1, 0.2)
average_energies = []

# Run simulations for each beta value
for beta in beta_values:
    print(f"Running simulation for beta = {beta:.1f}")
    # Reset types array for each beta
    types = np.repeat(np.arange(M), N_m)
    np.random.shuffle(types)

    # Run the simulation
    avg_energy = monte_carlo_simulation(types, I, neighbors, beta, NUM_STEPS, EQUILIBRATION_STEPS, PRINT_STRIDE)
    average_energies.append(avg_energy)
    print(f"  Average Energy = {avg_energy}")

# Plot average energy versus beta
plt.figure(figsize=(8, 6))
plt.plot(beta_values, average_energies, marker='o')
beta_max=2
plt.plot([0, beta_max], [0, -1*N*beta_max], 'r--')
plt.xlabel(r'Inverse temperature, $\beta$')
plt.ylabel(r'Average Total Energy, $U$')
plt.title(f'RIM: {D=}, {L=}, {N=}, {M=}')
plt.grid(True)
plt.show()
