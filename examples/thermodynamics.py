import numpy as np
import matplotlib.pyplot as plt
import randium as rd

def run_betas(plot=True):
    # Define model parameters
    L = 8  # Lattice size in each dimension
    D = 2  # Dimensionality of the lattice
    N_m = 8  # Number of particles of each type
    num_steps = 16_000_000  # Total number of Monte Carlo steps
    equilibration_steps = 8_000_000  # Number of equilibration steps
    energy_stride = 64  # Store energy every energy_stride steps

    N = L ** D  # Total number of particles
    print(f'{N=}')

    # Ensure that N is divisible by N_m
    if N % N_m != 0:
        raise ValueError("N_m does not divide N evenly. Please adjust N_m or L.")

    M = N // N_m  # Number of types
    print(f'{M=}')

    # Range of beta values from 0 to 2 in steps of 0.2
    num_replicas = 8
    betas = np.arange(0.2, 1.6, 0.2)
    average_energies = []
    sig_energies = []

    # Run simulations for each beta value
    #neighbors = rd.get_neighbours(L, D)
    for beta in betas:
        print(f"Running simulation for beta = {beta:.2f}")

        # Reset arras for each beta
        replica_energies = []
        for replica in range(num_replicas):
            # Run the simulation
            energies, acc_ratio = rd.monte_carlo_simulation(L, M, D, beta, num_steps, equilibration_steps, energy_stride)
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
        plt.title(f'RIM: {D=}, {L=}, {N=}, {M=} {beta=:0.2f}')
        plt.xlabel('Measurement number')
        plt.ylabel('Total Energy, $U$')
        plt.grid(True)
        plt.savefig(f'./figures/rim_enr_{D}_{L}_{N}_{N_m}_{beta:0.2f}.pdf', dpi=300, bbox_inches='tight')
        plt.show()


    # Plot average energy versus beta
    if plot:
        plt.figure(figsize=(8, 6))
        plt.errorbar(betas, average_energies, sig_energies, marker='o', capsize=5)
        beta_max = max(betas)
        plt.plot([0, beta_max], [0, -1 * N * beta_max], 'r--', label='High temperature approx.')
        plt.xlabel(r'Inverse temperature, $\beta$')
        plt.ylabel(r'Average Total Energy, $U$')
        plt.title(f'RIM: {D=}, {L=}, {N=}, {M=} {N_m=} {num_replicas=}')
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.savefig(f'./figures/rim_{D}_{L}_{N}_{N_m}.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    return betas, average_energies

if __name__ == "__main__":
    run_betas()
