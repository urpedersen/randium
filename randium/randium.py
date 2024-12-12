import numpy as np
import numba
from copy import deepcopy

def get_types(M, N_m):
    """ Initialize particle types. """
    types = np.repeat(np.arange(M), N_m)
    np.random.shuffle(types)
    return types

def get_neighbours(L, D):
    """ Neighbors indices for each particle for a simple square lattice in D-dimentions """
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

def get_interactions(M):
    from scipy.special import erfinv
    I = np.zeros((M, M))
    np.fill_diagonal(I, np.inf)
    N_M = M * (M + 1) // 2 - M
    k = np.arange(1, N_M + 1)
    x_k = (2 * k - 1) / N_M - 1
    y_k = np.sqrt(2) * erfinv(x_k)
    y_k_shuffled = y_k.copy()
    np.random.shuffle(y_k_shuffled)

    idx = 0
    for i in range(M):
        for j in range(i+1, M):
            val = y_k_shuffled[idx]
            I[i, j] = val
            I[j, i] = val
            idx += 1
    return I

@numba.njit
def get_total_energy(types, I, neighbors):
    E = 0.0
    N = types.size
    for n in range(N):
        t_n = types[n]
        for neigh in neighbors[n]:
            if neigh > n:
                E += I[t_n, types[neigh]]
    return E

@numba.njit
def simulation_monte_carlo_global(I, types, neighbors, beta, num_steps, print_stride):
    """ Monte Carlo simulation with global particle swaps. """
    N = types.size
    E = get_total_energy(types, I, neighbors)
    accepted_moves = 0
    attempted_moves = 0

    num_records = num_steps // print_stride
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
            type_neighbor = types[neighbor]
            dE -= I[type1, type_neighbor]
            dE += I[type2, type_neighbor]
        for neighbor in neighbors[n2]:
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
        if step% print_stride == 0:
            if record_index < num_records:
                energies[record_index] = E
                record_index += 1

    return energies, float(accepted_moves/attempted_moves)

@numba.njit
def simulation_monte_carlo_local(I, types, neighbors, beta, num_steps, print_stride):
    """ Monte Carlo simulation with particle swaps of neighbor particles """
    num_records = num_steps // print_stride
    energies = np.zeros(num_records)
    record_index = 0

    N = types.size
    E = get_total_energy(types, I, neighbors)
    accepted_moves = 0
    attempted_moves = 0

    for step in range(1, num_steps + 1):
        # Choose a random particle
        n = np.random.randint(0, N)

        # Choose a random neighbor of the particle
        neighbors_n = neighbors[n]
        neighbor_idx = np.random.randint(0, len(neighbors_n))
        n_neigh = neighbors_n[neighbor_idx]

        type1 = types[n]
        type2 = types[n_neigh]

        # Calculate energy change due to swapping
        dE = 0.0
        for neighbor in neighbors[n]:
            type_neighbor = types[neighbor]
            if neighbor != n_neigh:
                dE -= I[type1, type_neighbor]
                dE += I[type2, type_neighbor]
        for neighbor in neighbors[n_neigh]:
            type_neighbor = types[neighbor]
            if neighbor != n:
                dE -= I[type2, type_neighbor]
                dE += I[type1, type_neighbor]

        attempted_moves += 1

        # Metropolis criterion
        if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
            # Accept the swap
            types[n], types[n_neigh] = types[n_neigh], types[n]
            E += dE
            accepted_moves += 1

        # Record energy every PRINT_STRIDE steps after equilibration
        if step % print_stride == 0:
            if record_index < num_records:
                energies[record_index] = E
                record_index += 1

    return energies, float(accepted_moves / attempted_moves)

class Lattice:
    """ Lattice class. """
    def __init__(self, L: int, M: int, D: int, beta: float):
        self.L: int = L
        self.M: int = M
        self.D: int = D
        self.beta: float = beta
        
        self.N: int = self.L ** self.D
        self.N_m: int = self.N // self.M
        if self.N_m * self.M != self.N:
            raise ValueError("f{self.N_m * self.M=} != {self.N=}")
        self.types = get_types(M, L ** D // M)
        self.neighbors = get_neighbours(L, D)
        self.I = get_interactions(M)

    def copy(self):
        return deepcopy(self)

    def get_types_on_lattice(self):
        """ Return array with types on the lattice, so the lattice configuration can be printed. """
        if self.D != 2:
            raise ValueError("Only 2D lattices are supported.")
        array = np.zeros((self.L, self.L), dtype=np.uint32)
        for n in range(self.N):
            x = n // self.L
            y = n % self.L
            array[x, y] = self.types[n]
        return array

    def get_total_energy(self):
        return get_total_energy(self.types, self.I, self.neighbors)

    def simulation_monte_carlo_global(self, steps, print_stride=None):
        if print_stride is None:
            print_stride = steps
        return simulation_monte_carlo_global(self.I, self.types, self.neighbors, self.beta, steps, print_stride)

    def simulation_monte_carlo_local(self, steps, print_stride=None):
        if print_stride is None:
            print_stride = steps
        return simulation_monte_carlo_local(self.I, self.types, self.neighbors, self.beta, steps, print_stride)

    def to_png(self, filename='image.png'):
        import matplotlib.pyplot as plt
        arr = self.get_types_on_lattice()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr, cmap='hsv')
        ax.set_title(f'M={self.M}, N_M={self.N_m}, beta={self.beta:.2f}')
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def overlap(self, other):
        """ Return overlap parameter between self and other lattice. """
        if self.D != other.D or self.N != other.N or self.M != other.M:
            raise ValueError('Error: Trying to compute overlap of lattices that are not the same kind')
        overlap_counter = 0
        overlaps = np.zeros(self.N, dtype=np.bool)
        for n in range(self.N):
            if self.types[n] == other.types[n]:
                overlap_counter += 1
                overlaps[n] = True
        overlap_counter = overlap_counter / self.N
        Q_max = 1.0
        Q_min = 1/self.M
        return (overlap_counter - Q_min) / (Q_max - Q_min), overlaps

def default_lattice():
    L = 16  # 32
    M = 64  # 128
    D = 2
    beta = 2.0
    lat = Lattice(L, M, D, beta)
    return lat

def main():
    import matplotlib.pyplot as plt
    import time
    # lat = default_lattice()
    lat = Lattice(
        L=16,
        M=32,
        D=2,
        beta=1.4)
    print(f'{lat.D=} {lat.L=} {lat.N=} {lat.M=} {lat.N_m=} ')
    lat.simulation_monte_carlo_global(steps=4*lat.N, print_stride=4*lat.N)  # Equilibration
    print_stride = 4*lat.N
    steps = print_stride*1024
    plt.figure()
    plt.title(f'{lat.L}x{lat.L} lattice (N={lat.N}), M={lat.M} (N_M={lat.N_m}), beta={lat.beta:.2f}')
    for sim_idx in range(8):
        tic = time.perf_counter()
        energies, acceptance_rate = lat.simulation_monte_carlo_global(steps=steps, print_stride=print_stride)
        toc = time.perf_counter()
        # Plot energy trajectory
        t = np.linspace(0, steps, steps//print_stride)/lat.N
        plt.plot(t, energies, label=f'Time-block: {sim_idx}')
        print(lat.get_types_on_lattice())
        print(f'Total energy: {lat.get_total_energy()}')
        print(f'Time elapsed: {toc - tic:0.4f} seconds, acceptance rate: {acceptance_rate:.4f}')
    plt.legend()
    plt.show()

    # Test that local moves work
    lat_ref = lat.copy()
    overlap_timeseries = [lat.overlap(lat_ref)[0]]
    plt.figure()
    plt.title(f'{lat.L}x{lat.L} lattice (N={lat.N}), M={lat.M} (N_M={lat.N_m}), beta={lat.beta:.2f}')
    for sim_idx in range(16):
        tic = time.perf_counter()
        energies, acceptance_rate = lat.simulation_monte_carlo_local(steps=steps, print_stride=print_stride)
        overlap_timeseries.append(lat.overlap(lat_ref)[0])
        toc = time.perf_counter()
        # Plot energy trajectory
        t = np.linspace(0, steps, steps//print_stride)/lat.N
        plt.plot(t, energies, label=f'Time-block: {sim_idx}')
        print(lat.get_types_on_lattice())
        print(f'Total energy: {lat.get_total_energy()}')
        print(f'Time elapsed: {toc - tic:0.4f} seconds, acceptance rate: {acceptance_rate:.4f}')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('overlaps')
    plt.plot(overlap_timeseries, '--o')
    plt.plot([0, len(overlap_timeseries)], [0,0],'k--')
    plt.ylim(-0.2, 1)
    plt.xlim(0, None)
    plt.show()

    plt.figure()
    overlaps = lat.overlap(lat_ref)[1]
    # Convert from Nx1 to a LxL array
    overlaps.resize((lat.L, lat.L))
    # make image where true is black and red is false
    plt.imshow(overlaps, cmap='gray')
    plt.show()

    print(overlap_timeseries)

if __name__ == "__main__":
    main()
