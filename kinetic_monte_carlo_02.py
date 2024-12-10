import time
import numpy as np
from scipy.special import erfinv
import numba


# -------------------------
# Numba-jitted helper functions
# -------------------------

def fill_interactions(M):
    I = np.zeros((M, M))
    np.fill_diagonal(I, 100)
    N_M = M * (M + 1) // 2 - M
    k = np.arange(1, N_M + 1)
    x_k = (2 * k - 1) / N_M - 1
    y_k = np.sqrt(2) * erfinv(x_k)
    y_k_shuffled = y_k.copy()
    np.random.shuffle(y_k_shuffled)

    idx = 0
    for i in range(M):
        for j in range(i + 1, M):
            val = y_k_shuffled[idx]
            I[i, j] = val
            I[j, i] = val
            idx += 1
    return I


@numba.njit
def compute_total_energy(types, I, neighbors):
    E = 0.0
    N = types.size
    for n in range(N):
        t_n = types[n]
        for neigh in neighbors[n]:
            if neigh > n:
                E += I[t_n, types[neigh]]
    return E


@numba.njit
def event_energy_change(n1, n2, types, I, neighbors):
    t1, t2 = types[n1], types[n2]

    # Before swap
    E_n1_before = 0.0
    for neigh in neighbors[n1]:
        E_n1_before += I[t1, types[neigh]]
    E_n2_before = 0.0
    for neigh in neighbors[n2]:
        E_n2_before += I[t2, types[neigh]]

    # After swap
    types[n1], types[n2] = t2, t1
    E_n1_after = 0.0
    for neigh in neighbors[n1]:
        E_n1_after += I[types[n1], types[neigh]]
    E_n2_after = 0.0
    for neigh in neighbors[n2]:
        E_n2_after += I[types[n2], types[neigh]]
    # Swap back
    types[n1], types[n2] = t1, t2

    dE = (E_n1_after + E_n2_after) - (E_n1_before + E_n2_before)
    return dE


@numba.njit
def compute_event_rate(dE, beta):
    if dE <= 0:
        return 1.0
    else:
        return np.exp(-beta * dE)


@numba.njit
def pick_event(cumulative):
    R_total = cumulative[-1]
    if R_total == 0.0:
        return -1, R_total
    r = np.random.rand() * R_total
    left = 0
    right = cumulative.size - 1
    while left < right:
        mid = (left + right) // 2
        if cumulative[mid] > r:
            right = mid
        else:
            left = mid + 1
    return left, R_total


@numba.njit
def update_event_rates_after_swap(n1, n2,
                                  types, I, neighbors, beta,
                                  event_array, rates, cumulative,
                                  site_events_arr, counts):
    # Identify affected sites
    affected_sites = set([n1, n2])
    for neigh in neighbors[n1]:
        affected_sites.add(neigh)
    for neigh in neighbors[n2]:
        affected_sites.add(neigh)

    # Recompute rates for affected events
    for s in affected_sites:
        for idx_pos in range(counts[s]):
            e_idx = site_events_arr[s, idx_pos]
            nA, nB = event_array[e_idx]
            dE = event_energy_change(nA, nB, types, I, neighbors)
            rates[e_idx] = compute_event_rate(dE, beta)

    # Rebuild cumulative
    cumulative[:] = np.cumsum(rates)


@numba.njit
def run_simulation_until_time(min_time,
                              types, I, neighbors,
                              event_array, rates, cumulative,
                              site_events_arr, counts,
                              beta, E, current_time):
    """
    Run the event-driven simulation until current_time >= min_time.
    Returns the final E, current_time, and modified 'types'.
    """

    while current_time < min_time:
        chosen_idx, R_total = pick_event(cumulative)
        if chosen_idx == -1 or R_total == 0.0:
            # No more events possible
            break
        n1, n2 = event_array[chosen_idx]
        u = np.random.rand()
        dt = -np.log(u) / R_total
        current_time += dt

        if current_time > min_time:
            # We've reached the desired simulation time
            break

        dE = event_energy_change(n1, n2, types, I, neighbors)
        # Perform the swap
        t1, t2 = types[n1], types[n2]
        types[n1], types[n2] = t2, t1
        E += dE

        # Update rates
        update_event_rates_after_swap(n1, n2, types, I, neighbors, beta,
                                      event_array, rates, cumulative,
                                      site_events_arr, counts)

    return E, current_time


# -------------------------
# Class definition
# -------------------------

class Lattice:
    def __init__(self, L, M, D, beta, seed=42):
        self.L = L
        self.M = M
        self.D = D
        self.beta = beta
        self.seed = seed
        np.random.seed(self.seed)

        self.N = L ** D
        N = self.N
        M = self.M
        if N % M != 0:
            raise ValueError("Number of particles not divisible by M.")

        self.types = self._init_types(M, N // M)
        self.I = fill_interactions(M)
        self.neighbors = self._init_neighbors(L, D)
        self.E = compute_total_energy(self.types, self.I, self.neighbors)

        # Initialize events
        self.event_array, self.rates, self.site_events_arr, self.counts = self._init_events()

        # Build cumulative rates
        self.cumulative = np.cumsum(self.rates)
        self.current_time = 0.0

    def _init_types(self, M, N_m):
        types = np.repeat(np.arange(M), N_m)
        np.random.shuffle(types)
        return types

    def _init_neighbors(self, L, D):
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

    def _init_events(self):
        # Collect unique neighbor pairs
        N = self.N
        pairs = []
        for n in range(N):
            for neigh in self.neighbors[n]:
                if neigh > n:
                    pairs.append((n, neigh))
        event_array = np.array(pairs, dtype=np.int32)

        # Compute initial rates
        rates = np.zeros(event_array.shape[0], dtype=np.float64)
        for i, (n1, n2) in enumerate(event_array):
            dE = event_energy_change(n1, n2, self.types, self.I, self.neighbors)
            rates[i] = compute_event_rate(dE, self.beta)

        # Create site to events mapping
        site_events = [[] for _ in range(self.N)]
        for i, (n1, n2) in enumerate(event_array):
            site_events[n1].append(i)
            site_events[n2].append(i)

        max_len = max(len(se) for se in site_events)
        site_events_arr = -1 * np.ones((self.N, max_len), dtype=np.int32)
        counts = np.zeros(self.N, dtype=np.int32)
        for s in range(self.N):
            for idx in site_events[s]:
                site_events_arr[s, counts[s]] = idx
                counts[s] += 1

        return event_array, rates, site_events_arr, counts

    def run(self, min_time):
        """
        Run simulation until min_time is reached.
        Updates self.types, self.E, and self.current_time in place.
        """
        self.E, self.current_time = run_simulation_until_time(
            min_time,
            self.types, self.I, self.neighbors,
            self.event_array, self.rates, self.cumulative,
            self.site_events_arr, self.counts,
            self.beta, self.E, self.current_time
        )


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    L = 16
    M = 2
    D = 2
    beta = 2.0
    min_time = 10.0  # run until simulation time reaches 10.0

    lat = Lattice(L, M, D, beta)
    start = time.perf_counter()
    lat.run(min_time)
    end = time.perf_counter()

    print(f"Simulation run until time {min_time} completed in {end - start:.4f} s")
    print("Final time:", lat.current_time)
    print("Final energy:", lat.E)
    # Print a small sample of types
    print("Sample of final types:", lat.types[:10])