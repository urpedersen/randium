import math
from itertools import product

import numpy as np
import numba.cuda
from numba.cuda.random import create_xoroshiro128p_states

from . import backend

class Randium_2d_gpu:
    def __init__(
            self,
            threads_per_block=(8, 8),
            blocks=(16, 16),
            tiles=(8, 8),
            num_of_each_type=64,
            seed=2025
    ):
        self.threads_per_block = np.uint32(threads_per_block[0]), np.uint32(threads_per_block[1])
        self.blocks = np.uint32(blocks[0]), np.uint32(blocks[1])
        self.tiles = np.uint32(tiles[0]), np.uint32(tiles[1])
        self.rows = tiles[0] * blocks[0] * threads_per_block[0]
        self.cols = tiles[1] * blocks[1] * threads_per_block[1]
        self.N = self.rows * self.cols
        self.num_of_each_type = self.N_m = np.uint32(num_of_each_type)
        self.num_types = self.M = np.uint32(self.N // self.num_of_each_type)
        self.N_M = self.M * (self.M - 1) // 2
        # 5**2 * 139 * 479 = 1664525
        if math.gcd(1664525, self.N_M) != 1:
            print(f'{self.N_M = }, {math.gcd(1664525, self.N_M) = }')
            raise ValueError('gcd(1664525, self.N_M) should be 1.')
        if self.num_of_each_type * self.num_types != self.N:
            print(f'{self.N = }, {self.num_of_each_type = }, {self.num_types = }')
            raise ValueError("value of num_of_each_type is wrong")
        self.seed = seed

        # Setup Lattice
        self.lattice = np.array([[t] * num_of_each_type for t in range(self.num_types)], dtype=np.int32).flatten()
        np.random.shuffle(self.lattice)
        self.lattice = self.lattice.reshape((self.rows, self.cols))
        self.d_lattice = numba.cuda.to_device(self.lattice)

        # Setup random number generator
        tile_size = self.tiles[0] * self.tiles[1]
        n_threads = self.N // tile_size
        self.rng_states = create_xoroshiro128p_states(int(n_threads), seed=2025)

        # Data collected during run executions
        self.steps = []
        self.wallclock_times = []

    def __repr__(self):
        out = f'Randium(threads_per_block=({self.threads_per_block[0]}, {self.threads_per_block[1]}), '
        out += f'blocks=({self.blocks[0]}, {self.blocks[1]}), '
        out += f'tiles=({self.tiles[0]}, {self.tiles[1]}), '
        out += f'num_of_each_type={self.num_of_each_type})'
        return out

    def __str__(self):
        out = self.__repr__() + '\n'
        out += f'  System size: {int(self.cols)} x {int(self.rows)} = {int(self.N)}' '\n'
        out += f'  Number of types: {int(self.num_types)}' '\n'
        out += f'  Unique type pairs: {int(self.N_M)}'
        return out

    def run_global(self, beta=1.0, steps=1):
        self.lattice = backend.h_global_mc(self.lattice, self.M, beta, steps)

    def run_global_cache(self, beta=1.0, steps=1):
        self.lattice = backend.c_global_mc(self.lattice, self.M, beta, steps)


    def run(self, beta=1.0, steps=1):
        start = numba.cuda.event()
        end = numba.cuda.event()

        start.record()
        backend.kernel_run_simulation[self.blocks, self.threads_per_block](
            self.d_lattice,
            self.M,
            beta,
            self.tiles,
            self.rng_states,
            steps
        )
        end.record()
        end.synchronize()
        wallclock_time = start.elapsed_time(end)

        self.lattice = self.d_lattice.copy_to_host()

        self.wallclock_times.append(wallclock_time)
        self.steps.append(steps)

        return wallclock_time

    def energy(self):
        return backend.h_lattice_energy(self.lattice, self.M)/self.N

    def get_benchmark(self):
        first_delta_t = self.wallclock_times[0]
        delta_t_avg = np.mean(self.wallclock_times[1:])
        mc_attempts_per_step = self.rows * self.cols * 4
        steps_avg = np.mean(self.steps[1:])
        return dict(
            first_delta_t=float(first_delta_t),  # in ms
            delta_t_avg=float(delta_t_avg),  # in ms
            mc_attempts_per_step=int(mc_attempts_per_step),
            steps_avg=int(steps_avg),
            mc_attempts_per_sec=float(steps_avg * mc_attempts_per_step / (delta_t_avg/1000)),
        )

    def meta_info(self):
        return dict(
            rows=int(self.rows),
            cols=int(self.cols),
            N=int(self.N),
            M=int(self.M),
            N_m=int(self.N_m),
            N_M=int(self.N_M),
            threads_per_block=(int(self.threads_per_block[0]), int(self.threads_per_block[1])),
            blocks=(int(self.blocks[0]), int(self.blocks[1])),
            tiles=(int(self.tiles[0]), int(self.tiles[1])),
            energy=self.energy(),
            **self.get_benchmark(),
            lattice=[int(self.lattice[x, y]) for x, y in product(range(self.cols), range(self.rows))]
        )

    def get_shuffle_indexes(self, x_range, y_range):
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]

        I = np.array([
            backend.h_get_shuffle_idx(x, y, self.M)
            for x, y in product(range(x_range[0], x_range[1]), range(y_range[0], y_range[1]))
        ]).reshape((dx, dy))
        return I

    def get_interaction_matrix(self, x_range, y_range):
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]

        I = np.array([
            backend.h_get_pair_energy(x, y, self.M)
            for x, y in product(range(x_range[0], x_range[1]), range(y_range[0], y_range[1]))
        ]).reshape((dx, dy))
        return I
