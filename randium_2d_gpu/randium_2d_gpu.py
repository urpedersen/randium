import math

import numpy as np
import numba.cuda
from numba.cuda.random import create_xoroshiro128p_states

from backend import *

class Randium_2d_gpu:
    def __init__(
            self,
            threads_per_block=(8, 8),
            blocks=(16, 16),
            tiles=(8, 8),
            num_of_each_type=64,
            abc=(1664525, 1013904223, 4),
            seed=2025
    ):
        self.threads_per_block = np.uint32(threads_per_block[0]), np.uint32(threads_per_block[1])
        self.blocks = np.uint32(blocks[0]), np.uint32(blocks[1])
        self.tiles = np.uint32(tiles[0]), np.uint32(tiles[1])
        self.rows = tiles[0] * blocks[0] * threads_per_block[0]
        self.cols = tiles[1] * blocks[1] * threads_per_block[1]
        self.N = np.uint32(self.rows * self.cols)
        self.num_of_each_type = self.N_m = np.uint32(num_of_each_type)
        self.num_types = self.M = np.uint32(self.N // self.num_of_each_type)
        if self.num_of_each_type * self.num_types != self.N:
            print(f'{self.N = }, {self.num_of_each_type = }, {self.num_types = }')
            raise ValueError("value of num_of_each_type is wrong")
        self.a = np.uint32(abc[0])
        self.b = np.uint32(abc[1])
        self.c = np.uint32(abc[2])
        self.abc = self.a, self.b, self.c
        self.Mabc = self.num_types, self.a, self.b, self.c
        self.seed = seed

        # Check if interaction matrix can be computed correctly
        self.N_M = int(self.M) * (int(self.M) - 1) // 2
        if math.gcd(self.N_M, self.a) != 1:
            print(f'Warning: {self.M = }, {self.N_M = }, {self.a = }, {math.gcd(self.N_M, self.a) = } (should be 1)')
            raise ValueError("The gcd(N_M, a) is not 1.")

        # Check for possible overflow of np.int32
        test_0 = int(self.a) * (self.N_M - 1) + int(self.b)
        test_1 = test_0 % int(self.N_M)
        max_value = np.iinfo(np.uint64).max
        if test_0 > max_value:
            print(f'Warning: {np.iinfo(np.int64).max = }, {test_0 = }, {test_1 = }')
            raise OverflowError("Try to change abc to avoid overflow")
        converted_test_0 = np.uint64(test_0)  # Also raise error
        converted_test_1 = np.uint64(test_1)

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
        out += f'num_of_each_type={self.num_of_each_type}, '
        out += f'abc=({int(self.a)}, {int(self.b)}, {int(self.c)})'
        out += ')'
        return out

    def __str__(self):
        out = self.__repr__() + '\n'
        out += f'  System size: {int(self.cols)} x {int(self.rows)} = {int(self.N)}' '\n'
        out += f'  Number of types: {int(self.num_types)}' '\n'
        out += f'  Unique type pairs: {int(self.N_M)}'
        return out

    def run(self, beta=1.0, steps=1):
        start = cuda.event()
        end = cuda.event()

        start.record()
        kernel_run_simulation[self.blocks, self.threads_per_block](
            self.d_lattice,
            *self.Mabc,
            beta,
            self.tiles,
            self.rng_states,
            steps
        )
        end.record()
        end.synchronize()
        self.lattice = self.d_lattice.copy_to_host()

        wallclock_time = start.elapsed_time(end)
        self.wallclock_times.append(wallclock_time)
        self.steps.append(steps)

        return wallclock_time

    def get_benchmark(self):
        first_delta_t = self.wallclock_times[0]
        delta_t_avg = np.mean(self.wallclock_times[1:])
        mc_attempts_per_step = self.rows * self.cols * 4
        steps_avg = np.mean(self.steps[1:])
        return dict(
            first_delta_t = float(first_delta_t),
            delta_t_avg = float(delta_t_avg),
            mc_attempts_per_step = int(mc_attempts_per_step),
            steps_avg = int(steps_avg),
            mc_attempts_per_sec = float(steps_avg*mc_attempts_per_step/delta_t_avg),
        )



def main():
    randium = Randium_2d_gpu()
    print(randium)
    print(randium.lattice)
    print(f'Compile in {randium.run(1)} ms')
    steps_per_time_block = 16
    for time_block in range(16):
        wc = randium.run(beta = 1.0, steps = steps_per_time_block)
        print(f'{time_block:<4} {wc:2.2f}')
    print(randium.lattice)
    from pprint import pprint
    pprint(randium.get_benchmark())
    print(f'MC attempts per secound: {randium.get_benchmark()['mc_attempts_per_sec']:0.2e}')


if __name__ == '__main__':
    main()
