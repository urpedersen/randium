import randium_2d_gpu as rd2

import numpy as np
import matplotlib.pyplot as plt


def main():
    # randium = Randium_2d_gpu(threads_per_block=(8, 8), blocks=(16, 16), tiles=(8, 8), num_of_each_type=16384)
    rdm = rd2.Randium_2d_gpu(threads_per_block=(8, 8), blocks=(16, 16), tiles=(8, 8), num_of_each_type=1)
    #rdm = rd2.Randium_2d_gpu()
    print(rdm)
    beta = 2.0

    # Plot (part of) interaction matrix
    i_max = 1024
    I = rdm.get_interaction_matrix((0, i_max), (0, i_max))
    plt.figure()
    plt.title(f'Upper corner of {rdm.M}×{rdm.M} interaction matrix')
    plt.imshow(I)
    plt.xlabel('Type $i$')
    plt.xlabel('Type $j$')
    cbar = plt.colorbar()
    cbar.set_label('Pair energy')
    plt.show()

    indices = np.triu_indices(I.shape[0], k=1)
    U_upper = I[indices]
    plt.figure()
    plt.hist(U_upper, bins=64, density=True, alpha=0.75)
    xs = np.linspace(-5, 5, 128)
    plt.plot(xs, np.exp(-0.5 * xs * xs) / (2 * np.pi) ** 0.5)
    plt.yscale('log')
    plt.show()

    print(f'{I[0,0] = }, {I[1,1] = }, {I[0,1] = }, {I[1,0] = }, {I[1,2] = }, {I[2,1] = }')

    print(rdm.lattice)
    print(f'Compile in {rdm.run(1)} ms')
    time_blocks = 8
    steps_per_time_block = 16
    for time_block in range(time_blocks):
        wc = rdm.run(beta=beta, steps=steps_per_time_block)
        print(f'{time_block:<4} {wc:2.2f}')
    print(rdm.lattice)
    from pprint import pprint
    pprint(rdm.get_benchmark())
    print(f"MC attempts per second: {rdm.get_benchmark()['mc_attempts_per_sec']:0.2e}")


if __name__ == '__main__':
    main()
