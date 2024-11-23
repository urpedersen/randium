import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfinv
import numba

np.set_printoptions(precision=3)


def get_interaction_matrix(M=5):
    """ Create a symmetric interaction matrix """
    I = np.zeros((M, M))

    # Put infinity in the diagonal
    np.fill_diagonal(I, np.inf)

    # Number if unique elements in the upper triangle of a symmetric matrix (minus diagonal)
    N_M = M * (M + 1) // 2 - M
    print(f'{N_M=}')

    k = np.arange(1, N_M + 1)
    x_k = (2 * k - 1) / N_M - 1
    y_k = np.sqrt(2) * erfinv(x_k)
    y_k_shuffled = y_k.copy()
    np.random.shuffle(y_k_shuffled)

    # Fill elements into matrx
    upper_indices = np.triu_indices(M, k=1)
    I[upper_indices] = y_k_shuffled
    I = I + I.T  # Make the matrix symmetric
    return I

@numba.njit
def weighted_means_of_columns(I, beta=2):
    """ Weighted means of the columns, ignoring diagonal elements."""
    M, _ = I.shape
    means = np.zeros(M)
    Zs = np.zeros(M)
    for u in range(M):
        for v in range(M):
            if u == v:
                ...
            else:
                Zs[u] += np.exp(-beta * I[u, v])
                means[u] += np.exp(-beta * I[u, v]) * I[u, v]
        means[u] /= Zs[u]
    return means

def plot_matrix(I, title='Interaction matrix'):
    plt.matshow(I, cmap='Greens_r', vmin=-4, vmax=0)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_means(I, betas):
    data = []
    for beta in betas:
        mu = weighted_means_of_columns(I, beta)
        data.append(mu)
    data = np.array(data)
    return data

def get_difference_meassure(I, betas):
    data = get_means(I, betas)
    std = data.std(axis=1)
    diff_idx = np.trapz(std, betas)
    return diff_idx

@numba.njit
def swop_two_elements(I, u_0, v_0, u_1, v_1):
    # Check that matrix is symmetric
    assert I[u_0, v_0] == I[v_0, u_0]
    assert I[u_1, v_1] == I[v_1, u_1]
    if u_0 == v_0 or u_1 == v_1:
        return I
    elm_0 = I[u_0, v_0]
    elm_1 = I[u_1, v_1]
    I[u_0, v_0] = elm_1
    I[v_0, u_0] = elm_1
    I[u_1, v_1] = elm_0
    I[v_1, u_1] = elm_0
    return I

def make_swoops(I, betas, n_swoops=100, verbose=False, prints=32):
    print_stride = n_swoops // prints
    M, _ = I.shape
    As = []
    for idx in range(n_swoops):
        u_0, v_0 = np.random.randint(M), np.random.randint(M)
        u_1, v_1 = np.random.randint(M), np.random.randint(M)
        A_old = get_difference_meassure(I, betas)
        if idx%print_stride == 0:
            print(f'{idx}: A={A_old:.4f}')
        As.append(A_old)
        I = swop_two_elements(I, u_0, v_0, u_1, v_1)
        A_new = get_difference_meassure(I, betas)
        if A_new < A_old:
            if verbose:
                print(f'{idx}: {A_old=:.2f} -> {A_new=:.2f}')
        else:  # Undo the swap
            I = swop_two_elements(I, u_0, v_0, u_1, v_1)
    return I, As

def plot_means(I, betas, title=''):
    data = get_means(I, betas)
    means = data.mean(axis=1)
    std = data.std(axis=1)
    plt.figure(figsize=(6, 8))
    plt.subplot(211)
    plt.title(title)
    plt.plot(betas, data, '-', color='gray', lw=1)
    plt.plot(betas, means, '--', color='blue', lw=3, label=r'$\langle \mu_{v\beta}\rangle_v$')
    plt.plot(betas, means + std, '--', color='blue', lw=2, label=r'$\langle \mu_{v\beta}\rangle_v \pm \sigma_{v\beta}$')
    plt.plot(betas, means - std, '--', color='blue', lw=2)
    plt.ylim(-3.5, 1.0)
    plt.legend()
    plt.xlabel(r'$\beta$')
    plt.xlim(min(betas), max(betas))
    plt.ylabel(r'$\mu_{v\beta}$')
    plt.subplot(212)
    plt.subplots_adjust(hspace=0)
    plt.plot(betas, std, '--', color='red', lw=2, label=r'$\sigma_{v\beta}$')
    plt.text(0.2, 0.5, f'A={get_difference_meassure(I, betas):.2f}', fontsize=16)
    plt.ylabel(r'Integrand: $\sigma_{v\beta}$')
    plt.xlabel(r'$\beta$')
    plt.xlim(min(betas), max(betas))
    plt.ylim(0, 1)
    plt.savefig(f'means_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    M = 256
    print(f'{M=}')
    I = get_interaction_matrix(M)
    plot_matrix(I, 'Interaction matrix (before)')
    betas = np.linspace(0, 2, 16)
    plot_means(I, betas=betas, title='before')
    print(f'A(before)={get_difference_meassure(I, betas):.2f}')
    tic = time.perf_counter()
    I, As = make_swoops(I, betas=betas, n_swoops=32*512)
    toc = time.perf_counter()
    print(f'Elapsed time: {toc-tic:.4f} seconds')
    print(f'A(after)={get_difference_meassure(I, betas):.2f}')
    plot_means(I, betas=betas, title='after')
    plot_matrix(I, 'Interaction matrix (after)')

    # Plot change of difference measure
    plt.figure(figsize=(5, 5))
    plt.plot(As, 'o')
    plt.ylabel('Difference measure')
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.xlabel('# swoops')
    plt.savefig('difference_measure.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save matrix to disk as numpy array and csv
    np.save('interaction_matrix.npy', I)
    np.savetxt('interaction_matrix.csv', I, delimiter=',')

if __name__ == '__main__':
    main()