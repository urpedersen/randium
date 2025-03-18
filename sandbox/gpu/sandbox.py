import math

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import scipy

def from_xy_to_idx(i, j, L, a=433, b=0):
    if i == j:
        return -1
    i, j = min(i, j), max(i, j)
    ii = i * (L - 1)-i*(i-1)//2+(j-i-1)
    N = L*(L-1)//2
    idx_shuffle = (a * ii + b) % N
    return idx_shuffle



def inverf_series_expansion(x):
    """ Crap ...
    Found on Wikipedia: https://en.wikipedia.org/wiki/Error_function
    See also:
        https://oeis.org/A092676
        https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-13/issue-2/The-inverse-of-the-error-function/pjm/1103035736.full
    """
    a = [1/12, 7/480, 127/40320, 4369/5806080, 34807/182476800,
         20036983 / 762048187200,
         2280356863 / 802241960064000,
         49020204823 / 2662722137600000,
         65967241200001 / 578323603040000000,
         15773461423793767 / 14204940704000000000
         ]
    polynomial = 0.0
    for i in range(len(a)-1, -1, -1):
        polynomial = polynomial*x*x*math.pi+a[i]
    polynomial = polynomial*x + x
    return 0.5*math.pi**0.5*polynomial

def inverf_winitzki(x):
    """ S. Winitzki’s (2008)
    A handy approximation for the error function and its inverse
    Lecture Notes in Computer Science series, volume 2667
    https://link.springer.com/chapter/10.1007/3-540-44839-X_82
    See also: Approximations to inverse error functions, Stephen Dyer
    See also: https://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function/
    See also: https://www.scribd.com/document/82414963/Winitzki-Approximation-to-Error-Function
    """
    a = 0.147    # 0.1400122886866665
    s = math.copysign(1.0, x)
    xx = 1 - x*x
    log_xx = math.log(xx)
    t = 2/(math.pi*a) + 0.5*log_xx
    inner = t*t - (1/a)*log_xx
    return s * math.sqrt(math.sqrt(inner) - t)  # Eq. (7) in "A handy approximation ..."

def plot_inverf():
    num = 18432
    xs = [(2 * x + 1) / num - 1 for x in range(num)]
    inverf_0 = []
    inverf_1 = []
    inverf_2 = []
    for x in xs:
        # print(x, scipy.special.erfinv(x), inverf_series_expansion(x), inverf_winitzki(x))
        inverf_0.append(scipy.special.erfinv(x))
        inverf_1.append(inverf_series_expansion(x))
        inverf_2.append(inverf_winitzki(x))
    print(f'{min(xs) = }, {inverf_winitzki(min(xs)) = }, {scipy.special.erfinv(min(xs)) = }')

    plt.figure()
    plt.title(f'{num = }')
    plt.plot(xs, inverf_0, 'r-', label="scipy.special.erfinv")
    plt.plot(xs, inverf_2, 'g-', label="inverf_winitzki")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f'Comparing SciPy and winitzki')
    plt.plot(xs,
             (np.array(inverf_2)-np.array(inverf_0)),
             'r-',
             label="relative difference")
    plt.legend()
    plt.show()

def erf_f77(x):
    """ Found on Wikipedia, but from
    Press, William H. (1992).
    Numerical Recipes in Fortran 77: The Art of Scientific Computing.
    Cambridge University Press. p. 214. ISBN 0-521-43064-X. """
    t = 1/(1+0.5*abs(x))
    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806,
         0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277]
    polynomial = 0.0
    for i in range(len(a)-1, -1, -1):
        polynomial = polynomial * t + a[i]
    tau = t*math.exp(-x*x+polynomial)
    if x < 0.0:
        return tau - 1
    else:
        return 1 - tau

def plot_erf_f77():
    num = 16+1
    xs = [(2*x+1)/num-1 for x in range(num)]
    for x in xs:
        print(x, scipy.special.erf(x), erf_f77(x))

def get_pair_energy(row, col, M, a=433, b=0):
    """ N.B. gcd(N,a) = 1 to ensure bijection of idx_ordered <-> idx_shuffle """
    if row == col:
        return np.inf
    row, col = min(row, col), max(row, col)
    idx_ordered = row * (M - 1) - row * (row - 1) // 2 + (col - row - 1)  # Ordered upper‐triangular index
    N_M = M * (M - 1) // 2
    idx_shuffle = (a * idx_ordered + b)%N_M  # shuffled index
    x_k = (2*idx_shuffle + 1)/N_M - 1
    import scipy
    # rnd = scipy.special.erfinv(x_k)
    rnd = inverf_winitzki(x_k)
    return rnd

def plot_pair_energies(M=512, a=829, b=0):
    I = [get_pair_energy(i, j, M, a=a, b=b) for i, j in product(range(M),range(M))]
    I = np.array(I).reshape((M, M))
    print(np.min(I), np.max(I[I!=np.inf]))
    print(np.float32(np.min(I)), np.float32(np.max(I[I != np.inf])))
    print(I)

    plt.figure()
    plt.title(f'Interaction matrix: {M = }, {a = }, {b = }')
    plt.imshow(I, cmap='gray', vmin=-2.0, vmax=-0.0)
    plt.colorbar()
    plt.show()

def test_from_xy_to_idx():
    # Test bijection between shuffled and ordered indexes
    L = 1024
    idxs = []
    for i, j in product(range(L), range(L)):
        idx = from_xy_to_idx(i, j, L)
        idxs.append(idx)
    idxs_set = set(idxs)
    assert L*(L-1)//2 == len(idxs_set)-1, "The shuffled indices is a different set than the original indices."


def plot_idx():
    idx_M = np.array([
        from_xy_to_idx(x, y, 1024)
        for x, y
        in product(range(1024), range(1024))
    ])
    print(idx_M)

    plt.figure()
    plt.imshow(idx_M.reshape(1024, 1024), cmap='rainbow')
    plt.colorbar()
    plt.show()

def main():
    plot_inverf()
    plot_pair_energies()


if __name__ == "__main__":
    main()
