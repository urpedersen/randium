import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
np.set_printoptions(precision=3)

# Define the size of the matrix
M = 4

N_M = M * (M + 1) // 2 - M
print(f'Number of unique elements in the upper triangle of a symmetric matrix (minus diagonal) of size {M}x{M}: {N_M}')

k = np.arange(1, N_M + 1)
x_k = (2 * k - 1) / N_M - 1
y_k = np.sqrt(2) * erfinv(x_k)
# Print first three and last three values ofk,  x_k and y_k
print(f'k: {k[:3]} ... {k[-3:]}')
print(f'x_k: {x_k[:3]} ... {x_k[-3:]}')
print(f'y_k: {y_k[:3]} ... {y_k[-3:]}')

# Add the y_k elements randomly to the upper triangle of the matrix but not in diagonal
I = np.zeros((M, M))
upper_indices = np.triu_indices(M, k=1)
np.random.shuffle(y_k)
I[upper_indices] = y_k
I = I + I.T  # Make the matrix symmetric
np.fill_diagonal(I, np.inf)  # Put infinity in the diagonal

# Print matrix with two significant digits
print('I:')
print(I)

def weighted_means_of_columns(I, beta=1):
    """Compute the Boltzmann weighted means of the columns of I, ignoring diagonal elements.
    The weight is given by the Boltzmann factor exp(-beta * I_uv) """
    M = I.shape[0]
    # Use vectorized computation to compute the weighted means
    I_no_diag = I.copy()
    np.fill_diagonal(I_no_diag, 0)
    weights = np.exp(-beta * I_no_diag)
    mu = (weights @ I) / weights.sum(axis=0)
    return mu


# Compute the weighted means of the columns of I
mu = weighted_means_of_columns(I, beta=1)
print(f'Weighted means of the columns of I: {mu}')