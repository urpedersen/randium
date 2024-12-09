import numpy as np
from scipy.special import erfinv

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
