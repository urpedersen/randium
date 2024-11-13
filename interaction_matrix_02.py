import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

# Define the size of the matrix
M = 50
N_M = M * (M + 1) // 2  # Number of unique elements in a symmetric matrix

# Generate x_k values evenly distributed in (-1, 1)
k = np.arange(1, N_M + 1)
x_k = (2 * k - 1) / N_M - 1  # x_k in (-1,1)

# Generate y_k values using the inverse error function
y_k = np.sqrt(2) * erfinv(x_k)

# Shuffle y_k values
np.random.shuffle(y_k)

# Initialize the interaction matrix I
I = np.zeros((M, M))

# Get the indices for the upper triangle including the diagonal
upper_indices = np.triu_indices(M)

# Assign y_k values to the upper triangle of I
I[upper_indices] = y_k

# Make the matrix symmetric
I = I + I.T - np.diag(I.diagonal())


def compute_D(I):
    """Compute D as per the given formula."""
    M = I.shape[0]

    # Compute the means and variances of the rows
    row_means = I.mean(axis=1)
    row_vars = I.var(axis=1)

    # Compute the mean and variance of the diagonal elements
    diag_elements = np.diag(I)
    diag_mean = diag_elements.mean()
    diag_var = diag_elements.var()

    # Compute the averages of row means and variances
    mean_row_mean = np.mean(np.abs(row_means))
    mean_row_var = np.mean(np.abs(row_vars - 1))

    # Compute D
    D = mean_row_mean + (mean_row_var-1) + np.abs(diag_mean) + np.abs(diag_var)
    return D, mean_row_mean, mean_row_var, diag_mean, diag_var


# Compute the initial D
current_D, mean_row_mean, mean_row_var, diag_mean, diag_var = compute_D(I)

# Create a list of upper triangle indices
upper_indices_list = list(zip(upper_indices[0], upper_indices[1]))
num_elements = len(upper_indices_list)

# Perform swapping to minimize D
matrix_changed = True
iteration = 0
max_iterations = 100  # Prevent infinite loops

while matrix_changed and iteration < max_iterations:
    matrix_changed = False
    iteration += 1
    num_swaps = N_M  # Number of swap attempts per iteration
    swaps_made = 0
    for _ in range(num_swaps):
        # Randomly select two indices from the upper triangle
        idx1, idx2 = np.random.choice(num_elements, size=2, replace=False)
        u1, v1 = upper_indices_list[idx1]
        u2, v2 = upper_indices_list[idx2]

        # Save the original values to restore if needed
        orig_u1_v1 = I[u1, v1]
        orig_v1_u1 = I[v1, u1]
        orig_u2_v2 = I[u2, v2]
        orig_v2_u2 = I[v2, u2]

        # Swap elements I[u1, v1] and I[u2, v2]
        I[u1, v1] = orig_u2_v2
        I[u2, v2] = orig_u1_v1
        # Ensure the matrix remains symmetric
        I[v1, u1] = I[u1, v1]
        I[v2, u2] = I[u2, v2]

        # Recalculate D
        new_D, _, _, _, _ = compute_D(I)

        if new_D < current_D:
            # Accept the swap
            current_D = new_D
            matrix_changed = True
            swaps_made += 1
        else:
            # Revert the swap
            I[u1, v1] = orig_u1_v1
            I[u2, v2] = orig_u2_v2
            # Restore symmetry
            I[v1, u1] = orig_v1_u1
            I[v2, u2] = orig_v2_u2
    print(f"Iteration {iteration}, D: {current_D:.6f}, Swaps made: {swaps_made}")
    if swaps_made == 0:
        break

# Final D computation
final_D, mean_row_mean, mean_row_var, diag_mean, diag_var = compute_D(I)

# Print final statistics
print("\nFinal Statistics:")
print(f"Mean of row means (absolute): {mean_row_mean:.6f}")
print(f"Mean of row variances deviation from 1 (absolute): {mean_row_var:.6f}")
print(f"Mean of diagonal elements: {diag_mean:.6f}")
print(f"Variance of diagonal elements: {diag_var:.6f}")
print(f"Final D: {final_D:.6f}")

# Confirm that the entire matrix has the expected mean and variance
matrix_elements = I[np.triu_indices(M)]
overall_mean = np.mean(matrix_elements)
overall_var = np.var(matrix_elements)
print(f"\nOverall matrix mean: {overall_mean:.6f}")
print(f"Overall matrix variance: {overall_var:.6f}")

# Plot the means and variances of the rows
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Plot row means
axs[0].plot(np.arange(M), I.mean(axis=1), marker='o')
axs[0].axhline(0, color='red', linestyle='--')
axs[0].set_title('Means of the Rows')
axs[0].set_xlabel('Row Index')
axs[0].set_ylabel('Mean')

# Plot row variances
axs[1].plot(np.arange(M), I.var(axis=1), marker='o', color='green')
axs[1].axhline(1, color='red', linestyle='--')
axs[1].set_title('Variances of the Rows')
axs[1].set_xlabel('Row Index')
axs[1].set_ylabel('Variance')

plt.tight_layout()
plt.show()

# Plot the mean and variance of the diagonal elements
diag_elements = np.diag(I)
diag_mean = np.mean(diag_elements)
diag_var = np.var(diag_elements)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(diag_elements, bins=20, color='purple')
plt.title('Diagonal Elements Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.axvline(diag_mean, color='red', linestyle='--', label=f'Mean: {diag_mean:.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist((diag_elements - diag_mean) ** 2, bins=20, color='orange')
plt.title('Diagonal Elements Variance')
plt.xlabel('Squared Deviation')
plt.ylabel('Frequency')
plt.axvline(diag_var, color='red', linestyle='--', label=f'Variance: {diag_var:.2f}')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the interaction matrix
plt.imshow(I, cmap='viridis', interpolation='none')
plt.colorbar()
plt.title('Interaction Matrix I')
plt.xlabel('Index v')
plt.ylabel('Index u')
plt.show()


# Make histogram of elements in the matrix
plt.figure(figsize=(8, 4))
plt.hist(matrix_elements, bins=32, color='blue', alpha=0.7, density=True, label='Histogram of matrix elements')
# Standard normal distribution for comparison
x = np.linspace(-3, 3, 1000)
plt.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color='red', label='Standard Normal Distribution')
plt.yscale('log')
plt.title(f'Histogram of y_k values (after), $M={M}$, $N_M={N_M}$')
plt.xlabel('Value')
plt.show()