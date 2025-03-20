
## Native Python

def lcg_python(x, y):
    # Linear Congruential Generator
    a = 1664525
    b = 1013904223
    c = 2 ** 32
    return ((a * x + b * y) % c) / c  # Normalize to [0,1]

print(f'Native python {lcg_python(27, 12) = }')


## Cuda device function

from numba import cuda
import numpy as np


@cuda.jit(device=True)
def lcg(x):
    a = 1664525
    b = 1013904223
    m = 2 ** 20
    return (a * x + b) % m

# Eksempel på brug i en CUDA kernel
@cuda.jit
def kernel(out):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if x < out.size:
        out[x] = lcg(x)

import numpy as np

n = 10
out = np.zeros(n, dtype=np.int32)
d_out = cuda.to_device(out)

threads_per_block = 16
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

kernel[blocks_per_grid, threads_per_block](d_out)
out = d_out.copy_to_host()
print(out)

