from itertools import product

import numpy as np
import matplotlib.pyplot as plt

def random_hash(x: int, y: int) -> int:
    # Ensure both inputs are 32-bit.
    x &= 0xFFFFFFFF
    y &= 0xFFFFFFFF

    # Symmetric mix the inputs: start by combining them with an XOR and a multiplication constant.
    h = (x + y + x * y + 0x87654321) & 0xFFFFFFFF

    # Apply a series of mixing steps
    # Inspired by MurmurHash3, https://blog.teamleadnet.com/2012/08/murmurhash3-ultra-fast-hash-algorithm.html)

    # 1st round
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= h >> 16

    # 2nd round
    h ^= h >> 15
    h = (h * 0x27d4eb2d) & 0xFFFFFFFF
    h ^= h >> 16
    h ^= h >> 15
    h = (h * 0x165667b1) & 0xFFFFFFFF
    h ^= h >> 16

    return h

for i, j in product(range(4), range(4)):
    print(f'({i}, {j}): {random_hash(i, j) = }, {random_hash(i, j)/2**32 = }')

N = 512
M = N*(N-1)//2
I = [random_hash(x, y) for x, y in product(range(N), range(N))]

plt.figure()
plt.imshow(np.array(I).reshape(N, N))
plt.show()

for i, j in product(range(4), range(4)):
    print(f'({i}, {j}): {i+j = },  {j*i = }, {i+j+j*i = } ')
for i, j in product(range(4), range(4)):
    print(f'({i}, {j}): {i^j = },  {j^i = } {(i ^ (j * 0x85ebca6b)) + (i ^ (j * 0x85ebca6b)) = }')
for i, j in product(range(4), range(4)):
    a = (227*997*i+427965865) ^ (887*409*j)
    b = (227*997*j+427965865) ^ (887*409*i)
    idx = np.uint32(a+b)
    iidx = ~idx
    print(f'({i}, {j}): {a = }, {b = }, {idx = },  {iidx = }')