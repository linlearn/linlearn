import numpy as np
from numba import njit


@njit
def median_of_means(x, block_size, blocks_means):
    n = x.shape[0]
    n_blocks = n // block_size
    last_block_size = n % block_size
    sum_block = 0.0
    n_block = 0
    for i in range(n):
        # Update current sum in the block
        # print(sum_block, "+=", x[i])
        sum_block += x[i]
        if (i != 0) and ((i + 1) % block_size == 0):
            # It's the end of the block, save its mean
            # print("sum_block: ", sum_block)
            blocks_means[n_block] = sum_block / block_size
            n_block += 1
            sum_block = 0.0

    if last_block_size != 0:
        blocks_means[n_blocks] = sum_block / last_block_size

    mom = np.median(blocks_means)
    return mom, blocks_means


x = np.arange(0, 12).astype("float64")
x = np.random.permutation(x)
x[4] = -1200

n = x.shape[0]
block_size = 2
n_blocks = n // block_size

if n % block_size == 0:
    blocks_means = np.empty(n_blocks)
else:
    blocks_means = np.empty(n_blocks + 1)

print(x)
mom, blocks_means = median_of_means(x, block_size, blocks_means)

print(blocks_means)

print("mom: ", mom)
