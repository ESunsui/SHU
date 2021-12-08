import numpy as np


def solve_lru(pagelist, blocksize):
    def top(block, PAGE):
        pos = np.argwhere(block == PAGE)
        for i in range(pos[0][0], 0, -1):
            block[i] = block[i-1]
        block[0] = PAGE

    def add(block, PAGE):
        for i in range(blocksize-1, 0, -1):
            block[i] = block[i-1]
        block[0] = PAGE

    count = 0
    block = np.zeros(blocksize)
    for PAGE in pagelist:
        if PAGE in block:
            top(block, PAGE)
            print(block)
        else:
            add(block, PAGE)
            count = count + 1
            print(block)
    return count


page = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
print(solve_lru(page, 3))
