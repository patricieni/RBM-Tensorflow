# Copyright 2017 Author @Patric Fulop.
import numpy as np


def get_data(rng):
    '''
    :param rng: A random number generator
    :return: A dataset containing 4x4 matrices either with rows as ones, or columns as ones.
    '''
    all_data = np.zeros(shape=16)
    size = 4
    big_enough = 0
    while big_enough < 500:
        data_i = np.zeros(shape=(4, 4))
        if rng.uniform() < 0.5:
            # to see whether we fill horizontally
            # direction = horizontal
            for s in range(0, size):
                if rng.uniform() < 0.5:
                    data_i[s] = np.zeros(shape=size)
                else:
                    data_i[s] = np.ones(shape=size)
            all_data = np.vstack([all_data, data_i.reshape(-1)])
        else:
            # direction = vertical
            for s in range(0, size):
                if rng.uniform() < 0.5:
                    data_i[:, s] = np.zeros(shape=size)
                else:
                    data_i[:, s] = np.ones(shape=size)
            all_data = np.vstack([all_data, data_i.reshape(-1)])
        big_enough += 1
    # uniqueness
    y = np.vstack({tuple(row) for row in all_data})
    return y
