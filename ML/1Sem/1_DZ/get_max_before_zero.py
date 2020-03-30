import numpy as np


def get_max_before_zero(x):
    mask = x == 0
    mask = np.roll(mask, 1)
    x = x[1:]
    mask = mask[1:]
    x = x[mask]
    if x.size > 0:
        return x.max()
    else:
        return None
