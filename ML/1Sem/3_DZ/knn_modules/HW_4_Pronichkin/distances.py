import numpy as np


def euclidean_distance(x, y, Neibors_norm=None):
    if Neibors_norm is None:
        return np.sqrt(- 2 * np.inner(x, y) +
                       np.sum(x ** 2, axis=1)[:, None] +
                       np.sum(y ** 2, axis=1))
    else:
        return np.sqrt(- 2 * np.inner(x, y) +
                       np.sum(x ** 2, axis=1)[:, None] +
                       Neibors_norm)


def cosine_distance(x, y, Neibors_norm=None):
    if Neibors_norm is None:
        return ((- np.inner(x, y) / np.linalg.norm(x, axis=1)[:, np.newaxis]) /
                np.linalg.norm(y, axis=1) + 1)
    else:
        return ((- np.inner(x, y) / np.linalg.norm(x, axis=1)[:, np.newaxis]) /
                Neibors_norm + 1)
