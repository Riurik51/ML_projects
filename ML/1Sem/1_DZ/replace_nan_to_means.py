import numpy as np


def replace_nan_to_means(X):
    sred = np.nanmean(X, axis=0)
    mask = np.isnan(sred)
    sred[mask] = 0
    wherenan = np.where(np.isnan(X))
    X[wherenan] = sred[wherenan[1]]
    return X
