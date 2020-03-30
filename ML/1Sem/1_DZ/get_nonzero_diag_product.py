import numpy as np


def get_nonzero_diag_product(X):
    Diag = np.diagonal(X)
    mask = Diag > 0
    Diag = Diag[mask]
    if Diag.size > 0:
        return Diag.prod()
    else:
        return None
