import numpy as np


def calc_expectations(h, w, X, Q):
    C_1 = np.ones((Q.shape[0], Q.shape[0]))
    C_1 = np.triu(C_1, 1 - h)
    C_1 = np.tril(C_1)
    Q = np.dot(C_1, Q)
    C_1 = np.ones((Q.shape[1], Q.shape[1]))
    C_1 = np.triu(C_1)
    C_1 = np.tril(C_1, w - 1)
    Q = np.dot(Q, C_1)
    return Q * X
