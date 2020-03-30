import numpy as np


def encode_rle(x):
    last = x[-1]
    ans_1 = x[1:]
    x = x[:-1]
    mask = x == ans_1
    ans_1 = x[~mask]
    ans_1 = np.append(ans_1, last)
    ans_2 = np.nonzero(~mask)
    ans_2 = ans_2[0]
    if ans_2.size > 0:
        last = x.size - ans_2[-1]
        temp = np.array([0])
        temp = np.hstack([temp, ans_2])
        temp = temp[:-1]
        ans_2
 = ans_2 - temp
        ans_2[0] += 1
        ans_2 = np.append(ans_2, last)
    else :
        ans_2 = np.array([x.size + 1])
    return (ans_1, ans_2)
