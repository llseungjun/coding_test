import numpy as np


def mse(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.mean((y - p) ** 2))


def rmse(y, p):
    return mse(y, p) ** 0.5


def mae(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)
