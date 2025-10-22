import numpy as np


def naive_forecast(series, steps=1):
    return np.array([series[-1]] * steps, dtype=float)


def seasonal_naive_forecast(series, m=7, steps=1):
    pred = []
    for i in range(steps):
        pred.append(series[-m + (i % m)])
    return np.array(pred, dtype=float)


def moving_average_forecast(series, k=5, steps=1):
    mean_val = np.mean(series[-k:])
    return np.array([mean_val] * steps, dtype=float)
