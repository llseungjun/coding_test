import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mape(y, p):
    return float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-8))) * 100)


def evaluate_all(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
