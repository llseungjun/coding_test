import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_series(csv_path, date_col="date", value_col="value"):
    df = pd.read_csv(csv_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    s = df[value_col].astype(float).values
    # 간단 결측 처리
    s = pd.Series(s).interpolate(method="linear").bfill().ffill().values
    return s, df


def train_val_split(series, val_ratio=0.2):
    n = len(series)
    idx = int(n * (1 - val_ratio))
    return series[:idx], series[idx:]


def make_windows(arr, window, horizon=1):
    X, y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i : i + window])
        y.append(arr[i + window : i + window + horizon])
    X = np.array(X)
    y = np.array(y).squeeze()
    return X, y


class SeriesScaler:
    def __init__(self):
        self.sc = StandardScaler()

    def fit_transform(self, x):
        return self.sc.fit_transform(x.reshape(-1, 1)).ravel()

    def transform(self, x):
        return self.sc.transform(x.reshape(-1, 1)).ravel()

    def inverse_transform(self, x):
        return self.sc.inverse_transform(np.array(x).reshape(-1, 1)).ravel()
