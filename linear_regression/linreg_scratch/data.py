import numpy as np
import pandas as pd


def load_dataset(
    csv_path: str | None,
    target_col: str | None,
    n_samples=1000,
    n_features=10,
    noise_std=1.0,
    seed=42,
):
    if csv_path:
        df = pd.read_csv(csv_path)
        assert target_col is not None, "CSV 사용 시 target_col 지정 필요"
        X = df.drop(columns=[target_col]).to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        return X, y
    # 합성 데이터 (y = Xw + noise)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=(n_features, 1))
    y = (X @ true_w).ravel() + rng.normal(scale=noise_std, size=(n_samples,))
    return X, y
