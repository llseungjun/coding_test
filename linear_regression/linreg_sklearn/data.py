import pandas as pd


def load_dataset(csv_path: str | None = None, target_col: str | None = None):
    if csv_path:
        df = pd.read_csv(csv_path)
        assert target_col is not None, "CSV 사용 시 target_col 지정 필요"
        X = df.drop(columns=[target_col]).to_numpy()
        y = df[target_col].to_numpy()
    else:
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
    return X, y
