import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from common.utils import load_pickle


def run_evaluate(model_path: str, csv_path: str, target_col: str):
    bundle = load_pickle(model_path)
    model = bundle["model"]
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    print(
        f"[EVAL] RMSE={mse**0.5:.4f} | MAE={mean_absolute_error(y,pred):.4f} | R2={r2_score(y,pred):.4f}"
    )
    return pred
