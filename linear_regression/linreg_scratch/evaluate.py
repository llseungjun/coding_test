import pandas as pd
import numpy as np
from common.utils import load_pickle
from .features import add_bias
from .metrics import rmse, mae, r2


def run_evaluate(model_path: str, csv_path: str, target_col: str):
    bundle = load_pickle(model_path)
    model, scaler, cfg = bundle["model"], bundle["scaler"], bundle["config"]
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    if cfg["training"]["add_bias"]:
        X = add_bias(X)
    pred = model.predict(X)
    print(
        f"[EVAL] RMSE={rmse(y,pred):.4f} | MAE={mae(y,pred):.4f} | R2={r2(y,pred):.4f}"
    )
    return pred
