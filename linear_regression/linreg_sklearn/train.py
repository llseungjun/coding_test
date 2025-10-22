import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data import load_dataset
from .model import build_model
from common.utils import set_seed, load_yaml, save_pickle


def run_train(cfg_path="configs/sklearn.yaml"):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["random_seed"])
    X, y = load_dataset(cfg["data"]["csv_path"], cfg["data"].get("target"))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["random_seed"]
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg["data"]["val_size"], random_state=cfg["random_seed"]
    )

    model = build_model(
        normalize=cfg["training"]["normalize"],
        fit_intercept=cfg["training"]["fit_intercept"],
    )
    model.fit(X_tr, y_tr)

    def metrics(splitX, splity, name):
        pred = model.predict(splitX)
        mse = mean_squared_error(splity, pred)
        print(
            f"[{name}] RMSE={mse**0.5:.4f} | MAE={mean_absolute_error(splity,pred):.4f} | R2={r2_score(splity,pred):.4f}"
        )

    metrics(X_val, y_val, "VAL")
    metrics(X_te, y_te, "TEST")

    save_pickle({"model": model, "config": cfg}, cfg["paths"]["model_path"])
    print(f"Saved → {cfg['paths']['model_path']}")
