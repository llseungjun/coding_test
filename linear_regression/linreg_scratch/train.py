import numpy as np
from common.utils import set_seed, load_yaml, save_pickle
from .data import load_dataset
from .features import StandardScaler, add_bias
from .model import LinearRegressionGD, LinearRegressionNE
from .metrics import rmse, mae, r2


def _split(X, y, test_size, val_size, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    remain = idx[n_test:]
    n_val = int(len(remain) * val_size)
    val_idx = remain[:n_val]
    train_idx = remain[n_val:]
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def run_train(cfg_path="configs/scratch.yaml"):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["random_seed"])
    X, y = load_dataset(
        cfg["data"]["csv_path"],
        cfg["data"].get("target"),
        cfg["data"]["n_samples"],
        cfg["data"]["n_features"],
        cfg["data"]["noise_std"],
        cfg["random_seed"],
    )
    X_tr, y_tr, X_val, y_val, X_te, y_te = _split(
        X, y, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["random_seed"]
    )

    scaler = None
    if cfg["training"]["normalize"]:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te = scaler.transform(X_te)

    if cfg["training"]["add_bias"]:
        X_tr = add_bias(X_tr)
        X_val = add_bias(X_val)
        X_te = add_bias(X_te)

    if cfg["training"]["solver"] == "gd":
        model = LinearRegressionGD(
            lr=cfg["training"]["lr"], epochs=cfg["training"]["epochs"]
        ).fit(X_tr, y_tr)
    else:
        model = LinearRegressionNE(l2=cfg["training"]["l2"]).fit(X_tr, y_tr)

    pv = model.predict(X_val)
    pt = model.predict(X_te)
    print(
        f"[VAL] RMSE={rmse(y_val,pv):.4f} | MAE={mae(y_val,pv):.4f} | R2={r2(y_val,pv):.4f}"
    )
    print(
        f"[TEST] RMSE={rmse(y_te,pt):.4f} | MAE={mae(y_te,pt):.4f} | R2={r2(y_te,pt):.4f}"
    )

    save_pickle(
        {"model": model, "scaler": scaler, "config": cfg}, cfg["paths"]["model_path"]
    )
    print(f"Saved → {cfg['paths']['model_path']}")
