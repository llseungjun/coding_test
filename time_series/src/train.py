import os, argparse, json, yaml, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from .data import load_series, make_windows, SeriesScaler, train_val_split
from .metrics import evaluate_all
from .baselines import naive_forecast, seasonal_naive_forecast, moving_average_forecast
from .models import LSTMForecaster


class SeriesDS(Dataset):
    def __init__(self, arr, window, horizon=1):
        X, y = make_windows(arr, window, horizon)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N,W,1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_seed(seed):
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--date_col", type=str, default=None)
    ap.add_argument("--value_col", type=str, default=None)
    ap.add_argument(
        "--model",
        type=str,
        choices=["naive", "seasonal_naive", "moving_avg", "arima", "lstm"],
        default=None,
    )
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--seasonal_m", type=int, default=None)
    ap.add_argument("--ma_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # override cfg with CLI
    for k in [
        "date_col",
        "value_col",
        "model",
        "window",
        "horizon",
        "epochs",
        "batch_size",
        "lr",
        "hidden",
        "layers",
        "dropout",
        "seasonal_m",
        "ma_k",
        "seed",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    set_seed(cfg.get("seed", 42))
    os.makedirs(args.out_dir, exist_ok=True)

    series, df = load_series(
        args.data_csv,
        date_col=cfg.get("date_col", "date"),
        value_col=cfg.get("value_col", "value"),
    )

    # Train/Val split
    tr, va = train_val_split(series, val_ratio=0.2)

    # baselines
    if cfg["model"] in ["naive", "seasonal_naive", "moving_avg"]:
        if cfg["model"] == "naive":
            pred = naive_forecast(
                va[: -cfg["horizon"]] if cfg["horizon"] > 0 else va, steps=len(va)
            )
        elif cfg["model"] == "seasonal_naive":
            pred = seasonal_naive_forecast(va, m=cfg["seasonal_m"], steps=len(va))
        else:
            pred = moving_average_forecast(va, k=cfg["ma_k"], steps=len(va))
        metrics = evaluate_all(va, pred)
        print(metrics)
        with open(
            os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        # save a pseudo checkpoint
        torch.save({"cfg": cfg}, os.path.join(args.out_dir, "best.pt"))
        return

    # ARIMA (optional)
    if cfg["model"] == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        # simple fit on train, predict on val length
        model = ARIMA(tr, order=(2, 1, 2))
        res = model.fit()
        pred = res.forecast(steps=len(va))
        metrics = evaluate_all(va, pred)
        print(metrics)
        with open(
            os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        torch.save({"cfg": cfg}, os.path.join(args.out_dir, "best.pt"))
        return

    # LSTM
    sc = SeriesScaler()
    tr_sc = sc.fit_transform(tr)
    va_sc = sc.transform(va)

    train_ds = SeriesDS(tr_sc, window=cfg["window"], horizon=cfg["horizon"])
    val_ds = SeriesDS(
        np.concatenate([tr_sc[-cfg["window"] :], va_sc]),
        window=cfg["window"],
        horizon=cfg["horizon"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(
        input_size=1,
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        out=cfg["horizon"],
        drop=cfg["dropout"],
    ).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    best_rmse, best_path = float("inf"), os.path.join(args.out_dir, "best.pt")

    for e in range(1, cfg["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            tr_loss += loss.item() * bs
            n += bs
        # val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).squeeze().detach().cpu().numpy()
                preds.append(pred)
                gts.append(yb.numpy())
        import numpy as np

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        # inverse scale
        preds_inv = sc.inverse_transform(preds)
        gts_inv = sc.inverse_transform(gts)
        metrics = evaluate_all(gts_inv, preds_inv)
        print(
            f"[{e:03d}] train_loss={tr_loss/max(1,n):.4f} | RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} MAPE={metrics['mape']:.2f}%"
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "scaler_mean": float(sc.sc.mean_),
                    "scaler_scale": float(sc.sc.scale_),
                },
                best_path,
            )

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"rmse": best_rmse}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
