import argparse, json, os, numpy as np, torch
from .data import load_series, SeriesScaler, make_windows
from .metrics import evaluate_all
from .models import LSTMForecaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--date_col", type=str, default="date")
    ap.add_argument("--value_col", type=str, default="value")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    series, _ = load_series(args.data_csv, args.date_col, args.value_col)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["model"] != "lstm":
        raise ValueError("evaluate.py currently supports lstm checkpoints")

    sc = SeriesScaler()
    sc.sc.mean_ = np.array([ckpt["scaler_mean"]])
    sc.sc.scale_ = np.array([ckpt["scaler_scale"]])

    s_sc = sc.transform(series)
    X, y = make_windows(s_sc, args.window, args.horizon)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)

    model = LSTMForecaster(
        input_size=1,
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        out=cfg["horizon"],
        drop=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X)):
            p = model(X[i : i + 1]).cpu().numpy().ravel()
            preds.append(p[0] if args.horizon == 1 else p)
    preds = np.array(preds).squeeze()

    preds_inv = sc.inverse_transform(preds)
    y_inv = sc.inverse_transform(y)
    metrics = evaluate_all(y_inv, preds_inv)
    print(metrics)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
