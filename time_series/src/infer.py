import argparse, numpy as np, torch
from .data import load_series, SeriesScaler
from .models import LSTMForecaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--recent_csv", type=str, required=True)
    ap.add_argument("--date_col", type=str, default="date")
    ap.add_argument("--value_col", type=str, default="value")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    series, _ = load_series(args.recent_csv, args.date_col, args.value_col)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = SeriesScaler()
    sc.sc.mean_ = np.array([ckpt["scaler_mean"]])
    sc.sc.scale_ = np.array([ckpt["scaler_scale"]])

    buf = sc.transform(series).tolist()
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
        for _ in range(args.horizon):
            x = np.array(buf[-args.window :], dtype=np.float32).reshape(
                1, args.window, 1
            )
            x = torch.tensor(x).to(device)
            p = model(x).cpu().numpy().ravel()[0]
            preds.append(p)
            buf.append(p)

    preds_inv = sc.inverse_transform(preds)
    print("Forecast:", preds_inv.tolist())


if __name__ == "__main__":
    main()
