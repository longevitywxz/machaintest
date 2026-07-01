from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import CNNTransformerForecaster, LSTMForecaster, TransformerForecaster


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "daily_all.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TARGET = "global_active_power"
INPUT_WINDOW = 90


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray
    target_mean: float
    target_std: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_windows(values: np.ndarray, target: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, starts = [], [], []
    for start in range(0, len(values) - INPUT_WINDOW - horizon + 1):
        x_end = start + INPUT_WINDOW
        y_end = x_end + horizon
        xs.append(values[start:x_end])
        ys.append(target[x_end:y_end])
        starts.append(x_end)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(starts)


def load_data(horizon: int, test_days: int = 365) -> tuple[TensorDataset, TensorDataset, Scale, list[str], pd.Series]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feature_cols = [c for c in df.columns if c != "date"]
    values_raw = df[feature_cols].astype(float).to_numpy()
    target_raw = df[TARGET].astype(float).to_numpy()
    train_cutoff = len(df) - test_days

    train_rows = values_raw[:train_cutoff]
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0)
    std[std == 0] = 1.0
    target_idx = feature_cols.index(TARGET)
    scale = Scale(mean=mean, std=std, target_mean=float(mean[target_idx]), target_std=float(std[target_idx]))
    values = (values_raw - mean) / std
    target = (target_raw - scale.target_mean) / scale.target_std

    xs, ys, starts = make_windows(values, target, horizon)
    train_mask = starts + horizon <= train_cutoff
    test_mask = starts >= train_cutoff - INPUT_WINDOW
    x_train, y_train = xs[train_mask], ys[train_mask]
    x_test, y_test = xs[test_mask], ys[test_mask]
    dates = df["date"].iloc[starts[test_mask][0] : starts[test_mask][0] + horizon].reset_index(drop=True)

    return (
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        TensorDataset(torch.from_numpy(x_test[:1]), torch.from_numpy(y_test[:1])),
        scale,
        feature_cols,
        dates,
    )


def build_model(name: str, n_features: int, horizon: int) -> nn.Module:
    if name == "lstm":
        return LSTMForecaster(n_features, horizon)
    if name == "transformer":
        return TransformerForecaster(n_features, horizon)
    if name == "cnn_transformer":
        return CNNTransformerForecaster(n_features, horizon)
    raise ValueError(f"Unknown model: {name}")


def evaluate(model: nn.Module, loader: DataLoader, scale: Scale, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    pred_arr = np.concatenate(preds) * scale.target_std + scale.target_mean
    true_arr = np.concatenate(trues) * scale.target_std + scale.target_mean
    mse = float(np.mean((pred_arr - true_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    return mse, mae, pred_arr[0], true_arr[0]


def train_one(model_name: str, horizon: int, seed: int, epochs: int, batch_size: int, lr: float) -> dict[str, float | str | int]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, scale, feature_cols, dates = load_data(horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    model = build_model(model_name, len(feature_cols), horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    mse, mae, pred, true = evaluate(model, test_loader, scale, device)
    if seed == 0:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIG_DIR / f"{model_name}_{horizon}.png"
        plt.figure(figsize=(11, 4))
        plt.plot(dates.iloc[:horizon], true[:horizon], label="Ground Truth", linewidth=2)
        plt.plot(dates.iloc[:horizon], pred[:horizon], label="Prediction", linewidth=2)
        plt.title(f"{model_name} {horizon}-day forecast")
        plt.xlabel("Date")
        plt.ylabel("Global active power")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
    return {"model": model_name, "horizon": horizon, "seed": seed, "mse": mse, "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "cnn_transformer"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    rows = []
    done = set()
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        rows = existing.to_dict("records")
        done = set(zip(existing["model"], existing["horizon"], existing["seed"]))
    for horizon in args.horizons:
        for model_name in args.models:
            for seed in args.seeds:
                if (model_name, horizon, seed) in done:
                    print(f"skipping model={model_name} horizon={horizon} seed={seed}")
                    continue
                print(f"training model={model_name} horizon={horizon} seed={seed}")
                rows.append(train_one(model_name, horizon, seed, args.epochs, args.batch_size, args.lr))
                pd.DataFrame(rows).to_csv(metrics_path, index=False)

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["model", "horizon"])
        .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"))
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
