from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
MEAN_COLUMNS = ["voltage", "global_intensity"]
WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {
        "global_active_power": "global_active_power",
        "global_reactive_power": "global_reactive_power",
        "voltage": "voltage",
        "global_intensity": "global_intensity",
        "sub_metering_1": "sub_metering_1",
        "sub_metering_2": "sub_metering_2",
        "sub_metering_3": "sub_metering_3",
        "date": "date",
        "time": "time",
        "datetime": "datetime",
        "rr": "RR",
        "nbjrr1": "NBJRR1",
        "nbjrr5": "NBJRR5",
        "nbjrr10": "NBJRR10",
        "nbjbrou": "NBJBROU",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    return df


def _read_course_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", dayfirst=True)
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError(f"{path} must contain datetime, date, or date+time columns")
    df.insert(0, "datetime", dt)
    return df.dropna(subset=["datetime"])


def _download_uci() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "household_power_consumption.zip"
    txt_path = RAW_DIR / "household_power_consumption.txt"
    if not txt_path.exists():
        if not zip_path.exists():
            with urlopen(UCI_URL, timeout=120) as response:
                zip_path.write_bytes(response.read())
        with zipfile.ZipFile(zip_path) as zf:
            member = next(name for name in zf.namelist() if name.endswith(".txt"))
            txt_path.write_bytes(zf.read(member))
    return pd.read_csv(
        txt_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )


def load_minute_data() -> pd.DataFrame:
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    typo_test_path = RAW_DIR / "tes.csv"
    if train_path.exists() and (test_path.exists() or typo_test_path.exists()):
        test_path = test_path if test_path.exists() else typo_test_path
        return pd.concat([_read_course_csv(train_path), _read_course_csv(test_path)], ignore_index=True)
    df = _download_uci()
    df = _normalize_columns(df)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce", dayfirst=True)
    return df.dropna(subset=["datetime"])


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    available_numeric = [c for c in SUM_COLUMNS + MEAN_COLUMNS + WEATHER_COLUMNS if c in df.columns]
    for col in available_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg: dict[str, str] = {}
    for col in SUM_COLUMNS:
        if col in df.columns:
            agg[col] = "sum"
    for col in MEAN_COLUMNS:
        if col in df.columns:
            agg[col] = "mean"
    for col in WEATHER_COLUMNS:
        if col in df.columns:
            agg[col] = "first"

    daily = df.set_index("datetime").resample("D").agg(agg)
    daily = daily.sort_index()
    daily = daily.interpolate(limit_direction="both").ffill().bfill()

    if {"global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3"}.issubset(daily.columns):
        daily["sub_metering_remainder"] = (
            daily["global_active_power"] * 1000 / 60
            - daily["sub_metering_1"]
            - daily["sub_metering_2"]
            - daily["sub_metering_3"]
        )

    daily["dayofweek_sin"] = np.sin(2 * np.pi * daily.index.dayofweek / 7)
    daily["dayofweek_cos"] = np.cos(2 * np.pi * daily.index.dayofweek / 7)
    daily["month_sin"] = np.sin(2 * np.pi * daily.index.month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily.index.month / 12)
    daily.index.name = "date"
    return daily.reset_index()


def split_daily(daily: pd.DataFrame, test_days: int = 365) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if len(daily) <= 90 + test_days:
        raise ValueError("Not enough daily rows for 90-day input and requested test split")
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()
    daily.to_csv(PROCESSED_DIR / "daily_all.csv", index=False)
    train.to_csv(PROCESSED_DIR / "daily_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "daily_test.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=365)
    args = parser.parse_args()
    daily = aggregate_daily(load_minute_data())
    split_daily(daily, test_days=args.test_days)
    print(f"Wrote {len(daily)} daily rows to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
