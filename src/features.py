import argparse

import numpy as np
import pandas as pd

from config import (
    PROCESSED_FILE, FEATURES_FILE, AQI_BINS, AQI_LABELS,
    AQI_LABEL_TO_INT, LAG_DAYS, ROLLING_WINDOW,
)

WEATHER_COLS = ["temp_max", "temp_min", "precipitation", "wind_speed", "humidity"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- AQI history, known as of "today" ---------------------------
    for lag in LAG_DAYS:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    df["pm25_rolling_mean_7"] = df["pm25"].rolling(ROLLING_WINDOW).mean()
    df["pm25_rolling_std_7"] = df["pm25"].rolling(ROLLING_WINDOW).std()

    # --- calendar, known as of "today" -------------------------------
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- tomorrow's weather forecast ---------------------------------
    # shift(-1) pulls each weather column one row UP, so the row for
    # day t holds day t+1's weather. This is the "forecast" feature set.
    for col in WEATHER_COLS:
        df[f"fcst_{col}"] = df[col].shift(-1)

    # --- target: tomorrow's AQI category -----------------------------
    df["pm25_next_day"] = df["pm25"].shift(-1)
    df["aqi_category"] = pd.cut(df["pm25_next_day"], bins=AQI_BINS, labels=AQI_LABELS)
    df["target"] = df["aqi_category"].map(AQI_LABEL_TO_INT)

    feature_cols = (
        [f"pm25_lag_{lag}" for lag in LAG_DAYS]
        + ["pm25_rolling_mean_7", "pm25_rolling_std_7"]
        + ["month", "day_of_week", "is_weekend"]
        + [f"fcst_{col}" for col in WEATHER_COLS]
    )

    keep = ["date"] + feature_cols + ["pm25_next_day", "aqi_category", "target"]
    out = df[keep].dropna().reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out, feature_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROCESSED_FILE))
    parser.add_argument("--output", default=str(FEATURES_FILE))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out, feature_cols = build_features(df)
    out.to_csv(args.output, index=False)

    print(f"Built {len(out)} labeled rows, {len(feature_cols)} features -> {args.output}")
    print(f"Dropped {len(df) - len(out)} rows to NaN (lag/rolling warm-up + last day with no 'tomorrow')")
    print("\nClass balance:")
    print(out["aqi_category"].value_counts())


if __name__ == "__main__":
    main()
