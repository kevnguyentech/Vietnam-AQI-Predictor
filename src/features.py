"""
Turns the daily pm25+weather table into a supervised learning table.

Framing, which matters more than the code: each row represents
"today" (day t). The features available are everything you'd
actually have in hand on day t if you were really doing this:
  - your own PM2.5 history up through today (lags, rolling stats)
  - tomorrow's WEATHER FORECAST (temp/rain/wind/humidity for day t+1)
The target is tomorrow's AQI category (day t+1).

That second point is the one easy mistake to make here, so it's
worth being explicit about it: weather forecasts for tomorrow are
known today and reasonably accurate, so using them as a feature
isn't leakage. Using tomorrow's PM2.5 reading as a feature WOULD be
leakage, since that's the thing you're trying to predict. We never
do that — pm25 only enters as a lag/rolling feature computed from
data up to and including today.

In this project, the weather "forecast" is approximated using the
ACTUAL recorded weather for day t+1 (since we only have historical
observed data, not archived forecasts). That's a reasonable stand-in
for training, since next-day forecasts are usually close to the
actual outcome, but it's a simplification worth knowing about: a
production system would log the forecast as it was actually issued,
not the ground truth, to avoid a train/serve skew where the model
learns to expect more accurate "forecasts" than it'll get in
production.
"""

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

    # Lag/rolling below shift by ROW position, not calendar date. If the
    # input has a missing day (e.g. a sensor outage upstream in
    # fetch_data.py), row-position shifting would silently treat the
    # next available day as "yesterday" instead of failing or dropping
    # it. Reindexing to a complete daily range turns any gap into an
    # explicit NaN row, which the existing dropna() below already
    # removes correctly, same effect as fetch_data.py's inner join
    # already has when both sources are complete: this only changes
    # behavior when there's a gap to catch.
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range).rename_axis("date").reset_index()

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
