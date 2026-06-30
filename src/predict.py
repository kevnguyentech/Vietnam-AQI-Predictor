"""
The actual tool. Run this day-to-day once the model is trained:

    python src/predict.py --temp-max 19 --temp-min 14 \\
        --humidity 85 --wind-speed 6 --precipitation 0

Those five numbers are tomorrow's weather FORECAST — pull them from
any weather app for Hanoi the night before. The script reads your
own recent PM2.5 history from data/processed/aqi_weather_daily.csv
automatically (it needs the last 7 days to compute the same lag and
rolling features train.py used) and prints tomorrow's predicted AQI
category with a probability breakdown.

This only works if the trained model's idea of "today" lines up with
where your data file actually ends. If you've been running
fetch_data.py daily to keep the CSV current, that's automatic. If
you're working off the one-time simulated dataset, "today" is
whatever the last row of that file says (2024-12-31), not the
literal current date — see --date to override.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import (
    PROCESSED_FILE, MODEL_FILE, MODEL_META_FILE, AQI_LABELS, LAG_DAYS, ROLLING_WINDOW,
)


def load_model():
    meta = joblib.load(MODEL_META_FILE)
    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))
    return model, meta


def build_live_feature_row(history: pd.DataFrame, as_of_date: pd.Timestamp,
                            forecast: dict) -> pd.DataFrame:
    """
    Mirrors features.py exactly, but for a single live row instead of
    a full historical table — same lag/rolling logic, computed from
    whatever PM2.5 history is available up through as_of_date, plus
    the forecast dict standing in for "tomorrow's weather."
    """
    hist = history[history["date"] <= as_of_date].sort_values("date")
    if len(hist) < max(LAG_DAYS + [ROLLING_WINDOW]):
        raise ValueError(
            f"Need at least {max(LAG_DAYS + [ROLLING_WINDOW])} days of PM2.5 "
            f"history before {as_of_date.date()}, only have {len(hist)}."
        )

    pm25_series = hist["pm25"].values
    row = {}
    for lag in LAG_DAYS:
        row[f"pm25_lag_{lag}"] = pm25_series[-lag]
    row["pm25_rolling_mean_7"] = pm25_series[-ROLLING_WINDOW:].mean()
    row["pm25_rolling_std_7"] = pm25_series[-ROLLING_WINDOW:].std()

    tomorrow = as_of_date + pd.Timedelta(days=1)
    row["month"] = tomorrow.month
    row["day_of_week"] = tomorrow.dayofweek
    row["is_weekend"] = int(tomorrow.dayofweek >= 5)

    row["fcst_temp_max"] = forecast["temp_max"]
    row["fcst_temp_min"] = forecast["temp_min"]
    row["fcst_precipitation"] = forecast["precipitation"]
    row["fcst_wind_speed"] = forecast["wind_speed"]
    row["fcst_humidity"] = forecast["humidity"]

    return pd.DataFrame([row]), tomorrow


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temp-max", type=float, required=True, help="forecasted high, °C")
    parser.add_argument("--temp-min", type=float, required=True, help="forecasted low, °C")
    parser.add_argument("--humidity", type=float, required=True, help="forecasted mean relative humidity, %%")
    parser.add_argument("--wind-speed", type=float, required=True, help="forecasted max wind speed, km/h")
    parser.add_argument("--precipitation", type=float, required=True, help="forecasted rain sum, mm")
    parser.add_argument("--date", default=None,
                         help="treat this date (YYYY-MM-DD) as 'today'; defaults to the "
                              "last date in the processed data file")
    args = parser.parse_args()

    model, meta = load_model()
    history = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])

    as_of_date = pd.Timestamp(args.date) if args.date else history["date"].max()

    forecast = {
        "temp_max": args.temp_max, "temp_min": args.temp_min,
        "humidity": args.humidity, "wind_speed": args.wind_speed,
        "precipitation": args.precipitation,
    }
    X_live, target_date = build_live_feature_row(history, as_of_date, forecast)
    X_live = X_live[meta["feature_cols"]]  # enforce exact training column order

    probs = model.predict_proba(X_live)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = AQI_LABELS[pred_idx]

    print(f"\nAs of {as_of_date.date()}, forecasting {target_date.date()}:")
    print(f"  Predicted AQI category: {pred_label}  ({probs[pred_idx]*100:.0f}% confidence)")
    print("\n  Full breakdown:")
    for label, p in sorted(zip(AQI_LABELS, probs), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"    {label:10s} {p*100:5.1f}%  {bar}")

    if pred_label == "Unhealthy":
        print("\n  -> Consider a mask outdoors, limit prolonged outdoor exercise.")
    elif pred_label == "Good":
        print("\n  -> Good day to be outside.")


if __name__ == "__main__":
    main()
