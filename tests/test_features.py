import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_features


@pytest.fixture
def toy_df():
    # 20 days, so day 15 has 7+ days of lag/rolling history AND a "tomorrow"
    dates = pd.date_range("2024-01-01", periods=20)
    return pd.DataFrame({
        "date": dates,
        "pm25": list(range(10, 210, 10)),  # 10, 20, 30, ..., 200
        "temp_max": list(range(30, 50)),
        "temp_min": list(range(20, 40)),
        "precipitation": [0] * 20,
        "wind_speed": [5] * 20,
        "humidity": [60] * 20,
    })


@pytest.fixture
def built(toy_df):
    out, feature_cols = build_features(toy_df)
    row = out[out["date"] == "2024-01-15"]
    assert len(row) == 1, "Expected exactly one row for 2024-01-15"
    return toy_df, out, feature_cols, row


def test_lag_1_pulls_from_previous_day(built):
    toy_df, out, feature_cols, row = built
    expected_lag1 = toy_df[toy_df["date"] == "2024-01-14"]["pm25"].values[0]
    assert row["pm25_lag_1"].values[0] == expected_lag1


def test_forecast_weather_is_tomorrows(built):
    toy_df, out, feature_cols, row = built
    expected_fcst = toy_df[toy_df["date"] == "2024-01-16"]["temp_max"].values[0]
    assert row["fcst_temp_max"].values[0] == expected_fcst


def test_target_is_tomorrows_pm25(built):
    toy_df, out, feature_cols, row = built
    expected_next = toy_df[toy_df["date"] == "2024-01-16"]["pm25"].values[0]
    assert row["pm25_next_day"].values[0] == expected_next


def test_no_raw_pm25_leakage(built):
    toy_df, out, feature_cols, row = built
    assert "pm25" not in feature_cols


def test_last_day_dropped(toy_df):
    out, feature_cols = build_features(toy_df)
    last_date = toy_df["date"].max()
    assert last_date not in out["date"].values