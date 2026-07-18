import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_features
from predict import build_live_feature_row


@pytest.fixture
def toy_df():
    dates = pd.date_range("2024-01-01", periods=20)
    return pd.DataFrame({
        "date": dates,
        "pm25": list(range(10, 210, 10)),  # 10, 20, ..., 200
        "temp_max": list(range(30, 50)),
        "temp_min": list(range(20, 40)),
        "precipitation": [0] * 20,
        "wind_speed": [5] * 20,
        "humidity": [60] * 20,
    })


def test_live_lag_features_match_training(toy_df):
    """
    Regression test for a train/serve skew bug: build_live_feature_row
    used to index pm25_series[-lag], which returns as_of_date's own
    reading for lag=1 instead of the previous day's, and used numpy's
    population std instead of pandas' sample std. Asserts the live
    feature row for a given as_of_date exactly matches the row
    features.py would have built for that same date during training.
    """
    out, feature_cols = build_features(toy_df)
    train_row = out[out["date"] == "2024-01-15"]

    forecast = {
        "temp_max": toy_df.loc[toy_df["date"] == "2024-01-16", "temp_max"].values[0],
        "temp_min": toy_df.loc[toy_df["date"] == "2024-01-16", "temp_min"].values[0],
        "precipitation": 0,
        "wind_speed": 5,
        "humidity": 60,
    }
    live_row, tomorrow = build_live_feature_row(
        toy_df, pd.Timestamp("2024-01-15"), forecast
    )

    for col in ["pm25_lag_1", "pm25_lag_3", "pm25_lag_7",
                "pm25_rolling_mean_7", "pm25_rolling_std_7",
                "month", "day_of_week", "is_weekend"]:
        assert live_row[col].values[0] == pytest.approx(train_row[col].values[0]), (
            f"{col} mismatch between live and training features"
        )
    assert tomorrow == pd.Timestamp("2024-01-16")


def test_insufficient_history_raises(toy_df):
    short_history = toy_df.iloc[:5]
    forecast = {"temp_max": 30, "temp_min": 20, "precipitation": 0,
                "wind_speed": 5, "humidity": 60}
    with pytest.raises(ValueError):
        build_live_feature_row(short_history, pd.Timestamp("2024-01-05"), forecast)

    
def test_load_model_missing_meta_exits(tmp_path, monkeypatch):
    import model_io
    fake_model = tmp_path / "aqi_model.json"
    fake_model.touch()  # exists, so MODEL_FILE check passes
    monkeypatch.setattr(model_io, "MODEL_FILE", fake_model)
    monkeypatch.setattr(model_io, "MODEL_META_FILE", tmp_path / "no_meta.pkl")
    with pytest.raises(SystemExit):
        model_io.load_model()


def test_load_model_missing_model_file_exits(tmp_path, monkeypatch):
    import joblib, model_io
    fake_meta = tmp_path / "model_meta.pkl"
    joblib.dump({"feature_cols": [], "labels": []}, fake_meta)
    monkeypatch.setattr(model_io, "MODEL_META_FILE", fake_meta)
    monkeypatch.setattr(model_io, "MODEL_FILE", tmp_path / "no_model.json")
    with pytest.raises(SystemExit):
        model_io.load_model()