import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from features import build_features

# 20 days, so day 15 has 7+ days of lag/rolling history AND a "tomorrow"
dates = pd.date_range("2024-01-01", periods=20)
df = pd.DataFrame({
    "date": dates,
    "pm25": list(range(10, 210, 10)),  # 10, 20, 30, ..., 200
    "temp_max": list(range(30, 50)),
    "temp_min": list(range(20, 40)),
    "precipitation": [0]*20,
    "wind_speed": [5]*20,
    "humidity": [60]*20,
})

out, feature_cols = build_features(df)

# Test 1: lag_1 on day 15 equals pm25 from day 14
row = out[out["date"] == "2024-01-15"]
assert len(row) == 1, "Expected exactly one row for 2024-01-15"
expected_lag1 = df[df["date"] == "2024-01-14"]["pm25"].values[0]
assert row["pm25_lag_1"].values[0] == expected_lag1, f"lag_1 wrong: {row['pm25_lag_1'].values[0]}"
print("PASS: pm25_lag_1 pulls from the correct previous day")

# Test 2: forecast weather is tomorrow's weather, not today's
expected_fcst = df[df["date"] == "2024-01-16"]["temp_max"].values[0]
assert row["fcst_temp_max"].values[0] == expected_fcst, f"fcst_temp_max wrong: {row['fcst_temp_max'].values[0]}"
print("PASS: fcst_temp_max pulls from tomorrow, not today")

# Test 3: target is derived from tomorrow's pm25, not today's
expected_next = df[df["date"] == "2024-01-16"]["pm25"].values[0]
assert row["pm25_next_day"].values[0] == expected_next, f"pm25_next_day wrong: {row['pm25_next_day'].values[0]}"
print("PASS: pm25_next_day is tomorrow's value, not today's")

# Test 4: no leakage — today's pm25 itself must not appear as a feature
assert "pm25" not in feature_cols, "Raw pm25 leaked into feature_cols"
print("PASS: raw pm25 not present in feature_cols (no direct leakage)")

# Test 5: last row (no "tomorrow" to look up) gets dropped
last_date = df["date"].max()
assert last_date not in out["date"].values, "Last row should be dropped (no tomorrow data)"
print("PASS: last day dropped due to missing forecast/target")

print("\nAll tests passed.")