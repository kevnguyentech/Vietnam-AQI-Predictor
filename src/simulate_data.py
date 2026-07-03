"""
Generates a synthetic but statistically realistic 2-year daily dataset
for Hanoi: PM2.5 + weather, written to the exact same schema that
fetch_data.py produces from the real APIs.

WHY THIS EXISTS: fetch_data.py needs an OpenAQ API key and live
internet access. If you don't have a key yet, or just want to see the
whole pipeline (features -> training -> SHAP -> predict) run
immediately, use this instead. Swap to real data later by running
fetch_data.py — nothing downstream changes, because the output columns
are identical.

The simulation isn't just noise. It encodes the actual mechanisms
that drive Hanoi's air quality so the model has real signal to find
(and SHAP has something real to explain):

  - Seasonal baseline: PM2.5 peaks in Jan, troughs in Jul. This is the
    single biggest driver — Hanoi's dry, cool winter traps pollution
    under a thermal inversion; the summer monsoon washes it out.
  - Autocorrelation: pollution is sticky day-to-day (an AR(1) process),
    not just a function of the calendar.
  - Weather effects: rain suppresses PM2.5 (washout), wind disperses
    it, cold temperatures make it worse (inversion strength), high
    humidity in cold months traps haze.
  - Burning-season spikes: clustered multi-day smog events in
    Oct-Nov (post-harvest crop burning) and Jan-Mar (Tet period +
    continued inversion), not just smoothly varying noise.

None of this is fitted to real measurements — it's built from how
Hanoi's air quality is documented to behave, so don't cite the
specific numbers it produces anywhere. The point is to have a dataset
where lag features, season, and weather all carry genuine, separable
signal, the way they would in the real thing.
"""

import argparse

import numpy as np
import pandas as pd

from config import DATA_RAW, PROCESSED_FILE, RANDOM_SEED


def seasonal_curve(doy: np.ndarray, mean: float, amplitude: float, peak_doy: float) -> np.ndarray:
    """A smooth yearly cosine cycle peaking at peak_doy."""
    return mean + amplitude * np.cos(2 * np.pi * (doy - peak_doy) / 365.0)


def simulate(start: str, end: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values.astype(float)
    month = dates.month.values

    # ---- weather ---------------------------------------------------
    temp_max = seasonal_curve(doy, mean=25, amplitude=8, peak_doy=197) \
        + rng.normal(0, 1.8, n)
    temp_min = temp_max - rng.uniform(4, 8, n)

    humidity = seasonal_curve(doy, mean=78, amplitude=5, peak_doy=227) \
        + rng.normal(0, 4, n)
    humidity = np.clip(humidity, 40, 98)

    # No seasonal component here on purpose. An earlier version gave
    # wind a seasonal cycle, which created a confound: pm25 is so
    # strongly seasonal (amplitude 30) that ANY seasonal pattern in a
    # weather variable correlates with it through the shared calendar,
    # regardless of the variable's actual direct causal effect. Wind's
    # -1.8 dispersal coefficient below was getting swamped by that
    # confound, producing a model that learned "high wind -> dirtier
    # air" — backwards. Pure day-to-day noise keeps wind's marginal
    # correlation with pm25 driven by the real causal channel only.
    wind_speed = 10 + 2.5 * (rng.lognormal(mean=0, sigma=0.35, size=n) - 1)
    wind_speed = np.clip(wind_speed, 1, None)

    rain_prob = 0.12 + 0.40 * (0.5 * (1 + np.cos(2 * np.pi * (doy - 197) / 365.0)))
    is_rainy = rng.random(n) < rain_prob
    rain_scale = 4 + 10 * (0.5 * (1 + np.cos(2 * np.pi * (doy - 197) / 365.0)))
    precipitation = np.where(is_rainy, rng.gamma(shape=2.0, scale=rain_scale), 0.0)

    # ---- PM2.5: seasonal baseline + weather effects + AR(1) + spikes ----
    seasonal_base = seasonal_curve(doy, mean=45, amplitude=30, peak_doy=15)

    cold_indicator = np.clip((22 - temp_max) / 10.0, 0, 1)
    weather_effect = (
        -0.9 * np.sqrt(precipitation)
        - 1.8 * (wind_speed - 10)
        - 1.1 * (temp_max - 25)
        + 0.15 * (humidity - 78) * cold_indicator
    )
    raw_level = seasonal_base + weather_effect

    pm25 = np.zeros(n)
    pm25[0] = max(seasonal_base[0], 5)
    noise = rng.normal(0, 6, n)

    # Burning-season spike state machine: smog episodes cluster in
    # specific windows instead of hitting randomly every day, the way
    # real multi-day smog events do.
    in_burn_window = ((month == 10) | (month == 11) |
                       (month == 1) | (month == 2) | (month == 3))
    burst_active, burst_remaining, burst_mult = False, 0, 1.0
    burst_multipliers = np.ones(n)

    for t in range(n):
        if burst_active:
            burst_multipliers[t] = burst_mult
            burst_remaining -= 1
            if burst_remaining <= 0:
                burst_active = False
        elif in_burn_window[t] and rng.random() < 0.035:
            burst_active = True
            burst_remaining = rng.integers(3, 7)
            burst_mult = rng.uniform(1.3, 1.7)
            burst_multipliers[t] = burst_mult

    # Burst multiplies the exogenous driver (raw_level) only — not the
    # carried-forward AR state. Multiplying pm25[t-1] directly would let
    # one burst's boost get re-multiplied by the next burst, compounding
    # exponentially across consecutive smog days, which is a modeling
    # bug, not a real atmospheric effect.
    raw_level_with_burst = raw_level * burst_multipliers
    for t in range(1, n):
        pm25[t] = 0.55 * pm25[t - 1] + 0.45 * raw_level_with_burst[t] + noise[t]

    pm25 = np.clip(pm25, 5, 220)

    df = pd.DataFrame({
        "date": dates,
        "pm25": pm25.round(1),
        "temp_max": temp_max.round(1),
        "temp_min": temp_min.round(1),
        "precipitation": precipitation.round(1),
        "wind_speed": wind_speed.round(1),
        "humidity": humidity.round(1),
    })
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = simulate(args.start, args.end, seed=args.seed)

    df.to_csv(DATA_RAW / "simulated_raw.csv", index=False)
    df.to_csv(PROCESSED_FILE, index=False)

    print(f"Simulated {len(df)} days ({args.start} to {args.end}) -> {PROCESSED_FILE}")
    print(f"PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} µg/m³, "
          f"mean {df['pm25'].mean():.1f}")
    print("\nMonthly PM2.5 averages (check: should peak Dec-Feb, trough Jun-Aug):")
    monthly = df.groupby(df["date"].dt.month)["pm25"].mean().round(1)
    for m, v in monthly.items():
        print(f"  month {m:2d}: {v}")


if __name__ == "__main__":
    main()
