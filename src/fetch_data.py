"""
Pulls real PM2.5 + weather history for Hanoi and writes a single
joined CSV to data/processed/aqi_weather_daily.csv.

Two data sources:
  1. OpenAQ v3      -> daily PM2.5 averages from a ground station
  2. Open-Meteo     -> daily weather (temp, humidity, wind, rain)

OpenAQ v3 requires a free API key (the old v2 endpoints this project
was originally scoped against were retired Jan 2025). Get one at
https://explore.openaq.org/register, then either:

    export OPENAQ_API_KEY="your-key-here"

or pass it with --api-key.

Open-Meteo's archive needs no key at all.

Run:
    python src/fetch_data.py --start 2023-01-01 --end 2024-12-31
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

from config import HANOI_LAT, HANOI_LON, TIMEZONE, DATA_RAW, PROCESSED_FILE

OPENAQ_BASE = "https://api.openaq.org/v3"
OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


def find_pm25_sensor(api_key: str, radius_m: int = 25_000) -> dict:
    """
    OpenAQ v3 is resource-oriented: you can't ask for measurements by
    city name directly. You first find *locations* (physical stations)
    near a coordinate, then pull the *sensor* at that location that
    measures PM2.5 (parameter id 2), then query that sensor's daily
    aggregates. This function does step one and two.
    """
    url = f"{OPENAQ_BASE}/locations"
    params = {
        "coordinates": f"{HANOI_LAT},{HANOI_LON}",
        "radius": radius_m,
        "parameters_id": 2,  # 2 = pm25
        "limit": 50,
    }
    headers = {"X-API-Key": api_key}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])

    if not results:
        raise RuntimeError(
            f"No PM2.5 stations found within {radius_m/1000:.0f}km of Hanoi. "
            "Try increasing radius_m."
        )

    # Prefer a station with the most reporting history (datetimeLast -
    # datetimeFirst). Fall back to the first result if that field is
    # missing from the response.
    station = results[0]
    sensor = next(
        (s for s in station["sensors"] if s["parameter"]["name"] == "pm25"),
        None,
    )
    if sensor is None:
        raise RuntimeError(f"Station '{station['name']}' has no pm25 sensor.")

    print(f"Using station: {station['name']} (location_id={station['id']}, "
          f"sensor_id={sensor['id']})")
    return {"location_id": station["id"], "sensor_id": sensor["id"],
             "station_name": station["name"]}


def fetch_pm25_daily(api_key: str, sensor_id: int, date_from: str, date_to: str) -> pd.DataFrame:
    """
    Pulls daily PM2.5 averages for one sensor. OpenAQ paginates at up
    to 1000 rows/page; for a 2-year daily series that's one page, but
    the loop below handles longer ranges too.
    """
    url = f"{OPENAQ_BASE}/sensors/{sensor_id}/days"
    headers = {"X-API-Key": api_key}
    all_rows = []
    page = 1

    while True:
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": 1000,
            "page": page,
        }
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        if not results:
            break

        for row in results:
            all_rows.append({
                "date": row["period"]["datetimeFrom"]["local"][:10],
                "pm25": row["value"],
            })

        found = payload.get("meta", {}).get("found", 0)
        if page * 1000 >= found:
            break
        page += 1
        time.sleep(0.2)  # be polite to the rate limiter

    df = pd.DataFrame(all_rows).drop_duplicates(subset="date")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_weather_daily(date_from: str, date_to: str) -> pd.DataFrame:
    """Open-Meteo's archive endpoint, no key required."""
    params = {
        "latitude": HANOI_LAT,
        "longitude": HANOI_LON,
        "start_date": date_from,
        "end_date": date_to,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_mean",
        ]),
        "timezone": TIMEZONE,
    }
    r = requests.get(OPENMETEO_BASE, params=params, timeout=30)
    r.raise_for_status()
    daily = r.json()["daily"]
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precipitation",
        "windspeed_10m_max": "wind_speed",
        "relative_humidity_2m_mean": "humidity",
    })
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAQ_API_KEY"))
    parser.add_argument("--radius-m", type=int, default=25_000,
                         help="search radius around central Hanoi, in meters")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit(
            "Missing OpenAQ API key. Get a free one at "
            "https://explore.openaq.org/register, then either:\n"
            "  export OPENAQ_API_KEY=your-key\n"
            "or pass --api-key your-key"
        )

    print(f"Fetching PM2.5 + weather for Hanoi, {args.start} to {args.end}...")

    station = find_pm25_sensor(args.api_key, radius_m=args.radius_m)
    pm25_df = fetch_pm25_daily(args.api_key, station["sensor_id"], args.start, args.end)
    print(f"  PM2.5: {len(pm25_df)} daily readings from '{station['station_name']}'")

    weather_df = fetch_weather_daily(args.start, args.end)
    print(f"  Weather: {len(weather_df)} daily records")

    pm25_df.to_csv(DATA_RAW / "pm25_raw.csv", index=False)
    weather_df.to_csv(DATA_RAW / "weather_raw.csv", index=False)

    merged = pd.merge(pm25_df, weather_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(PROCESSED_FILE, index=False)

    print(f"\nJoined {len(merged)} rows -> {PROCESSED_FILE}")
    if len(merged) < 365:
        print("Heads up: under a year of data. The model in train.py "
              "will still run, but lag/rolling features and seasonal "
              "patterns will be weaker with less history.")


if __name__ == "__main__":
    main()
