from pathlib import Path

# ---- paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PROCESSED_FILE = DATA_PROCESSED / "aqi_weather_daily.csv"
FEATURES_FILE = DATA_PROCESSED / "features.csv"
MODEL_FILE = MODELS_DIR / "aqi_model.json"
MODEL_META_FILE = MODELS_DIR / "model_meta.pkl"

# ---- location ----------------------------------------------------------
# Central Hanoi. OpenAQ location search and Open-Meteo archive both
# take a single lat/lon pair.
HANOI_LAT = 21.0285
HANOI_LON = 105.8542
TIMEZONE = "Asia/Bangkok"  # UTC+7, same offset as Asia/Ho_Chi_Minh

# ---- AQI categories ------------------------------------------------------
# Simplified 3-bucket version of the US EPA PM2.5 breakpoints (24h mean,
# µg/m³). Real EPA scale has 6 buckets; collapsing Unhealthy-for-Sensitive
# through Hazardous into one "Unhealthy" bucket keeps the classifier
# usable with ~2 years of daily data instead of needing 10x more rows
# to populate six classes.
AQI_BINS = [-float("inf"), 12.0, 35.4, float("inf")]
AQI_LABELS = ["Good", "Moderate", "Unhealthy"]
AQI_LABEL_TO_INT = {label: i for i, label in enumerate(AQI_LABELS)}

# ---- modeling ----------------------------------------------------------
LAG_DAYS = [1, 3, 7]
ROLLING_WINDOW = 7
TEST_HOLDOUT_DAYS = 150  # most recent N days held out as the final test set
# 150, not something smaller like 60, on purpose: this dataset is
# strongly seasonal, and the last 60-90 days of the year are almost
# entirely deep-winter "Unhealthy" days in Hanoi. A short trailing
# holdout would test the model on a single class and report a
# meaningless 100% accuracy. 150 days reaches back into late summer,
# so the test set actually contains a mix of all three categories —
# see train.py for the real class counts this produces.
RANDOM_SEED = 42

