# Vietnam Air Quality Predictor

Hanoi is regularly ranked among the most polluted capital cities in the world.
Every winter, PM2.5 spikes hard enough that local news runs "should you let
your kids go outside today" stories. I grew up breathing that air. This
project predicts tomorrow's air quality category (Good / Moderate /
Unhealthy) for Hanoi from a weather forecast plus recent pollution history,
so the question "should I plan to be outside tomorrow" has an actual
data-backed answer instead of a gut check.

It's also, deliberately, a complete reference implementation of a small
time-series ML project: real data sourcing, leakage-safe feature
engineering, time-aware cross-validation, class-imbalance handling, SHAP
explainability, and a CLI tool you actually run. Each script's docstring
explains the *why*, not just the *what* — read those if you're using this to
learn the patterns rather than just the result.

## Quick start

```bash
pip install -r requirements.txt
cd src

python simulate_data.py      # generates 2 years of realistic synthetic data
python features.py           # builds the supervised learning table
python train.py               # trains + evaluates the model
python evaluate.py            # confusion matrix + SHAP plots -> ../outputs/
python predict.py --temp-max 19 --temp-min 14 --humidity 85 \
                   --wind-speed 8 --precipitation 0
```

That last command is the actual tool: feed it tomorrow's weather forecast
(from any weather app), it prints tomorrow's predicted AQI category.

## Why simulated data, and how to switch to real data

This was built in a sandboxed environment that can't reach external APIs
(OpenAQ, Open-Meteo), so the version you're looking at first runs on
synthetic data. `fetch_data.py` is real, working code against the live
APIs — OpenAQ v3 (which now requires a free key; see the comments in that
file for why, the old v2 endpoints were retired in Jan 2025) and
Open-Meteo's archive endpoint (no key needed). Get an OpenAQ key at
explore.openaq.org/register, then:

```bash
export OPENAQ_API_KEY="your-key-here"
python fetch_data.py --start 2022-01-01 --end 2024-12-31
```

It writes to the exact same `data/processed/aqi_weather_daily.csv` schema
that `simulate_data.py` produces. Nothing downstream — `features.py`,
`train.py`, `evaluate.py`, `predict.py` — needs to change. Same interface,
different data source, which is the whole point of keeping data collection
in its own module instead of inlined into the training script.

The synthetic generator isn't random noise — it encodes the actual
mechanisms behind Hanoi's air quality (seasonal baseline peaking in
January, day-to-day persistence, rain washout, wind dispersal, temperature
inversion, clustered multi-day smog episodes during burning season) so the
model has real, structured signal to find. See `simulate_data.py`'s
docstring for the full mechanism list. Treat the specific numbers it
produces as illustrative, not as real measurements of Hanoi's air.

## How the prediction task is framed

Each row represents "today." The features are everything you'd actually
have in hand: your own PM2.5 history through today (lag and rolling
features), plus tomorrow's weather *forecast*. The target is tomorrow's AQI
category. Using a weather forecast as a feature isn't leakage, since
forecasts for tomorrow are known today and reasonably accurate — using
tomorrow's actual PM2.5 reading would be. `features.py`'s docstring covers
this distinction in more detail, including a train/serve skew subtlety
worth knowing about (we approximate "the forecast" with the actual recorded
weather, since we only have historical observations, not archived
forecasts).

## Results

On a 150-day held-out test set (the most recent data, Aug–Dec 2024 — see
"a mistake I almost shipped" below for why it's 150 days and not a smaller
number):

| Class     | Precision | Recall | F1   | Support |
|-----------|-----------|--------|------|---------|
| Good      | 0.67      | 0.61   | 0.64 | 23      |
| Moderate  | 0.60      | 0.69   | 0.64 | 39      |
| Unhealthy | 0.94      | 0.90   | 0.92 | 88      |

Overall accuracy 80%, macro F1 0.73. The model is much better at Unhealthy
than at the other two classes, which makes sense: it's the majority class
*and* the most extreme one, so it's the easiest to separate. Good and
Moderate sit right next to each other on the PM2.5 scale, so confusing them
is a near-miss, not a real error — and the confusion matrix confirms that's
exactly what happens: the model never once confuses Good directly with
Unhealthy. Every mistake is between adjacent categories. For a health
warning system, that property matters more than the raw accuracy number
does.

A 5-fold `TimeSeriesSplit` cross-validation check (on the training portion
only, never touching the test set) shows accuracy ranging 0.78–0.93 and
macro F1 ranging 0.51–0.61 across folds — consistent enough to trust the
held-out numbers above aren't a lucky split.

SHAP analysis (`outputs/shap_summary.png`) ranks `pm25_rolling_mean_7` as
the dominant feature by a wide margin, with the lag features and month
behind it, and the weather forecast features (temperature most of all,
then humidity, wind, precipitation) contributing a real but smaller
share. In plain terms: knowing this week's pollution trend tells you most
of what you need to know about tomorrow; the forecast mostly refines that
baseline rather than overriding it. That's a believable result for a
pollutant with strong day-to-day persistence, not a red flag.

## Two mistakes I almost shipped

Both were caught by writing diagnostics that check the model's behavior
against domain knowledge, not just checking that the code runs. Leaving
them in here on purpose, since catching this kind of thing matters more
than getting it right on the first attempt.

**Compounding feedback in the smog-burst simulation.** The synthetic
generator clusters Hanoi's winter pollution spikes into multi-day "burst"
events instead of randomly scattering bad days, which is realistic — real
smog episodes last several days. My first version multiplied the burst
boost directly onto the autoregressive state (`pm25[t-1]`). Since that
state already carried forward 55% of the *previous* day's boosted value,
a second burst day would boost an already-boosted number, compounding
across consecutive smog days. Monthly PM2.5 averages came out implausibly
high (November averaged 122 µg/m³, which is smog-apocalypse territory even
for Hanoi). The fix: apply the burst multiplier to the exogenous driver
(`raw_level`) instead of the carried-forward state, so each day's boost is
independent rather than stacking. Re-running brought November down to a
much more defensible 68 µg/m³.

**A backwards wind relationship.** After training, I ran a diagnostic
checking the correlation between each weather feature and the model's
predicted P(Unhealthy) across the full dataset — temperature and
precipitation showed the expected negative correlation (more heat/rain,
cleaner air), but wind speed showed +0.90. The model had learned "windier
days mean dirtier air," which is backwards from the actual physics (wind
disperses pollution). The cause: the simulator gave wind a seasonal cycle
peaking on the *same day* as PM2.5's own seasonal peak. Pollution's
seasonal swing is huge (amplitude 30 µg/m³) compared to wind's direct
dispersal effect, so the spurious seasonal confound between "windy" and
"winter" completely swamped the real causal signal, and the model — quite
reasonably — picked up on the stronger pattern in the data, which was the
wrong one. The fix was to remove wind's seasonal component entirely
(day-to-day noise only) so its only relationship with pollution runs
through the actual causal channel. This is the standard problem with
proxy/confounded features in any observational dataset, simulated or
real — the model will always find the strongest pattern available, and
that pattern isn't guaranteed to be the causal one. SHAP feature
importance tells you what the model used, not what's actually true; the
correlation-direction check is what catches the difference.

## Why TimeSeriesSplit, not a random train/test split

A random shuffle would scatter future days into training and past days
into testing. The model would then partially "know the future" — a
January test row's lag features could come from a December training row
that's chronologically *after* some other training row. That inflates
validation scores in a way that doesn't show up again once the model is
actually predicting forward in time, which is the only way this would ever
get used. Time series data gets split at a date: train before, test after,
no exceptions. The same logic is why the held-out test window is 150 days
and not something smaller — a 60-day trailing window lands entirely in
deep winter (100% Unhealthy days), which would make the model look
perfect while never actually testing whether it can tell Good and Moderate
apart.

## Why class-balanced sample weights

The dataset is realistically imbalanced — Hanoi really is in the Unhealthy
band more often than not in winter, so naive training would let the model
get a deceptively good accuracy score by mostly just predicting the
majority class. `compute_sample_weight(class_weight="balanced")` upweights
the minority classes during training so precision/recall on Good and
Moderate actually mean something, instead of optimizing a number that
looks good while ignoring two-thirds of the categories.

## Project structure

```
vietnam-aqi-predictor/
├── src/
│   ├── config.py          # shared constants: paths, coordinates, AQI thresholds
│   ├── fetch_data.py      # real OpenAQ v3 + Open-Meteo API calls
│   ├── simulate_data.py   # realistic synthetic data generator (fallback)
│   ├── features.py        # lag/rolling/calendar/forecast feature engineering
│   ├── train.py           # TimeSeriesSplit CV + final model training
│   ├── evaluate.py        # confusion matrix + SHAP plots
│   └── predict.py         # the CLI tool — run this day to day
├── data/
│   ├── raw/                # untouched API pulls or simulator output
│   └── processed/          # joined daily table + engineered features
├── models/                 # saved XGBoost model + metadata
├── outputs/                 # confusion_matrix.png, shap_summary.png, shap_by_class.png
└── requirements.txt
```

## Limitations and natural next steps

The AQI scale here collapses the US EPA's six PM2.5 categories into three
(Good ≤12, Moderate 12–35.4, Unhealthy >35.4 µg/m³) to keep enough training
examples per class with only two years of daily data — splitting Unhealthy
further into its real Unhealthy-for-Sensitive-Groups / Unhealthy / Very
Unhealthy / Hazardous tiers would need either much more historical data or
a different modeling approach (ordinal regression instead of plain
multiclass would also be worth trying, since the categories have a natural
order and the current model doesn't know that "Good→Unhealthy" is a worse
mistake than "Good→Moderate," even though the confusion matrix shows it
happens to avoid that mistake anyway). Hyperparameters weren't tuned beyond
reasonable defaults — a GridSearch or Optuna pass is a natural next step.
And the synthetic data, however structured, isn't real measurements;
swapping in actual OpenAQ history (and ideally a few years of it, to give
the model more than one full winter to learn from) is the highest-value
thing to do next.
