"""
Trains the XGBoost classifier and saves it.

Two-stage evaluation, on purpose:

  1. TimeSeriesSplit cross-validation across the training portion.
     This is a sanity check, not the final number — it tells you
     whether performance is stable across different historical
     windows, or whether one lucky/unlucky split is driving the
     result.
  2. A single chronological train/test split, where the test set is
     the most recent TEST_HOLDOUT_DAYS days. This produces the
     headline numbers (classification report, confusion matrix, SHAP)
     because it mirrors how the model will actually be used: trained
     on the past, evaluated on days it has never seen, in order.

Why TimeSeriesSplit and never train_test_split(shuffle=True): a
random shuffle would scatter future days into the training set and
past days into the test set. The model would then partially "know
the future" before being asked to predict it — its lag features for
a January test row could come from a December training row that's
chronologically AFTER some other training row. That's leakage, and
it makes validation scores look better than real-world performance
will be. Time series data only gets split one way: a hard line at
some date, train before, test after.
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    FEATURES_FILE, MODEL_FILE, MODEL_META_FILE, AQI_LABELS,
    TEST_HOLDOUT_DAYS, RANDOM_SEED,
)


def make_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(AQI_LABELS),
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        n_jobs=1,  # multi-threaded histogram splits aren't bit-deterministic
                   # even with random_state fixed; pin to 1 thread so reruns
                   # reproduce identical numbers, not just similar ones
    )


def run_cv_check(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    print(f"--- TimeSeriesSplit CV check ({n_splits} folds, training data only) ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold = 0
    for train_idx, val_idx in tscv.split(X):
        fold += 1
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # With only 2 years of data and strong seasonality, an early
        # fold's training prefix can be short enough that a class
        # (e.g. "Good", which doesn't show up until late May here)
        # simply hasn't happened yet. That's not a bug to suppress —
        # it's real information: you can't validate a model on a
        # season it has zero examples of, so skip rather than fudge it.
        if y_tr.nunique() < len(AQI_LABELS):
            seen = sorted(AQI_LABELS[i] for i in y_tr.unique())
            print(f"fold {fold}: skipped — only {len(X_tr)} days of training "
                  f"history, hasn't seen all classes yet (only {seen})")
            continue

        weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        model = make_model()
        model.fit(X_tr, y_tr, sample_weight=weights)
        preds = model.predict(X_val)

        report = classification_report(
            y_val, preds, labels=range(len(AQI_LABELS)), target_names=AQI_LABELS,
            output_dict=True, zero_division=0,
        )
        print(f"fold {fold}: n_train={len(X_tr):3d} n_val={len(X_val):3d}  "
              f"accuracy={report['accuracy']:.3f}  "
              f"macro_f1={report['macro avg']['f1-score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(FEATURES_FILE))
    parser.add_argument("--test-days", type=int, default=TEST_HOLDOUT_DAYS)
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    feature_cols = [c for c in df.columns
                    if c not in ("date", "pm25_next_day", "aqi_category", "target")]

    X = df[feature_cols]
    y = df["target"]

    # Class imbalance check — this dataset skews Unhealthy-heavy,
    # which is realistic for Hanoi winters but means a model that
    # just always predicts "Unhealthy" would score deceptively well
    # on raw accuracy. sample_weight="balanced" forces the model to
    # actually try on the minority classes (Good, Moderate) instead
    # of free-riding on the majority class.
    print("Class balance (full dataset):")
    print(y.value_counts().rename(index=dict(enumerate(AQI_LABELS))))
    print()

    # Final chronological split: last N days = test, everything else = train.
    # Computed before the CV check below so it never sees the held-out days.
    split_idx = len(df) - args.test_days
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = df["date"].iloc[split_idx:]

    run_cv_check(X_train, y_train)
    print()

    print(f"--- Final model: train on first {len(X_train)} days, "
          f"test on last {len(X_test)} days ({test_dates.min().date()} to {test_dates.max().date()}) ---")

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = make_model()
    model.fit(X_train, y_train, sample_weight=weights)

    preds = model.predict(X_test)
    print("\nHeld-out test set performance:")
    print(classification_report(y_test, preds, labels=range(len(AQI_LABELS)),
                                 target_names=AQI_LABELS, zero_division=0))

    model.save_model(str(MODEL_FILE))
    joblib.dump({
        "feature_cols": feature_cols,
        "labels": AQI_LABELS,
        "test_days": args.test_days,
    }, MODEL_META_FILE)

    # Saved separately for evaluate.py / predict.py so neither script
    # has to redo the train/test split or recompute weights.
    X_test.assign(date=test_dates.values, target=y_test.values, pred=preds) \
        .to_csv(MODEL_FILE.parent / "test_predictions.csv", index=False)

    print(f"\nSaved model -> {MODEL_FILE}")
    print(f"Saved metadata -> {MODEL_META_FILE}")


if __name__ == "__main__":
    main()
