"""
Shared model loading logic for predict.py and evaluate.py.
Keeping it here means one place to update if the save format changes.
"""

import sys

import joblib
from xgboost import XGBClassifier

from config import MODEL_FILE, MODEL_META_FILE


def load_model() -> tuple[XGBClassifier, dict]:
    if not MODEL_META_FILE.exists():
        sys.exit(
            f"Model metadata not found at {MODEL_META_FILE}.\n"
            "Run: python src/train.py"
        )
    if not MODEL_FILE.exists():
        sys.exit(
            f"Model file not found at {MODEL_FILE}.\n"
            "Run: python src/train.py"
        )
    meta = joblib.load(MODEL_META_FILE)
    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))
    return model, meta