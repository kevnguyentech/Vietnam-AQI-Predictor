import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from xgboost import XGBClassifier

from config import MODEL_FILE, MODEL_META_FILE, MODELS_DIR, OUTPUTS_DIR, AQI_LABELS


def load_model() -> tuple[XGBClassifier, dict]:
    meta = joblib.load(MODEL_META_FILE)
    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))
    return model, meta


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(AQI_LABELS)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=AQI_LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("AQI category: predicted vs. actual\n(held-out test days)")
    fig.tight_layout()
    path = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_shap_summary(model: XGBClassifier, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)

    # Multiclass SHAP values come back shaped (n_samples, n_features,
    # n_classes). There's no single "the" SHAP plot for a multiclass
    # model — averaging |SHAP| across classes gives one ranked view of
    # overall feature importance, which is what we want for a single
    # summary figure. Per-class breakdowns are saved separately below.
    values = sv.values
    if values.ndim == 3:
        mean_abs_per_class = np.abs(values).mean(axis=0)  # (n_features, n_classes)
        overall_importance = mean_abs_per_class.mean(axis=1)
    else:
        overall_importance = np.abs(values).mean(axis=0)

    order = np.argsort(overall_importance)[::-1]
    feature_names = np.array(X.columns)[order]
    importance = overall_importance[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(feature_names)), importance[::-1], color="#3B6FA0")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names[::-1])
    ax.set_xlabel("mean |SHAP value| (avg across Good/Moderate/Unhealthy)")
    ax.set_title("What drives the model's predictions")
    fig.tight_layout()
    path = OUTPUTS_DIR / "shap_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Per-class detail plot: a feature can matter a lot for predicting
    # "Unhealthy" specifically while barely mattering for "Good" — the
    # combined chart above can't show that, this one can.
    if values.ndim == 3:
        n_classes = values.shape[2]
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=True)
        for c in range(n_classes):
            class_importance = np.abs(values[:, :, c]).mean(axis=0)
            c_order = np.argsort(class_importance)[::-1][:8]
            axes[c].barh(range(len(c_order)), class_importance[c_order][::-1],
                         color="#3B6FA0")
            axes[c].set_yticks(range(len(c_order)))
            axes[c].set_yticklabels(np.array(X.columns)[c_order][::-1])
            axes[c].set_title(AQI_LABELS[c])
            axes[c].set_xlabel("mean |SHAP value|")
        fig.tight_layout()
        path = OUTPUTS_DIR / "shap_by_class.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    model, meta = load_model()
    test_df = pd.read_csv(MODELS_DIR / "test_predictions.csv")

    feature_cols = meta["feature_cols"]
    X_test = test_df[feature_cols]
    y_true = test_df["target"]
    y_pred = test_df["pred"]

    plot_confusion_matrix(y_true, y_pred)
    plot_shap_summary(model, X_test)


if __name__ == "__main__":
    main()

