"""Phase 2: feature-only baselines (XGBoost, Random Forest) on the time split.

Class imbalance is handled via scale_pos_weight / class_weight='balanced'
rather than resampling, so the label distribution stays representative for
any later graph-structure comparison (resampling would distort which nodes'
neighborhoods get seen).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data import ILLICIT, TEST_TIME_STEPS, TRAIN_TIME_STEPS, VAL_TIME_STEPS, load_raw
from src.metrics import compute_metrics, save_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def get_splits(data):
    labeled = data.labeled_mask()
    feature_cols = data.feature_cols

    def subset(time_steps):
        mask = labeled & data.features["time_step"].isin(time_steps)
        X = data.features.loc[mask, feature_cols].to_numpy()
        y = data.labels.loc[mask].to_numpy()
        return X, y

    X_train, y_train = subset(TRAIN_TIME_STEPS)
    X_val, y_val = subset(VAL_TIME_STEPS)
    X_test, y_test = subset(TEST_TIME_STEPS)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_xgb(X_train, y_train, **params) -> XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == ILLICIT).sum(), 1)
    defaults = dict(
        max_depth=6,
        n_estimators=300,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=0,
    )
    defaults.update(params)
    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train, **params) -> RandomForestClassifier:
    defaults = dict(n_estimators=300, max_depth=None, class_weight="balanced", random_state=0, n_jobs=-1)
    defaults.update(params)
    model = RandomForestClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def sweep_xgb(X_train, y_train, X_val, y_val, n_trials: int = 25) -> dict:
    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 50.0),
        )
        model = train_xgb(X_train, y_train, **params)
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return compute_metrics(y_val, preds, proba)["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Best XGBoost val F1={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def plot_feature_importance(model: XGBClassifier, feature_cols: list[str], out_path: Path) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Top 20 features — illicit vs. licit")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run an optuna sweep over XGBoost hyperparameters")
    parser.add_argument("--n-trials", type=int, default=25)
    args = parser.parse_args()

    data = load_raw()
    feature_cols = data.feature_cols
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(data)
    print(f"train={len(y_train)} (illicit={int((y_train == ILLICIT).sum())})  "
          f"val={len(y_val)} (illicit={int((y_val == ILLICIT).sum())})  "
          f"test={len(y_test)} (illicit={int((y_test == ILLICIT).sum())})")

    xgb_params = {}
    if args.sweep:
        xgb_params = sweep_xgb(X_train, y_train, X_val, y_val, n_trials=args.n_trials)

    xgb_model = train_xgb(X_train, y_train, **xgb_params)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_proba >= 0.5).astype(int)
    xgb_metrics = {"model": "xgboost", **compute_metrics(y_test, xgb_preds, xgb_proba)}
    print("XGBoost test:", json.dumps(xgb_metrics, indent=2))
    save_metrics(xgb_metrics, RESULTS_DIR / "baseline_metrics.json")
    plot_feature_importance(xgb_model, feature_cols, RESULTS_DIR / "figures" / "xgb_feature_importance.png")

    rf_model = train_rf(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = (rf_proba >= 0.5).astype(int)
    rf_metrics = {"model": "random_forest", **compute_metrics(y_test, rf_preds, rf_proba)}
    print("Random Forest test:", json.dumps(rf_metrics, indent=2))
    save_metrics(rf_metrics, RESULTS_DIR / "baseline_metrics.json")


if __name__ == "__main__":
    main()
