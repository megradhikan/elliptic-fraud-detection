"""Shared evaluation helpers so baseline and GNN phases report identical metrics.

The illicit class (label 1) is always the positive class — accuracy is
intentionally not reported since ~2% positive class makes it meaningless.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_score),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_illicit": int((y_true == 1).sum()),
        "n_licit": int((y_true == 0).sum()),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []
    existing.append(metrics)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Appended metrics to {path}")
