"""Small evaluation helpers: cost-based threshold choice and a readable confusion summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def confusion_at(y_true, proba, threshold: float) -> dict[str, float]:
    """Confusion-matrix counts and the main metrics when predicting ``proba >= threshold``."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def threshold_table(y_true, proba, cost_fp: float, cost_fn: float,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """One row per threshold with counts, metrics and the total cost ``FP·cost_fp + FN·cost_fn``."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [confusion_at(y_true, proba, t) for t in thresholds]
    table = pd.DataFrame(rows)
    table["cost"] = table["FP"] * cost_fp + table["FN"] * cost_fn
    return table


def best_threshold(y_true, proba, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest total cost (ties → the one closest to 0.5)."""
    table = threshold_table(y_true, proba, cost_fp, cost_fn)
    best = table[table["cost"] == table["cost"].min()].copy()
    best["dist"] = (best["threshold"] - 0.5).abs()
    return float(best.sort_values("dist")["threshold"].iloc[0])
