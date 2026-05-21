from __future__ import annotations

import numpy as np


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, sorted_labels.size - 1]

    tps = np.cumsum(sorted_labels == 1)[threshold_indices]
    fps = 1 + threshold_indices - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    tpr = tps / positives if positives else np.zeros_like(tps, dtype=float)
    fpr = fps / negatives if negatives else np.zeros_like(fps, dtype=float)
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return float("nan")

    fpr, tpr, _ = _roc_curve(labels, scores)
    return float(np.trapz(tpr, fpr))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = _roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_binary_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)

    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    metrics: dict[str, float | int] = {
        "accuracy": float(np.mean(labels == preds)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "eer": compute_eer(labels, scores),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_samples": int(len(labels)),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = compute_roc_auc(labels, scores)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics
