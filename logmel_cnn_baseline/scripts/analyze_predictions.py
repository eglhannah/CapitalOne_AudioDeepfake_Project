#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asv_baseline.evaluation.metrics import compute_binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction CSVs from baseline evaluation.")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV.")
    parser.add_argument("--protocol-path", help="Optional ASVspoof protocol file for speaker/attack grouping.")
    parser.add_argument("--output-dir", help="Defaults to the predictions file directory.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-group-size", type=int, default=5)
    return parser.parse_args()


def read_predictions(path: str | Path, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"utterance_id", "label", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            score = float(row["score"])
            label = int(row["label"])
            prediction = int(row.get("prediction", "") or score >= threshold)
            rows.append(
                {
                    **row,
                    "label": label,
                    "score": score,
                    "prediction": prediction,
                    "error": int(prediction != label),
                    "label_name": "spoof" if label == 1 else "bonafide",
                    "prediction_name": "spoof" if prediction == 1 else "bonafide",
                }
            )
    if not rows:
        raise ValueError(f"No rows found in prediction CSV: {path}")
    return rows


def read_protocol_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) < 2:
                continue
            utterance_id = parts[1]
            metadata[utterance_id] = {
                "speaker_id": parts[0],
                "attack_id": parts[3] if len(parts) > 3 else "unknown",
                "protocol_label": parts[-1] if parts else "unknown",
            }
    return metadata


def attach_metadata(rows: list[dict[str, Any]], metadata: dict[str, dict[str, str]]) -> None:
    for row in rows:
        row.update(metadata.get(row["utterance_id"], {}))
        row.setdefault("speaker_id", "unknown")
        row.setdefault("attack_id", "unknown")


def score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    return {
        "count": int(scores.size),
        "mean": finite_float(np.mean(scores)),
        "std": finite_float(np.std(scores)),
        "min": finite_float(np.min(scores)),
        "p05": finite_float(np.quantile(scores, 0.05)),
        "p25": finite_float(np.quantile(scores, 0.25)),
        "p50": finite_float(np.quantile(scores, 0.50)),
        "p75": finite_float(np.quantile(scores, 0.75)),
        "p95": finite_float(np.quantile(scores, 0.95)),
        "max": finite_float(np.max(scores)),
    }


def finite_float(value: float | np.floating) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def group_metrics(
    rows: list[dict[str, Any]],
    key: str,
    threshold: float,
    min_group_size: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)

    output: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in sorted(grouped.items()):
        if len(group_rows) < min_group_size:
            continue
        labels = [row["label"] for row in group_rows]
        scores = [row["score"] for row in group_rows]
        output[group_name] = {
            **compute_binary_metrics(labels, scores, threshold=threshold),
            "score_distribution": score_distribution(group_rows),
        }
    return output


def top_mistakes(rows: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    false_positives = [row for row in rows if row["label"] == 0 and row["prediction"] == 1]
    false_negatives = [row for row in rows if row["label"] == 1 and row["prediction"] == 0]

    false_positives.sort(key=lambda row: row["score"], reverse=True)
    false_negatives.sort(key=lambda row: row["score"])

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "utterance_id": row["utterance_id"],
            "label": row["label_name"],
            "prediction": row["prediction_name"],
            "score": row["score"],
            "speaker_id": row.get("speaker_id", "unknown"),
            "attack_id": row.get("attack_id", "unknown"),
            "path": row.get("path", ""),
        }

    return {
        "highest_score_false_positives": [compact(row) for row in false_positives[:top_k]],
        "lowest_score_false_negatives": [compact(row) for row in false_negatives[:top_k]],
    }


def build_summary(
    rows: list[dict[str, Any]],
    threshold: float,
    top_k: int,
    min_group_size: int,
    has_protocol_metadata: bool,
) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    label_counts = Counter(row["label_name"] for row in rows)
    prediction_counts = Counter(row["prediction_name"] for row in rows)

    summary: dict[str, Any] = {
        "threshold": threshold,
        "has_protocol_metadata": has_protocol_metadata,
        "label_counts": dict(label_counts),
        "prediction_counts": dict(prediction_counts),
        "overall_metrics": compute_binary_metrics(labels, scores, threshold=threshold),
        "score_distribution_all": score_distribution(rows),
        "score_distribution_by_label": {
            "bonafide": score_distribution([row for row in rows if row["label"] == 0]),
            "spoof": score_distribution([row for row in rows if row["label"] == 1]),
        },
        "top_mistakes": top_mistakes(rows, top_k),
    }

    if has_protocol_metadata:
        summary["metrics_by_attack_id"] = group_metrics(rows, "attack_id", threshold, min_group_size)
        summary["metrics_by_speaker_id"] = group_metrics(rows, "speaker_id", threshold, min_group_size)

    return summary


def write_text_report(summary: dict[str, Any], path: str | Path) -> None:
    metrics = summary["overall_metrics"]
    lines = [
        "Prediction Analysis",
        "===================",
        "",
        f"Threshold: {summary['threshold']}",
        f"Rows: {metrics['num_samples']}",
        f"Labels: {summary['label_counts']}",
        f"Predictions: {summary['prediction_counts']}",
        "",
        "Overall Metrics",
        "---------------",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"EER: {metrics['eer']:.4f}",
        f"FPR: {metrics['fpr']:.4f}",
        f"FNR: {metrics['fnr']:.4f}",
        f"Confusion: tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}",
        "",
        "Score Distributions",
        "-------------------",
        f"All: {summary['score_distribution_all']}",
        f"Bonafide: {summary['score_distribution_by_label']['bonafide']}",
        f"Spoof: {summary['score_distribution_by_label']['spoof']}",
        "",
        "Top False Positives",
        "-------------------",
    ]

    for row in summary["top_mistakes"]["highest_score_false_positives"]:
        lines.append(
            f"{row['utterance_id']} score={row['score']:.6f} "
            f"speaker={row['speaker_id']} attack={row['attack_id']}"
        )

    lines.extend(["", "Top False Negatives", "-------------------"])
    for row in summary["top_mistakes"]["lowest_score_false_negatives"]:
        lines.append(
            f"{row['utterance_id']} score={row['score']:.6f} "
            f"speaker={row['speaker_id']} attack={row['attack_id']}"
        )

    if "metrics_by_attack_id" in summary:
        lines.extend(["", "Metrics By Attack ID", "--------------------"])
        for attack_id, group_metrics_row in summary["metrics_by_attack_id"].items():
            lines.append(
                f"{attack_id}: n={group_metrics_row['num_samples']} "
                f"acc={group_metrics_row['accuracy']:.4f} "
                f"auc={group_metrics_row['roc_auc']:.4f} "
                f"eer={group_metrics_row['eer']:.4f} "
                f"fpr={group_metrics_row['fpr']:.4f} "
                f"fnr={group_metrics_row['fnr']:.4f}"
            )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_predictions(args.predictions, threshold=args.threshold)
    has_protocol_metadata = args.protocol_path is not None
    if args.protocol_path:
        attach_metadata(rows, read_protocol_metadata(args.protocol_path))

    summary = build_summary(
        rows,
        threshold=args.threshold,
        top_k=args.top_k,
        min_group_size=args.min_group_size,
        has_protocol_metadata=has_protocol_metadata,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.predictions).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "prediction_analysis.json"
    txt_path = output_dir / "prediction_analysis.txt"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text_report(summary, txt_path)

    print(f"Wrote analysis JSON: {json_path}")
    print(f"Wrote analysis report: {txt_path}")
    print(
        f"accuracy={summary['overall_metrics']['accuracy']:.4f} "
        f"auc={summary['overall_metrics']['roc_auc']:.4f} "
        f"eer={summary['overall_metrics']['eer']:.4f} "
        f"fpr={summary['overall_metrics']['fpr']:.4f} "
        f"fnr={summary['overall_metrics']['fnr']:.4f}"
    )


if __name__ == "__main__":
    main()
