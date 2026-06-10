#!/usr/bin/env python
"""Recompute per-compression EER for 2021 DF from predictions.csv + metadata.

Run AFTER the DF eval finishes. Reads the predictions.csv (which has utterance_id
+ score) and re-groups by compression column to produce the per-compression breakdown
that the buggy eval script missed.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

USER = os.environ["USER"]
BASE = Path(f"/scratch/{USER}/aasist")
CHASE_SRC = BASE / "code/CapitalOne_AudioDeepfake_Project/logmel_cnn_baseline/src"
sys.path.insert(0, str(CHASE_SRC))

from asv_baseline.evaluation.metrics import compute_binary_metrics, compute_eer


def load_metadata_compression(metadata_path):
    """Return dict: utterance_id -> compression (column 2 of trial_metadata.txt)."""
    out = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            utt_id = parts[1]
            compression = parts[2]
            out[utt_id] = compression
    return out


def main():
    predictions_csv = BASE / "outputs/eval/2021_DF_eval/predictions.csv"
    metadata_path = BASE / "data/2021/keys/DF/CM/trial_metadata.txt"
    out_json = BASE / "outputs/eval/2021_DF_eval/per_compression_eer.json"

    print(f"Loading metadata: {metadata_path}")
    utt_to_compression = load_metadata_compression(metadata_path)
    print(f"Metadata covers {len(utt_to_compression)} utterances")

    print(f"Loading predictions: {predictions_csv}")
    rows = []
    with open(predictions_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "utterance_id": row["utterance_id"],
                "label": int(row["label"]),
                "score": float(row["score"]),
            })
    print(f"Predictions: {len(rows)}")

    # Overall sanity check
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    overall = compute_binary_metrics(labels, scores, threshold=0.5)
    print(f"\n=== Overall (from predictions.csv) ===")
    print(f"EER={overall['eer']:.4f} | AUC={overall['roc_auc']:.4f} | n={overall['num_samples']}")

    # Per-compression
    bonafide_scores = np.array([r["score"] for r in rows if r["label"] == 0])
    spoof_by_compression = defaultdict(list)
    missing = 0
    for r in rows:
        if r["label"] == 1:
            cmp = utt_to_compression.get(r["utterance_id"])
            if cmp is None:
                missing += 1
                continue
            spoof_by_compression[cmp].append(r["score"])

    if missing:
        print(f"WARN: {missing} spoof predictions had no metadata match")

    per_compression = {}
    for cmp, spoofs in spoof_by_compression.items():
        spoofs = np.array(spoofs)
        lbl = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoofs))])
        scr = np.concatenate([bonafide_scores, spoofs])
        eer = compute_eer(lbl, scr)
        per_compression[cmp] = {"eer": float(eer), "n_spoof": int(len(spoofs))}

    print("\n=== Per-compression EER ===")
    for cmp in sorted(per_compression.keys()):
        info = per_compression[cmp]
        print(f"  {cmp:14s}: EER={info['eer']:.4f} (n_spoof={info['n_spoof']})")

    summary = {"overall": overall, "per_compression": per_compression}
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
