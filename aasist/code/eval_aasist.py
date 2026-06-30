#!/usr/bin/env python
"""Inference-only AASIST eval on an arbitrary ASVspoof protocol.

Loads a trained best.pt checkpoint, runs inference over a held-out eval set,
computes overall EER + per-attack-system EER breakdown.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

USER = os.environ["USER"]
BASE = Path(f"/scratch/{USER}/aasist")
CHASE_SRC = BASE / "code/CapitalOne_AudioDeepfake_Project/logmel_cnn_baseline/src"
AASIST_REPO = BASE / "code/aasist"
sys.path.insert(0, str(CHASE_SRC))
sys.path.insert(0, str(AASIST_REPO))

from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from asv_baseline.evaluation.metrics import compute_binary_metrics, compute_eer
from models.AASIST import Model as AASISTModel

AASIST_CFG = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--protocol", required=True, help="Path to .trl.txt eval protocol")
    p.add_argument("--audio-root", required=True, help="Path to flac/ directory")
    p.add_argument("--out-dir", required=True, help="Where to write results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Eval only first N (for smoke test)")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["waveform"].to(device, non_blocking=True)
        _, logits = model(x)
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        for utt_id, lbl, score in zip(
            batch["utterance_id"],
            batch["label"].cpu().numpy().tolist(),
            scores,
        ):
            rows.append({"utterance_id": utt_id, "label": int(lbl), "score": float(score)})
    return rows


def per_attack_eer(rows, items):
    """Compute EER per attack_id. Each attack's spoof scores vs all bonafide scores."""
    utt_to_attack = {it.utterance_id: it.attack_id for it in items}
    bonafide_scores = np.array([r["score"] for r in rows if r["label"] == 0])
    spoof_by_attack = defaultdict(list)
    for r in rows:
        if r["label"] == 1:
            atk = utt_to_attack.get(r["utterance_id"], "UNKNOWN")
            spoof_by_attack[atk].append(r["score"])

    out = {}
    for atk, spoofs in spoof_by_attack.items():
        spoofs = np.array(spoofs)
        labels = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoofs))])
        scores = np.concatenate([bonafide_scores, spoofs])
        eer = compute_eer(labels, scores)
        out[atk] = {"eer": float(eer), "n_spoof": int(len(spoofs))}
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AASISTModel(AASIST_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    src_epoch = ckpt.get("epoch", "?")
    src_metrics = ckpt.get("metrics", {})
    print(f"Checkpoint from epoch {src_epoch} (training dev_eer={src_metrics.get('eer', '?')})")

    print(f"Parsing protocol: {args.protocol}")
    items = parse_la_protocol(args.protocol, args.audio_root, limit=args.limit)
    label_counts = Counter(i.label_name for i in items)
    print(f"Total utterances: {len(items)} {dict(label_counts)}")
    spoof_dist = Counter(i.attack_id for i in items if i.label_name == "spoof")
    print(f"Attack distribution: {dict(sorted(spoof_dist.items()))}")

    ds = ASVspoofLADataset(
        items, sample_rate=16000, duration_sec=64600 / 16000, training=False
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    rows = run_inference(model, loader, device)

    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    overall = compute_binary_metrics(labels, scores, threshold=0.5)
    print(
        f"\nOverall: EER={overall['eer']:.4f} | AUC={overall['roc_auc']:.4f} | "
        f"FPR={overall['fpr']:.4f} | FNR={overall['fnr']:.4f} | "
        f"acc={overall['accuracy']:.4f} | n={overall['num_samples']}"
    )

    per_atk = per_attack_eer(rows, items)
    print("\nPer-attack EER:")
    for atk in sorted(per_atk.keys()):
        info = per_atk[atk]
        print(f"  {atk}: EER={info['eer']:.4f} (n_spoof={info['n_spoof']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": src_epoch,
        "protocol": args.protocol,
        "n_utterances": len(rows),
        "label_distribution": dict(label_counts),
        "attack_distribution": dict(spoof_dist),
        "overall": overall,
        "per_attack": per_atk,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utterance_id", "label", "score"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDONE. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
