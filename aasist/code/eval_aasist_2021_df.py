#!/usr/bin/env python
"""AASIST inference on ASVspoof 2021 DF (deepfake) eval set.

DF metadata format (from keys/DF/CM/trial_metadata.txt) - typical 9-column layout:
  speaker_id  utt_id  source  compression  attack_id  label  trim  variant  phase

If your actual format differs, set the column indices in _COLS below or pass via CLI.

Reports overall EER + per-compression EER + per-attack EER.
Robust against corrupted/unreadable audio files (3-tier loader: torchaudio -> soundfile -> ffmpeg).
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

USER = os.environ["USER"]
BASE = Path(f"/scratch/{USER}/aasist")
CHASE_SRC = BASE / "code/CapitalOne_AudioDeepfake_Project/logmel_cnn_baseline/src"
AASIST_REPO = BASE / "code/aasist"
sys.path.insert(0, str(CHASE_SRC))
sys.path.insert(0, str(AASIST_REPO))

from asv_baseline.evaluation.metrics import compute_binary_metrics, compute_eer
from models.AASIST import Model as AASISTModel

AASIST_CFG = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

NB_SAMP = 64600


@dataclass(frozen=True)
class Item:
    utterance_id: str
    path: Path
    label: int
    label_name: str
    attack_id: str
    compression: str


def parse_2021_df_protocol(metadata_path, audio_root, limit=None, eval_only=True,
                            col_utt=1, col_compression=2, col_attack=4):
    """Parse 2021 DF trial_metadata.txt.

    DF format (13+ columns):
      0:speaker 1:utt_id 2:compression 3:source 4:attack 5:label 6:trim 7:phase 8+:vocoder_meta

    Label and phase detected robustly anywhere in the row (resilient to format drift).
    """
    PHASE_TOKENS = {"eval", "progress", "hidden", "train", "dev"}
    audio_root = Path(audio_root)
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # Find label robustly
            label_name = None
            for tok in parts:
                if tok.lower() in ("bonafide", "spoof"):
                    label_name = tok.lower()
                    break
            if label_name is None:
                continue
            # Find phase robustly (search known phase tokens)
            phase = "unknown"
            for tok in parts:
                if tok.lower() in PHASE_TOKENS:
                    phase = tok.lower()
                    break
            if eval_only and phase != "eval":
                continue
            utt_id = parts[col_utt]
            try:
                compression = parts[col_compression]
            except IndexError:
                compression = "unknown"
            try:
                attack = parts[col_attack]
            except IndexError:
                attack = "unknown"
            items.append(Item(
                utterance_id=utt_id,
                path=audio_root / f"{utt_id}.flac",
                label=0 if label_name == "bonafide" else 1,
                label_name=label_name,
                attack_id=attack,
                compression=compression,
            ))
    if not items:
        raise ValueError(f"No usable rows in {metadata_path}")
    if limit is not None:
        items = items[:limit]
    return items


def robust_load(path):
    """Try torchaudio, then soundfile, then ffmpeg subprocess."""
    err = None
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).numpy().astype(np.float32)
        if sr != 16000:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
            wav = wav_t.squeeze(0).numpy().astype(np.float32)
        return wav, True, None
    except Exception as e:
        err = f"torchaudio: {type(e).__name__}: {e}"
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(round(len(wav) * ratio))
            idx = np.linspace(0, len(wav) - 1, new_len).astype(int)
            wav = wav[idx]
        return wav.astype(np.float32), True, None
    except Exception as e:
        err = f"{err} | soundfile: {type(e).__name__}: {e}"
    try:
        cmd = [
            "ffmpeg", "-loglevel", "quiet", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        wav = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if len(wav) == 0:
            raise RuntimeError("ffmpeg returned empty stream")
        return wav, True, None
    except Exception as e:
        err = f"{err} | ffmpeg: {type(e).__name__}: {e}"
    return np.zeros(NB_SAMP, dtype=np.float32), False, err


def fit_length(wav, n_target=NB_SAMP):
    n = len(wav)
    if n == n_target:
        return wav
    if n > n_target:
        start = (n - n_target) // 2
        return wav[start:start + n_target]
    repeats = (n_target + n - 1) // n
    return np.tile(wav, repeats)[:n_target]


class RobustDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        wav, ok, err = robust_load(item.path)
        wav = fit_length(wav)
        return {
            "waveform": torch.from_numpy(wav.astype(np.float32)),
            "label": torch.tensor(item.label, dtype=torch.long),
            "utterance_id": item.utterance_id,
            "ok": ok,
            "err": err if err else "",
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    # Column indices for DF metadata (override defaults if format differs)
    p.add_argument("--col-utt", type=int, default=1)
    p.add_argument("--col-compression", type=int, default=2)
    p.add_argument("--col-attack", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, bad_log):
    model.eval()
    rows = []
    n_skipped = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["waveform"].to(device, non_blocking=True)
        _, logits = model(x)
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        for utt_id, lbl, score, ok, err in zip(
            batch["utterance_id"],
            batch["label"].cpu().numpy().tolist(),
            scores,
            batch["ok"].cpu().numpy().tolist() if isinstance(batch["ok"], torch.Tensor) else batch["ok"],
            batch["err"],
        ):
            ok_bool = bool(ok)
            if not ok_bool:
                n_skipped += 1
                bad_log.append({"utterance_id": utt_id, "err": str(err)})
                continue
            rows.append({"utterance_id": utt_id, "label": int(lbl), "score": float(score)})
    return rows, n_skipped


def group_eer(rows, items, key_fn):
    utt_to_item = {it.utterance_id: it for it in items}
    bonafide_scores = np.array([r["score"] for r in rows if r["label"] == 0])
    by_group = defaultdict(list)
    for r in rows:
        if r["label"] == 1:
            item = utt_to_item.get(r["utterance_id"])
            if item is None:
                continue
            by_group[key_fn(item)].append(r["score"])
    out = {}
    for group, spoofs in by_group.items():
        spoofs = np.array(spoofs)
        labels = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoofs))])
        scores = np.concatenate([bonafide_scores, spoofs])
        eer = compute_eer(labels, scores)
        out[group] = {"eer": float(eer), "n_spoof": int(len(spoofs))}
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AASISTModel(AASIST_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Checkpoint from epoch {ckpt.get('epoch', '?')}")

    print(f"Parsing 2021 DF metadata: {args.metadata}")
    items = parse_2021_df_protocol(
        args.metadata, args.audio_root, limit=args.limit,
        col_utt=args.col_utt, col_compression=args.col_compression, col_attack=args.col_attack,
    )
    label_counts = Counter(i.label_name for i in items)
    compression_counts = Counter(i.compression for i in items)
    attack_counts = Counter(i.attack_id for i in items if i.label_name == "spoof")
    print(f"Total: {len(items)} {dict(label_counts)}")
    print(f"Compression dist ({len(compression_counts)} unique): {dict(sorted(compression_counts.items()))}")
    print(f"Attack dist ({len(attack_counts)} unique attacks)")
    if len(attack_counts) <= 20:
        print(f"  {dict(sorted(attack_counts.items()))}")
    else:
        sample_attacks = dict(sorted(attack_counts.items())[:10])
        print(f"  First 10: {sample_attacks}")

    ds = RobustDataset(items)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    bad_log = []
    rows, n_skipped = run_inference(model, loader, device, bad_log)
    print(f"\nDecoded successfully: {len(rows)} | Skipped (bad audio): {n_skipped}")

    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    overall = compute_binary_metrics(labels, scores, threshold=0.5)
    print(
        f"\n=== Overall ===\n"
        f"EER={overall['eer']:.4f} | AUC={overall['roc_auc']:.4f} | "
        f"FPR={overall['fpr']:.4f} | FNR={overall['fnr']:.4f} | "
        f"acc={overall['accuracy']:.4f} | n={overall['num_samples']}"
    )

    per_compression = group_eer(rows, items, lambda it: it.compression)
    print("\n=== Per-compression EER ===")
    for cmp in sorted(per_compression.keys()):
        info = per_compression[cmp]
        print(f"  {cmp:14s}: EER={info['eer']:.4f} (n_spoof={info['n_spoof']})")

    per_attack = group_eer(rows, items, lambda it: it.attack_id)
    print(f"\n=== Per-attack EER (top 15 by n_spoof) ===")
    sorted_attacks = sorted(per_attack.items(), key=lambda kv: -kv[1]["n_spoof"])
    for atk, info in sorted_attacks[:15]:
        print(f"  {atk}: EER={info['eer']:.4f} (n_spoof={info['n_spoof']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch", "?"),
        "metadata": args.metadata,
        "n_attempted": len(items),
        "n_decoded": len(rows),
        "n_skipped": n_skipped,
        "label_distribution": dict(label_counts),
        "compression_distribution": dict(compression_counts),
        "attack_distribution": dict(attack_counts),
        "overall": overall,
        "per_compression": per_compression,
        "per_attack": per_attack,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utterance_id", "label", "score"])
        w.writeheader()
        w.writerows(rows)

    if bad_log:
        with (out_dir / "skipped_files.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["utterance_id", "err"])
            w.writeheader()
            w.writerows(bad_log[:1000])

    print(f"\nDONE. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
