#!/usr/bin/env python
"""Train AASIST v2 on ASVspoof 2019 LA with RawBoost data augmentation.

Mirrors train_aasist.py but wraps the training dataset in a RawBoost augmenter.
Eval dataset (dev set) is left untouched — never augment eval data.

Key changes from v1:
    - New --rawboost flag (default ON for v2)
    - Configurable per-augmentation probabilities via CLI
    - Wraps Chase's ASVspoofLADataset with RawBoostASVspoofDataset for training only
    - run_name defaults to aasist_v2_rawboost_<timestamp>

Goal: close the cross-domain gap on 2021 LA / 2021 DF eval sets.
Target: 2021 DF EER 22.95% → ≤15%.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

USER = os.environ["USER"]
BASE = Path(f"/scratch/{USER}/aasist")
CHASE_SRC = BASE / "code/CapitalOne_AudioDeepfake_Project/logmel_cnn_baseline/src"
AASIST_REPO = BASE / "code/aasist"
DATA_ROOT = BASE / "data/LA"
OUTPUT_DIR = BASE / "outputs/runs"
sys.path.insert(0, str(CHASE_SRC))
sys.path.insert(0, str(AASIST_REPO))

# Add the directory containing rawboost.py (this script's directory) to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from asv_baseline.evaluation.metrics import compute_binary_metrics
from models.AASIST import Model as AASISTModel
from rawboost import RawBoostAugment

AASIST_CFG = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class RawBoostASVspoofDataset(ASVspoofLADataset):
    """Thin subclass that applies RawBoost augmentation in training mode."""

    def __init__(self, *args, augment: RawBoostAugment | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.augment is not None and self.training:
            wav = sample["waveform"].cpu().numpy()
            wav = self.augment(wav)
            sample["waveform"] = torch.from_numpy(wav.astype(np.float32))
        return sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-dev", type=int, default=None)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-name", type=str, default=None)
    # RawBoost flags
    p.add_argument("--rawboost", action="store_true", default=True,
                   help="Enable RawBoost augmentation during training (default: True for v2)")
    p.add_argument("--no-rawboost", dest="rawboost", action="store_false",
                   help="Disable RawBoost (degenerates to v1 baseline)")
    p.add_argument("--p-lnl", type=float, default=0.5, help="Probability of linear convolutive noise")
    p.add_argument("--p-isd", type=float, default=0.5, help="Probability of impulsive noise")
    p.add_argument("--p-ssi", type=float, default=0.5, help="Probability of stationary noise")
    p.add_argument("--p-codec", type=float, default=0.5, help="Probability of codec simulation")
    return p.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(args):
    proto_dir = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    tr_items = parse_la_protocol(proto_dir / "ASVspoof2019.LA.cm.train.trn.txt",
                                 DATA_ROOT / "ASVspoof2019_LA_train/flac",
                                 limit=args.limit_train, shuffle_seed=args.seed)
    dv_items = parse_la_protocol(proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
                                 DATA_ROOT / "ASVspoof2019_LA_dev/flac",
                                 limit=args.limit_dev, shuffle_seed=args.seed)
    print(f"train n={len(tr_items)} {dict(Counter(i.label_name for i in tr_items))}")
    print(f"dev   n={len(dv_items)} {dict(Counter(i.label_name for i in dv_items))}")

    augment = None
    if args.rawboost:
        # Note: seed=None so each DataLoader worker gets fresh stochasticity per epoch.
        # This is desirable for augmentation diversity.
        augment = RawBoostAugment(
            sample_rate=16000,
            p_lnl=args.p_lnl, p_isd=args.p_isd, p_ssi=args.p_ssi, p_codec=args.p_codec,
            seed=None,
        )
        print(f"RawBoost ENABLED: p_lnl={args.p_lnl} p_isd={args.p_isd} "
              f"p_ssi={args.p_ssi} p_codec={args.p_codec}")
    else:
        print("RawBoost DISABLED (v1 baseline mode)")

    tr_ds = RawBoostASVspoofDataset(
        tr_items, sample_rate=16000, duration_sec=64600 / 16000,
        training=True, augment=augment,
    )
    # Eval: no augmentation, ever.
    dv_ds = ASVspoofLADataset(
        dv_items, sample_rate=16000, duration_sec=64600 / 16000, training=False,
    )

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dv_ld = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    return tr_ld, dv_ld


def train_epoch(model, loader, optim, criterion, device, ep, total_ep):
    model.train(); tot, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train ep{ep:03d}/{total_ep}", leave=False)
    for batch in pbar:
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["label"].long().to(device, non_blocking=True)
        _, logits = model(x)
        loss = criterion(logits, y)
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        bs = x.size(0); tot += loss.item() * bs; n += bs
        pbar.set_postfix(loss=tot / max(n, 1))
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); tot, n = 0.0, 0; labels, scores = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["label"].long().to(device, non_blocking=True)
        _, logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        bs = x.size(0); tot += loss.item() * bs; n += bs
        labels.extend(y.cpu().numpy().tolist())
        scores.extend(probs.cpu().numpy().tolist())
    m = compute_binary_metrics(labels, scores, threshold=0.5)
    m["loss"] = tot / max(n, 1)
    return m


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tr_ld, dv_ld = build_loaders(args)

    model = AASISTModel(AASIST_CFG).to(device)
    print(f"AASIST params: {sum(p.numel() for p in model.parameters()):,}")

    weight = torch.FloatTensor([0.9, 0.1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay,
                             betas=(0.9, 0.999), amsgrad=False)
    total_steps = args.epochs * len(tr_ld)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=5e-6)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "rawboost" if args.rawboost else "no_rawboost"
    run_name = args.run_name or f"aasist_v2_{suffix}_{ts}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "model": AASIST_CFG, "version": "v2_rawboost"}, indent=2)
    )

    history, best_eer = [], float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr_ld, optim, criterion, device, ep, args.epochs)
        dv = evaluate(model, dv_ld, criterion, device)
        sched.step()
        history.append({"epoch": ep, "train_loss": tr_loss,
                        **{f"dev_{k}": v for k, v in dv.items()}})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"ep{ep:03d}/{args.epochs} | tr_loss={tr_loss:.4f} | "
              f"dv_loss={dv['loss']:.4f} | dv_eer={dv['eer']:.4f} | "
              f"dv_auc={dv['roc_auc']:.4f} | dv_acc={dv['accuracy']:.4f}")
        ckpt = {"model": model.state_dict(), "epoch": ep, "metrics": dv,
                "config": AASIST_CFG, "args": vars(args)}
        torch.save(ckpt, run_dir / "latest.pt")
        if math.isfinite(dv["eer"]) and dv["eer"] < best_eer:
            best_eer = dv["eer"]
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  -> NEW BEST: ep{ep} dev_eer={best_eer:.4f}")

    print(f"\nDONE. Best dev EER: {best_eer:.4f}. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
