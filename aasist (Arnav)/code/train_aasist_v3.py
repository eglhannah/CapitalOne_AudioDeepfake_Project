#!/usr/bin/env python
"""Train AASIST v3 on ASVspoof 2019 LA with real codec round trips + RawBoost noise.

What v3 changes vs v2:
    - Replaces RawBoost's closed-form mu-law/A-law codec simulation (p_codec)
      with real codec round trips (gsm / alaw / ulaw / g722 / opus) sampled from
      a pre-computed cache built by precompute_codec_train.py.
    - Keeps RawBoost's three noise transforms (LnL, ISD, SSI) at default
      probabilities. RawBoost's own p_codec is forced to 0.
    - run_name defaults to aasist_v3_codecreal_<timestamp>.

Why:
    v2 regressed on 2021 LA (5.67% -> 8.01%) while improving on 2021 DF
    (22.95% -> 17.20%). Hypothesis: RawBoost's closed-form codec simulation
    does not match real Asterisk PBX codec transmission. v3 tests this
    hypothesis by substituting real codec round trips at training time.

Prereq:
    python precompute_codec_train.py     # ~1 hour, ~20 GB on /scratch
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
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
CODEC_BASE_DEFAULT = BASE / "data/LA_codec_train"

sys.path.insert(0, str(CHASE_SRC))
sys.path.insert(0, str(AASIST_REPO))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from asv_baseline.evaluation.metrics import compute_binary_metrics
from models.AASIST import Model as AASISTModel
from rawboost import RawBoostAugment
from codec_aug import CodecAugment

AASIST_CFG = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class CodecPlusRawBoostDataset(ASVspoofLADataset):
    """Apply real codec round-trip substitution then RawBoost noise transforms.

    Order matters: codec round trip first (changes channel response), then
    additive noises on top (models environment / channel imperfection added
    after codec transmission).
    """

    def __init__(self, items, *args, codec_aug=None, rawboost_aug=None, **kwargs):
        super().__init__(items, *args, **kwargs)
        self._items = items
        self.codec_aug = codec_aug
        self.rawboost_aug = rawboost_aug

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not self.training:
            return sample

        wav = sample["waveform"].cpu().numpy()
        if self.codec_aug is not None:
            item = self._items[idx]
            wav, _ = self.codec_aug.maybe_substitute(item.path, wav, target_len=len(wav))
        if self.rawboost_aug is not None:
            wav = self.rawboost_aug(wav)
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
    # Codec augmentation
    p.add_argument("--codec-base", type=str, default=str(CODEC_BASE_DEFAULT))
    p.add_argument("--p-codec", type=float, default=0.5,
                   help="Probability of substituting a real codec round-tripped version per sample")
    p.add_argument("--codecs", nargs="+", default=["alaw", "ulaw", "g722", "opus"])
    # RawBoost noise transforms (codec sim DISABLED in v3 — handled by codec_aug above)
    p.add_argument("--p-lnl", type=float, default=0.5)
    p.add_argument("--p-isd", type=float, default=0.5)
    p.add_argument("--p-ssi", type=float, default=0.5)
    return p.parse_args()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(args):
    proto_dir = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    tr_items = parse_la_protocol(
        proto_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        DATA_ROOT / "ASVspoof2019_LA_train/flac",
        limit=args.limit_train, shuffle_seed=args.seed,
    )
    dv_items = parse_la_protocol(
        proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        DATA_ROOT / "ASVspoof2019_LA_dev/flac",
        limit=args.limit_dev, shuffle_seed=args.seed,
    )
    print(f"train n={len(tr_items)} {dict(Counter(i.label_name for i in tr_items))}")
    print(f"dev   n={len(dv_items)} {dict(Counter(i.label_name for i in dv_items))}")

    codec_aug = CodecAugment(
        codec_base=args.codec_base,
        codecs=args.codecs,
        p_codec=args.p_codec,
        seed=None,
    )
    rb_aug = RawBoostAugment(
        sample_rate=16000,
        p_lnl=args.p_lnl, p_isd=args.p_isd, p_ssi=args.p_ssi,
        p_codec=0.0,
        seed=None,
    )
    print(f"CodecAugment ENABLED: p_codec={args.p_codec} codecs={args.codecs} base={args.codec_base}")
    print(f"RawBoost noise ENABLED: p_lnl={args.p_lnl} p_isd={args.p_isd} p_ssi={args.p_ssi} p_codec=0 (v3)")

    tr_ds = CodecPlusRawBoostDataset(
        tr_items, sample_rate=16000, duration_sec=64600 / 16000,
        training=True, codec_aug=codec_aug, rawboost_aug=rb_aug,
    )
    dv_ds = ASVspoofLADataset(
        dv_items, sample_rate=16000, duration_sec=64600 / 16000, training=False,
    )

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dv_ld = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    return tr_ld, dv_ld


def train_epoch(model, loader, optim, criterion, device, ep, total_ep):
    model.train()
    tot, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train ep{ep:03d}/{total_ep}", leave=False)
    for batch in pbar:
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["label"].long().to(device, non_blocking=True)
        _, logits = model(x)
        loss = criterion(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=tot / max(n, 1))
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot, n = 0.0, 0
    labels, scores = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["label"].long().to(device, non_blocking=True)
        _, logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
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
    run_name = args.run_name or f"aasist_v3_codecreal_{ts}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "model": AASIST_CFG, "version": "v3_codecreal"}, indent=2)
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
