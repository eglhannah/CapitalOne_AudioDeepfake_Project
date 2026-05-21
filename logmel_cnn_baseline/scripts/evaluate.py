#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from asv_baseline.training.train import build_model, evaluate, write_predictions
from asv_baseline.training.utils import resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ASVspoof LA baseline checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio-root")
    parser.add_argument("--protocol-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--split",
        choices=["train", "dev"],
        default="dev",
        help="Which configured split to evaluate when paths are not provided.",
    )
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    seed_everything(config["seed"])
    if args.device is not None:
        config["device"] = args.device
    device = resolve_device(config["device"])

    feature_extractor, model = build_model(config, device)
    feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    model.load_state_dict(checkpoint["model"])

    data_cfg = config["data"]
    audio_root = args.audio_root or data_cfg[f"{args.split}_audio_root"]
    protocol_path = args.protocol_path or data_cfg[f"{args.split}_protocol_path"]
    items = parse_la_protocol(
        protocol_path,
        audio_root,
        file_ext=data_cfg.get("file_ext", ".flac"),
        limit=args.limit,
        shuffle_seed=data_cfg.get("shuffle_seed"),
        balanced_limit=False,
    )
    dataset = ASVspoofLADataset(
        items,
        sample_rate=data_cfg["sample_rate"],
        duration_sec=data_cfg["duration_sec"],
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    threshold = args.threshold if args.threshold is not None else config["training"].get("threshold", 0.5)
    metrics, rows = evaluate(feature_extractor, model, loader, nn.BCEWithLogitsLoss(), device, threshold)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    save_json(metrics, output_dir / "metrics_eval.json")
    write_predictions(rows, output_dir / "predictions_eval.csv")
    print(metrics)


if __name__ == "__main__":
    main()
