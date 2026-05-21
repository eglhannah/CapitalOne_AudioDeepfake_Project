#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asv_baseline.training.train import run_training
from asv_baseline.training.utils import load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ASVspoof LA log-mel CNN baseline.")
    parser.add_argument("--config", default="configs/baseline_logmel_cnn.yaml")
    parser.add_argument("--train-audio-root")
    parser.add_argument("--train-protocol-path")
    parser.add_argument("--dev-audio-root")
    parser.add_argument("--dev-protocol-path")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--no-balanced-limit", action="store_true")
    parser.add_argument("--no-class-weighting", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    data = config["data"]
    training = config["training"]

    for arg_name, cfg_name in [
        ("train_audio_root", "train_audio_root"),
        ("train_protocol_path", "train_protocol_path"),
        ("dev_audio_root", "dev_audio_root"),
        ("dev_protocol_path", "dev_protocol_path"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            data[cfg_name] = value

    if args.limit_train is not None:
        data["limit_train"] = args.limit_train
    if args.limit_dev is not None:
        data["limit_dev"] = args.limit_dev
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.no_balanced_limit:
        data["balanced_limit"] = False
    if args.no_class_weighting:
        training["class_weighting"] = False
    if args.device is not None:
        config["device"] = args.device
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    run_dir = run_training(config, device)
    print(f"Finished training. Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
