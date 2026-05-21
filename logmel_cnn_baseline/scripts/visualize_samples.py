#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from asv_baseline.features.logmel import LogMelSpectrogram
from asv_baseline.training.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save quick log-mel spectrogram previews.")
    parser.add_argument("--config", default="configs/baseline_logmel_cnn.yaml")
    parser.add_argument("--audio-root")
    parser.add_argument("--protocol-path")
    parser.add_argument("--output-dir", default="outputs/spectrograms")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=["train", "dev"],
        default="train",
        help="Which configured split to visualize when paths are not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    audio_root = args.audio_root or data_cfg[f"{args.split}_audio_root"]
    protocol_path = args.protocol_path or data_cfg[f"{args.split}_protocol_path"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = parse_la_protocol(
        protocol_path,
        audio_root,
        file_ext=data_cfg.get("file_ext", ".flac"),
        limit=args.num_samples,
        shuffle_seed=data_cfg.get("shuffle_seed"),
        balanced_limit=True,
    )
    dataset = ASVspoofLADataset(
        items,
        sample_rate=data_cfg["sample_rate"],
        duration_sec=data_cfg["duration_sec"],
        training=False,
    )
    feat_cfg = config["features"]
    extractor = LogMelSpectrogram(
        sample_rate=data_cfg["sample_rate"],
        n_mels=feat_cfg["n_mels"],
        n_fft=feat_cfg.get("n_fft", 1024),
        win_length_ms=feat_cfg["win_length_ms"],
        hop_length_ms=feat_cfg["hop_length_ms"],
        f_min=feat_cfg.get("f_min", 20),
        f_max=feat_cfg.get("f_max", 7600),
    )

    for idx in range(len(dataset)):
        sample = dataset[idx]
        features = extractor(sample["waveform"].unsqueeze(0)).squeeze().numpy()
        plt.figure(figsize=(8, 4))
        plt.imshow(features, aspect="auto", origin="lower")
        plt.title(f"{sample['utterance_id']} label={int(sample['label'].item())}")
        plt.xlabel("Frame")
        plt.ylabel("Mel bin")
        plt.colorbar(label="normalized log-mel")
        plt.tight_layout()
        plt.savefig(output_dir / f"{sample['utterance_id']}.png", dpi=150)
        plt.close()

    print(f"Wrote {len(dataset)} previews to {output_dir}")


if __name__ == "__main__":
    main()
