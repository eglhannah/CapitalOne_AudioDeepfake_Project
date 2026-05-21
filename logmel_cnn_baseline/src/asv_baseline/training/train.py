from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from asv_baseline.data.asvspoof_dataset import ASVspoofLADataset, parse_la_protocol
from asv_baseline.evaluation.metrics import compute_binary_metrics
from asv_baseline.features.logmel import LogMelSpectrogram
from asv_baseline.models.cnn import LogMelCNN
from asv_baseline.training.utils import make_run_dir, save_json


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_items = parse_la_protocol(
        data_cfg["train_protocol_path"],
        data_cfg["train_audio_root"],
        file_ext=data_cfg.get("file_ext", ".flac"),
        limit=data_cfg.get("limit_train"),
        shuffle_seed=data_cfg.get("shuffle_seed"),
        balanced_limit=data_cfg.get("balanced_limit", False),
    )
    dev_items = parse_la_protocol(
        data_cfg["dev_protocol_path"],
        data_cfg["dev_audio_root"],
        file_ext=data_cfg.get("file_ext", ".flac"),
        limit=data_cfg.get("limit_dev"),
        shuffle_seed=data_cfg.get("shuffle_seed"),
        balanced_limit=data_cfg.get("balanced_limit", False),
    )
    train_counts = Counter(item.label_name for item in train_items)
    dev_counts = Counter(item.label_name for item in dev_items)
    print(f"Loaded train items: {len(train_items)} {dict(train_counts)}")
    print(f"Loaded dev items: {len(dev_items)} {dict(dev_counts)}")

    train_ds = ASVspoofLADataset(
        train_items,
        sample_rate=data_cfg["sample_rate"],
        duration_sec=data_cfg["duration_sec"],
        training=True,
    )
    dev_ds = ASVspoofLADataset(
        dev_items,
        sample_rate=data_cfg["sample_rate"],
        duration_sec=data_cfg["duration_sec"],
        training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, dev_loader


def compute_pos_weight(loader: DataLoader, device: torch.device) -> torch.Tensor:
    labels = [item.label for item in loader.dataset.items]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(negatives / positives, dtype=torch.float32, device=device)


def build_model(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, nn.Module]:
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]

    feature_extractor = LogMelSpectrogram(
        sample_rate=data_cfg["sample_rate"],
        n_mels=feat_cfg["n_mels"],
        n_fft=feat_cfg.get("n_fft", 1024),
        win_length_ms=feat_cfg["win_length_ms"],
        hop_length_ms=feat_cfg["hop_length_ms"],
        f_min=feat_cfg.get("f_min", 20),
        f_max=feat_cfg.get("f_max", 7600),
    ).to(device)
    model = LogMelCNN(
        channels=model_cfg.get("channels", [32, 64, 128, 128]),
        dropout=model_cfg.get("dropout", 0.25),
    ).to(device)
    return feature_extractor, model


def train_one_epoch(
    feature_extractor: nn.Module,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_every_steps: int,
    grad_clip_norm: float | None,
) -> float:
    feature_extractor.train()
    model.train()
    running_loss = 0.0
    num_examples = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        waveform = batch["waveform"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(feature_extractor(waveform))
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = waveform.size(0)
        running_loss += loss.item() * batch_size
        num_examples += batch_size
        if step % log_every_steps == 0:
            progress.set_postfix(loss=running_loss / max(num_examples, 1))

    return running_loss / max(num_examples, 1)


def _score_summary(rows: list[dict[str, str | float | int]]) -> str:
    scores = [float(row["score"]) for row in rows]
    spoof_scores = [float(row["score"]) for row in rows if int(row["label"]) == 1]
    bona_scores = [float(row["score"]) for row in rows if int(row["label"]) == 0]

    def summarize(values: list[float]) -> str:
        if not values:
            return "n/a"
        return f"mean={sum(values) / len(values):.4f}, min={min(values):.4f}, max={max(values):.4f}"

    return (
        f"scores[{summarize(scores)}] | "
        f"bonafide[{summarize(bona_scores)}] | "
        f"spoof[{summarize(spoof_scores)}]"
    )


@torch.no_grad()
def evaluate(
    feature_extractor: nn.Module,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float | int], list[dict[str, str | float | int]]]:
    feature_extractor.eval()
    model.eval()
    total_loss = 0.0
    num_examples = 0
    labels: list[int] = []
    scores: list[float] = []
    rows: list[dict[str, str | float | int]] = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        waveform = batch["waveform"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        logits = model(feature_extractor(waveform))
        loss = criterion(logits, target)
        probs = torch.sigmoid(logits)

        batch_size = waveform.size(0)
        total_loss += loss.item() * batch_size
        num_examples += batch_size

        for utterance_id, path, label, score in zip(
            batch["utterance_id"],
            batch["path"],
            target.cpu().numpy().astype(int).tolist(),
            probs.cpu().numpy().tolist(),
            strict=True,
        ):
            labels.append(int(label))
            scores.append(float(score))
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "path": path,
                    "label": int(label),
                    "score": float(score),
                    "prediction": int(float(score) >= threshold),
                }
            )

    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    metrics["loss"] = total_loss / max(num_examples, 1)
    return metrics, rows


def write_predictions(rows: list[dict[str, str | float | int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utterance_id", "path", "label", "score", "prediction"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: str | Path,
    feature_extractor: nn.Module,
    model: nn.Module,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float | int],
) -> None:
    torch.save(
        {
            "feature_extractor": feature_extractor.state_dict(),
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def run_training(config: dict[str, Any], device: torch.device) -> Path:
    run_dir = make_run_dir(config["training"]["output_dir"])
    save_json(config, run_dir / "config.json")
    print(f"Writing run artifacts to: {run_dir}")
    print(f"Using device: {device}")

    train_loader, dev_loader = build_dataloaders(config)
    feature_extractor, model = build_model(config, device)

    pos_weight = compute_pos_weight(train_loader, device) if config["training"].get("class_weighting", True) else None
    if pos_weight is not None:
        print(f"Using BCE pos_weight={float(pos_weight):.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    history = []
    best_eer = float("inf")
    has_best_checkpoint = False
    threshold = config["training"].get("threshold", 0.5)

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(
            feature_extractor,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            log_every_steps=config["training"].get("log_every_steps", 25),
            grad_clip_norm=config["training"].get("grad_clip_norm"),
        )
        dev_metrics, dev_rows = evaluate(
            feature_extractor,
            model,
            dev_loader,
            criterion,
            device,
            threshold=threshold,
        )
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "dev": dev_metrics}
        history.append(epoch_record)
        save_json(history, run_dir / "history.json")
        save_json(dev_metrics, run_dir / "metrics_latest.json")
        write_predictions(dev_rows, run_dir / "predictions_dev_latest.csv")
        save_checkpoint(
            run_dir / "latest.pt",
            feature_extractor,
            model,
            config,
            epoch,
            dev_metrics,
        )

        print(
            "Epoch "
            f"{epoch}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={float(dev_metrics['loss']):.4f} | "
            f"acc={float(dev_metrics['accuracy']):.4f} | "
            f"auc={float(dev_metrics['roc_auc']):.4f} | "
            f"eer={float(dev_metrics['eer']):.4f} | "
            f"fpr={float(dev_metrics['fpr']):.4f} | "
            f"fnr={float(dev_metrics['fnr']):.4f}"
        )
        print(_score_summary(dev_rows))

        current_eer = float(dev_metrics["eer"])
        improved = math.isfinite(current_eer) and current_eer < best_eer
        if improved or not has_best_checkpoint:
            if math.isfinite(current_eer):
                best_eer = current_eer
            save_checkpoint(
                run_dir / "best.pt",
                feature_extractor,
                model,
                config,
                epoch,
                dev_metrics,
            )
            save_json(dev_metrics, run_dir / "metrics_best.json")
            write_predictions(dev_rows, run_dir / "predictions_dev_best.csv")
            has_best_checkpoint = True
            print(f"Saved best checkpoint at epoch {epoch}: {run_dir / 'best.pt'}")

    print(f"Finished training. Final artifacts are in: {run_dir}")
    return run_dir
