"""Load and validate the pinned AASIST model artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "aasist_v3_best.pt"


def _load_checkpoint() -> dict[str, Any]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must be a mapping containing a 'model' state dict")
    return checkpoint


def read_model_config() -> dict[str, Any]:
    checkpoint = _load_checkpoint()
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must contain an embedded model config")
    if config.get("architecture") != "AASIST":
        raise ValueError("Expected an AASIST checkpoint")
    if config.get("nb_samp") != 64_600:
        raise ValueError("Expected an AASIST checkpoint trained with 64,600-sample windows")
    return config


def read_training_config() -> dict[str, Any]:
    """Return a compatibility wrapper around the active checkpoint config.

    Earlier phases used a separate v2 training-config.json file. The v3
    checkpoint embeds the model config directly, so callers that only need the
    model contract can keep using this helper.
    """

    return {
        "version": "v3_codecaugment",
        "model": read_model_config(),
    }


def load_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    """Return a strictly validated CPU model and non-tensor checkpoint metadata.

    ``weights_only=True`` restricts PyTorch's unpickler to tensor/state-dict
    primitives. The checkpoint is still checksum-verified separately.
    """

    from vendor.aasist.models.AASIST import Model

    checkpoint = _load_checkpoint()
    config = read_model_config()

    model = Model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return model, metadata
