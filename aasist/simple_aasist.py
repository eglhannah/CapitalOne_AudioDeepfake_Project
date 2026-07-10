"""Simplest working AASIST v3 loader for prediction + SHAP feature extraction.

Loads the AASIST v3 checkpoint from HuggingFace, puts it in eval mode, and
exposes both the pooled embedding (for SHAP) and the softmax spoof probability
(the model output SHAP explains).

Mirrors the pattern of w2v/simple_model.py so both branches load the same way.

Usage:
    from aasist.simple_aasist import load_aasist_v3, predict
    model = load_aasist_v3()
    out = predict(model, audio)  # audio shape (B, 64600), 16 kHz float32
    print(out["spoof_prob"], out["embedding"].shape)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
_AASIST_VENDOR = REPO_ROOT / "deployment" / "aasist_lambda" / "vendor" / "aasist"
sys.path.insert(0, str(_AASIST_VENDOR))
from models.AASIST import Model as AASISTModel  # noqa: E402

AASIST_CFG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

HF_REPO_V3 = "arnavjain321/aasist-v3-codecaugment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_aasist_v3(repo_id: str = HF_REPO_V3, device: str = DEVICE):
    """Return AASIST v3 loaded from HuggingFace in eval mode.

    Args:
        repo_id:  swap for arnavjain321/aasist-v1-baseline or -v2-rawboost to load other versions
        device:   "cuda" or "cpu"

    Returns:
        torch.nn.Module in eval mode on the requested device
    """
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="aasist_v3_best.pt")
    model = AASISTModel(AASIST_CFG).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, audio: torch.Tensor) -> dict:
    """Score AASIST on a batch of audio.

    Args:
        model:  AASIST module returned by load_aasist_v3
        audio:  torch.Tensor shape (B, 64600), 16 kHz mono float32 in [-1, 1].
                Crop or repeat-pad your waveform to exactly 64600 samples per clip.

    Returns:
        dict with keys:
            embedding  (B, embed_dim)  pooled features - SHAP explains this
            logits     (B, 2)          [bonafide_logit, spoof_logit]
            spoof_prob (B,)            P(spoof), softmax over logits then index 1
    """
    device = next(model.parameters()).device
    audio = audio.to(device)
    embedding, logits = model(audio)
    spoof_prob = F.softmax(logits, dim=-1)[:, 1]
    return {"embedding": embedding, "logits": logits, "spoof_prob": spoof_prob}


def _smoke_test():
    print(f"Loading AASIST v3 from {HF_REPO_V3} on {DEVICE}...")
    model = load_aasist_v3()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Params: {n_params:,}")
    audio = torch.randn(2, 64600)
    out = predict(model, audio)
    print(f"embedding shape: {tuple(out['embedding'].shape)}")
    print(f"logits shape:    {tuple(out['logits'].shape)}")
    print(f"spoof_prob:      {out['spoof_prob'].detach().cpu().tolist()}")


if __name__ == "__main__":
    _smoke_test()


