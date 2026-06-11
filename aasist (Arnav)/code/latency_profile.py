#!/usr/bin/env python
"""Per-utterance latency profiling — AASIST inference with sliding-window
handling of variable-length audio.

Addresses Mustafa Recommendation #3 (June 9, 2026):
"Determine per-utterance latency profiling to ensure they calculate the metric
across a sliding window of variable file lengths (e.g., 1-second, 5-second,
and 10-second clips)."

AASIST is trained on fixed 64,600-sample clips (~4.04 sec @ 16 kHz). For
clips shorter than 4 sec, we pad. For clips longer than 4 sec, we slide a
4-sec window across the audio with configurable stride, average the per-window
spoof scores, and report the aggregated score plus total wall-clock latency.

Outputs:
  - Per-clip-length statistics (mean, median, p95 latency)
  - Number of model forward passes per clip
  - Throughput in clips/sec
  - JSON results to --out for downstream plotting

Compared against PRD NFR: ≤4 minutes per sample.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

USER = os.environ["USER"]
BASE = Path(f"/scratch/{USER}/aasist")
AASIST_REPO = BASE / "code/aasist"
sys.path.insert(0, str(AASIST_REPO))

from models.AASIST import Model as AASISTModel

AASIST_CFG = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64600   # 4.04 sec — AASIST's native input length


def make_synthetic_audio(duration_sec, sr=SAMPLE_RATE, seed=None):
    """Generate a synthetic noise clip of the given duration. We use random
    noise so the inference cost is realistic (the model still runs all layers)
    without needing actual audio files."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_sec * sr))
    return (rng.standard_normal(n_samples).astype(np.float32) * 0.1)


def fit_to_window(wav, n_target=WINDOW_SAMPLES):
    """Pad short clips with repeat-tile; do nothing for clips >= target."""
    n = len(wav)
    if n >= n_target:
        return wav  # caller handles long clips via sliding window
    repeats = (n_target + n - 1) // n
    return np.tile(wav, repeats)[:n_target]


def sliding_window_indices(n_samples, win=WINDOW_SAMPLES, stride=None):
    """Return start indices for non-overlapping(ish) sliding windows.

    If stride is None, default to win//2 (50% overlap). For exactly fitting
    a 4-sec window into a 5-sec clip, this gives ~2 windows. For 10 sec,
    ~5 windows.
    """
    if stride is None:
        stride = win // 2
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, stride))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)  # ensure last sample covered
    return starts


@torch.no_grad()
def score_with_sliding_window(model, wav_np, device, stride=None):
    """Score arbitrary-length audio by sliding the 4-sec window and averaging.

    Returns: (aggregated_spoof_prob, n_windows, total_inference_time_sec)
    """
    if len(wav_np) < WINDOW_SAMPLES:
        wav_np = fit_to_window(wav_np)
        starts = [0]
    else:
        starts = sliding_window_indices(len(wav_np), WINDOW_SAMPLES, stride)

    scores = []
    t_inference = 0.0
    for start in starts:
        window = wav_np[start:start + WINDOW_SAMPLES]
        x = torch.from_numpy(window).float().unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inference += time.perf_counter() - t0
        prob_spoof = torch.softmax(logits, dim=-1)[0, 1].item()
        scores.append(prob_spoof)
    return float(np.mean(scores)), len(starts), t_inference


def profile_clip_length(model, device, duration_sec, n_trials=50, stride=None):
    """Run n_trials inferences on synthetic clips of the given duration.

    Returns dict with latency stats.
    """
    latencies = []
    inference_only = []
    n_windows_list = []

    # Warm-up (3 trials, discarded)
    for _ in range(3):
        wav = make_synthetic_audio(duration_sec, seed=0)
        _, _, _ = score_with_sliding_window(model, wav, device, stride=stride)

    # Actual measurement
    for trial in range(n_trials):
        wav = make_synthetic_audio(duration_sec, seed=trial)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, n_wins, t_inf = score_with_sliding_window(model, wav, device, stride=stride)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - t0
        latencies.append(total_latency)
        inference_only.append(t_inf)
        n_windows_list.append(n_wins)

    latencies_ms = [x * 1000 for x in latencies]
    inference_ms = [x * 1000 for x in inference_only]

    def p95(xs):
        if not xs:
            return 0.0
        return float(np.percentile(xs, 95))

    return {
        "duration_sec": duration_sec,
        "n_trials": n_trials,
        "n_windows_mean": float(mean(n_windows_list)),
        "total_latency_ms": {
            "mean": float(mean(latencies_ms)),
            "median": float(median(latencies_ms)),
            "p95": p95(latencies_ms),
            "min": float(min(latencies_ms)),
            "max": float(max(latencies_ms)),
        },
        "inference_only_ms": {
            "mean": float(mean(inference_ms)),
            "median": float(median(inference_ms)),
            "p95": p95(inference_ms),
        },
        "throughput_clips_per_sec": 1000.0 / float(mean(latencies_ms)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to AASIST best.pt (e.g. v1 or v2 run dir)")
    p.add_argument("--out", required=True,
                   help="Where to write the JSON results")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="Profile on cuda (default) or cpu")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Per-duration trial count (after warm-up)")
    p.add_argument("--durations", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 5.0, 10.0, 30.0],
                   help="Clip durations (sec) to profile")
    p.add_argument("--stride-sec", type=float, default=2.0,
                   help="Sliding window stride (sec); default 2.0 = 50%% overlap")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AASISTModel(AASIST_CFG).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"AASIST params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")

    stride_samples = int(round(args.stride_sec * SAMPLE_RATE))
    print(f"Sliding window: {WINDOW_SAMPLES} samples ({WINDOW_SAMPLES/SAMPLE_RATE:.3f} sec), "
          f"stride {stride_samples} samples ({args.stride_sec:.1f} sec)")

    results = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch", "?"),
        "model_params": sum(p.numel() for p in model.parameters()),
        "window_samples": WINDOW_SAMPLES,
        "window_sec": WINDOW_SAMPLES / SAMPLE_RATE,
        "stride_sec": args.stride_sec,
        "n_trials_per_duration": args.n_trials,
        "per_duration": [],
    }

    print()
    print(f"{'duration':>10} {'n_windows':>10} {'mean_ms':>10} {'median_ms':>10} "
          f"{'p95_ms':>10} {'clips/sec':>10}")
    print("-" * 65)
    for d in args.durations:
        stats = profile_clip_length(model, device, d, n_trials=args.n_trials,
                                    stride=stride_samples)
        results["per_duration"].append(stats)
        print(f"{d:>9.1f}s {stats['n_windows_mean']:>10.1f} "
              f"{stats['total_latency_ms']['mean']:>10.2f} "
              f"{stats['total_latency_ms']['median']:>10.2f} "
              f"{stats['total_latency_ms']['p95']:>10.2f} "
              f"{stats['throughput_clips_per_sec']:>10.1f}")

    print()
    # PRD compliance check
    PRD_MAX_MS = 4 * 60 * 1000  # 4 minutes in ms
    print(f"PRD target: ≤4 min (240,000 ms) per inference")
    worst_p95 = max(s["total_latency_ms"]["p95"] for s in results["per_duration"])
    print(f"Worst observed p95 across all durations: {worst_p95:.2f} ms")
    print(f"Headroom vs PRD: {PRD_MAX_MS / worst_p95:.0f}× under target" if worst_p95 > 0 else "n/a")

    results["prd_target_ms"] = PRD_MAX_MS
    results["worst_p95_ms"] = worst_p95
    results["headroom_factor"] = PRD_MAX_MS / worst_p95 if worst_p95 > 0 else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
