#!/usr/bin/env python
"""Pre-compute codec round-tripped copies of the 2019 LA train set.

For each clean train .flac, encode through a real telephony/streaming codec
and decode back to 16 kHz PCM, saved as .wav under
    /scratch/$USER/aasist/data/LA_codec_train/<codec>/<filename>.wav

Codecs covered (4 of the originally planned 5):
    alaw  — PCM A-law       (8 kHz, 64 kbps)
    ulaw  — PCM mu-law      (8 kHz, 64 kbps)
    g722  — G.722 ADPCM     (16 kHz, 64 kbps)  *** wideband telephony ***
    opus  — libopus VoIP    (16 kHz, 16 kbps)  *** Asterisk default ***

    (gsm was planned but Rivanna's ffmpeg 8.1.1 is built without libgsm.
     Omitted. The four above still cover the 2021 LA pain points.)

Why this exists:
    v2's RawBoost approximates codec effects with a closed-form mu-law/A-law
    companding + 8-bit quantization. The 2021 LA eval set goes through real
    Asterisk PBX codec round trips that produce different artifacts. v2
    regressed on 2021 LA (5.67% to 8.01%). v3 replaces the closed-form
    simulation with real codec round trips through ffmpeg.

Approach:
    Two-step subprocess per codec using a tempfile in /tmp. The pipe-based
    one-shot approach was attempted first but several codecs (gsm, opus,
    g722, ulaw) don't pipe cleanly because non-seekable stdout produces
    invalid container headers.

Run once. Idempotent: skips files that already exist.

Usage:
    python precompute_codec_train.py
    python precompute_codec_train.py --codecs alaw opus
    python precompute_codec_train.py --workers 16
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

USER = os.environ["USER"]
BASE = Path(f"/scratch/{USER}/aasist")
SRC_DIR = BASE / "data/LA/ASVspoof2019_LA_train/flac"
DST_BASE = BASE / "data/LA_codec_train"

ALL_CODECS = ("alaw", "ulaw", "g722", "opus")

# Per-codec settings. tmp_ext is the container/extension used for the
# intermediate encoded file; ffmpeg picks the container from the extension
# when -f is not explicitly given.
CODEC_SETTINGS = {
    "alaw":  {"encode_args": ["-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", "-f", "wav"],
              "tmp_ext": ".wav"},
    "ulaw":  {"encode_args": ["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", "-f", "wav"],
              "tmp_ext": ".wav"},
    "g722":  {"encode_args": ["-ac", "1", "-c:a", "g722", "-f", "g722"],
              "tmp_ext": ".g722"},
    "opus":  {"encode_args": ["-ar", "16000", "-ac", "1", "-c:a", "libopus",
                              "-b:a", "16k", "-application", "voip"],
              "tmp_ext": ".ogg"},
}


def encode_decode(src_file: Path, codec: str) -> tuple[str, bool, str]:
    """Encode src_file through codec, decode back to 16 kHz mono PCM.

    Returns (codec, success, message).
    """
    dst_file = DST_BASE / codec / (src_file.stem + ".wav")
    if dst_file.exists() and dst_file.stat().st_size > 0:
        return (codec, True, "skipped (exists)")

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    if codec not in CODEC_SETTINGS:
        return (codec, False, f"unknown codec: {codec}")

    settings = CODEC_SETTINGS[codec]
    tmp_path = None
    try:
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=settings["tmp_ext"], prefix="codec_")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        encode_cmd = (
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_file)]
            + settings["encode_args"]
            + [str(tmp_path)]
        )
        r1 = subprocess.run(encode_cmd, capture_output=True, timeout=60)
        if r1.returncode != 0:
            return (codec, False, f"encode failed: {r1.stderr.decode().strip()[:200]}")

        decode_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tmp_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(dst_file),
        ]
        r2 = subprocess.run(decode_cmd, capture_output=True, timeout=60)
        if r2.returncode != 0:
            return (codec, False, f"decode failed: {r2.stderr.decode().strip()[:200]}")

        if not dst_file.exists() or dst_file.stat().st_size == 0:
            return (codec, False, "output missing or empty")
        return (codec, True, "ok")
    except subprocess.TimeoutExpired:
        return (codec, False, "subprocess timeout")
    except Exception as e:
        return (codec, False, f"exception: {e!r}")
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def task(args: tuple[Path, str]) -> tuple[str, bool, str]:
    src_file, codec = args
    return encode_decode(src_file, codec)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--codecs", nargs="+", default=list(ALL_CODECS),
                   help=f"Codecs to generate (default: {' '.join(ALL_CODECS)})")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes (default: 8)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N source files (for smoke testing)")
    args = p.parse_args(argv)

    bad = [c for c in args.codecs if c not in CODEC_SETTINGS]
    if bad:
        print(f"ERROR: unknown codec(s): {bad}", file=sys.stderr)
        return 2

    if not SRC_DIR.exists():
        print(f"ERROR: source dir missing: {SRC_DIR}", file=sys.stderr)
        return 2

    src_files = sorted(SRC_DIR.glob("*.flac"))
    if args.limit is not None:
        src_files = src_files[:args.limit]
    print(f"Source files: {len(src_files)}")
    print(f"Codecs:       {args.codecs}")
    print(f"Output base:  {DST_BASE}")
    print(f"Workers:      {args.workers}")

    tasks = [(f, c) for f in src_files for c in args.codecs]
    total = len(tasks)
    print(f"Total tasks:  {total}")

    counts = {c: {"ok": 0, "skipped": 0, "failed": 0} for c in args.codecs}
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            codec, success, msg = fut.result()
            if success and "skipped" in msg:
                counts[codec]["skipped"] += 1
            elif success:
                counts[codec]["ok"] += 1
            else:
                counts[codec]["failed"] += 1
                if len(failures) < 20:
                    failures.append(f"{codec}: {msg}")
            if i % 500 == 0 or i == total:
                done_summary = " ".join(
                    f"{c}:{counts[c]['ok'] + counts[c]['skipped']}/{counts[c]['failed']}f"
                    for c in args.codecs
                )
                print(f"[{i}/{total}] {done_summary}")

    print("\n=== Summary ===")
    for c in args.codecs:
        x = counts[c]
        print(f"{c:8s}  ok={x['ok']:>6d}  skipped={x['skipped']:>6d}  failed={x['failed']:>6d}")
    if failures:
        print(f"\nFirst {len(failures)} failures:")
        for f in failures:
            print(f"  {f}")
    any_failed = any(counts[c]['failed'] for c in args.codecs)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
