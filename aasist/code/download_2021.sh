#!/bin/bash
# Download ASVspoof 2021 LA + DF eval data + keys/protocols
# Run from any directory; data lands in /scratch/$USER/aasist/data/2021/
set -e

DATA_DIR="/scratch/$USER/aasist/data/2021"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
echo "Writing to: $DATA_DIR"

# --- 2021 LA eval (7.8 GB) ---
echo "[1/6] Downloading LA eval audio..."
wget -c -q --show-progress "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1" \
  -O ASVspoof2021_LA_eval.tar.gz

# --- 2021 LA keys/protocols ---
echo "[2/6] Downloading LA keys..."
wget -c -q --show-progress "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz" \
  -O LA-keys-full.tar.gz

# --- 2021 DF eval (4 parts, 34.5 GB total) ---
for i in 00 01 02 03; do
  echo "[$(( 3 + 10#$i ))/6] Downloading DF eval part $i..."
  wget -c -q --show-progress "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part${i}.tar.gz?download=1" \
    -O "ASVspoof2021_DF_eval_part${i}.tar.gz"
done

# --- 2021 DF keys/protocols ---
echo "[6/6] Downloading DF keys..."
wget -c -q --show-progress "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz" \
  -O DF-keys-full.tar.gz

echo ""
echo "=== All downloads complete ==="
ls -lh "$DATA_DIR"
echo ""
echo "Total disk used:"
du -sh "$DATA_DIR"
