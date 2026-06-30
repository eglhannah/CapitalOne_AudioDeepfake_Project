#!/usr/bin/env python
"""Generate AASIST degradation-curve PNG for the sponsor sync deck."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Four EER points across difficulty
DATASETS = [
    ("2019 LA dev",   0.90, "In-domain\n(seen attacks)"),
    ("2019 LA eval",  3.33, "Unknown attacks\n(A07-A19)"),
    ("2021 LA eval",  5.67, "+ Telephony codecs\n(7 conditions)"),
    ("2021 DF eval", 22.95, "+ Media compression\n+ 100+ unseen attacks"),
]

# Per-compression for 2021 DF (the robustness story)
DF_COMPRESSION = [
    ("low_ogg",  21.96),
    ("oggm4a",   22.10),
    ("high_ogg", 22.66),
    ("nocodec",  23.19),
    ("mp3m4a",   23.19),
    ("high_mp3", 23.19),
    ("high_m4a", 23.27),
    ("low_m4a",  23.32),
    ("low_mp3",  23.63),
]

PRD_GATE = 25.0  # %

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]})

# === Left panel: degradation curve ===
labels = [d[0] for d in DATASETS]
eers = [d[1] for d in DATASETS]
descriptions = [d[2] for d in DATASETS]

xs = list(range(len(labels)))
ax1.plot(xs, eers, marker="o", linewidth=2.5, markersize=12, color="#1f77b4", zorder=3)
for x, y, label in zip(xs, eers, labels):
    ax1.annotate(f"{y:.2f}%",
                 xy=(x, y),
                 xytext=(0, 14),
                 textcoords="offset points",
                 ha="center", fontsize=12, fontweight="bold", color="#1f77b4")

# PRD gate line
ax1.axhline(y=PRD_GATE, color="red", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
ax1.text(0.02, PRD_GATE + 0.5, f"PRD gate: {PRD_GATE}% EER", color="red", fontsize=10,
         fontweight="bold", transform=ax1.get_yaxis_transform())

# Shade the "passing" zone
ax1.axhspan(0, PRD_GATE, color="green", alpha=0.05, zorder=1)

ax1.set_xticks(xs)
ax1.set_xticklabels([f"{lbl}\n{desc}" for lbl, desc in zip(labels, descriptions)],
                    fontsize=10)
ax1.set_ylabel("Dev/Eval EER (%)", fontsize=12)
ax1.set_title("AASIST degradation curve — train on 2019 LA, test across 4 splits",
              fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, 30)

# === Right panel: per-compression on 2021 DF ===
codecs = [c[0] for c in DF_COMPRESSION]
codec_eers = [c[1] for c in DF_COMPRESSION]

bars = ax2.barh(codecs, codec_eers, color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)
for bar, eer in zip(bars, codec_eers):
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{eer:.2f}%", va="center", fontsize=9)

ax2.axvline(x=PRD_GATE, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.text(PRD_GATE - 0.2, len(codecs) - 0.3, "PRD gate (25%)",
         color="red", fontsize=9, ha="right", fontweight="bold")
ax2.set_xlabel("EER (%)", fontsize=11)
ax2.set_title("2021 DF — per-compression EER\n(remarkably consistent → robustness)",
              fontsize=12, fontweight="bold")
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis="x")
ax2.set_xlim(20, 26)

# === Overall figure caption ===
fig.suptitle(
    "AASIST cross-domain evaluation — all 4 datasets under PRD gate, graceful degradation",
    fontsize=10, y=1.00, color="gray"
)

plt.tight_layout()
out_path = Path(__file__).resolve().parent / "aasist_degradation_curve.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
