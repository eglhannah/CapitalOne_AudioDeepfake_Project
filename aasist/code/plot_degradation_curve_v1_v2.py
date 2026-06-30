#!/usr/bin/env python
"""Generate v1 vs v2 degradation-curve PNG showing RawBoost impact."""
from pathlib import Path
import matplotlib.pyplot as plt

# Four test sets, sorted by difficulty
DATASETS = [
    ("2019 LA dev",   "In-domain\n(seen attacks)",            0.90, 0.78),
    ("2019 LA eval",  "Unknown attacks\n(A07-A19)",           3.33, 1.87),
    ("2021 LA eval",  "+ Telephony codecs\n(7 conditions)",   5.67, 8.01),
    ("2021 DF eval",  "+ Media compression\n+ 100+ unseen",  22.95, 17.20),
]

# 2021 DF per-attack improvements (the headline)
DF_ATTACK_DELTAS = [
    ("A09",  0.79,  1.17),   # Griffin-Lim, trivial
    ("A13",  3.96,  4.24),
    ("A14",  8.37,  4.41),
    ("Task2-team12", 7.78, 2.63),
    ("A11",  8.90, 11.49),
    ("A08", 13.31,  8.26),
    ("A07", 15.87, 15.69),
    ("A16", 16.22,  8.14),
    ("A15", 16.21, 17.12),
    ("A12", 17.52, 18.27),
    ("A10", 17.22, 20.96),
    ("A19", 26.57,  6.29),
    ("Task2-team29", 28.24, 27.39),
    ("A17", 31.82, 10.13),
    ("A18", 40.93, 27.38),
]

PRD_GATE = 25.0
TARGET = 15.0  # Mustafa's "12-15% or below" target

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]})

# ─────────────────────────────────────────
# Left panel: degradation curve, v1 vs v2
# ─────────────────────────────────────────
ax1 = axes[0]
labels = [d[0] for d in DATASETS]
descriptions = [d[1] for d in DATASETS]
v1 = [d[2] for d in DATASETS]
v2 = [d[3] for d in DATASETS]
xs = list(range(len(labels)))

ax1.plot(xs, v1, marker="o", linewidth=2.5, markersize=11, color="#888888",
         label="v1 (no augmentation)", zorder=3)
ax1.plot(xs, v2, marker="s", linewidth=2.5, markersize=11, color="#1f77b4",
         label="v2 (+ RawBoost)", zorder=4)

# Annotate each point
for x, y_v1, y_v2 in zip(xs, v1, v2):
    ax1.annotate(f"{y_v1:.2f}%", xy=(x, y_v1), xytext=(0, 12),
                 textcoords="offset points", ha="center",
                 fontsize=10, color="#666666")
    color = "#1f77b4" if y_v2 < y_v1 else "#d62728"
    ax1.annotate(f"{y_v2:.2f}%", xy=(x, y_v2), xytext=(0, -22),
                 textcoords="offset points", ha="center",
                 fontsize=11, fontweight="bold", color=color)

# PRD gate + sponsor target lines
ax1.axhline(y=PRD_GATE, color="red", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
ax1.text(0.02, PRD_GATE + 0.5, f"PRD gate: {PRD_GATE}% EER", color="red",
         fontsize=9, fontweight="bold", transform=ax1.get_yaxis_transform())
ax1.axhline(y=TARGET, color="#2ca02c", linestyle=":", linewidth=1.5, alpha=0.7, zorder=2)
ax1.text(0.02, TARGET + 0.3, f"Sponsor target: ≤{TARGET}%", color="#2ca02c",
         fontsize=9, fontweight="bold", transform=ax1.get_yaxis_transform())

ax1.axhspan(0, PRD_GATE, color="green", alpha=0.04, zorder=1)

ax1.set_xticks(xs)
ax1.set_xticklabels([f"{lbl}\n{desc}" for lbl, desc in zip(labels, descriptions)],
                    fontsize=10)
ax1.set_ylabel("EER (%)", fontsize=12)
ax1.set_title("AASIST degradation curve — v1 vs v2 (+RawBoost)",
              fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=11, frameon=True)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, 30)

# ─────────────────────────────────────────
# Right panel: 2021 DF per-attack v1 vs v2
# ─────────────────────────────────────────
ax2 = axes[1]

# Sort by v1 EER (hardest to easiest) so improvements are visually clear
sorted_data = sorted(DF_ATTACK_DELTAS, key=lambda x: -x[1])
attacks = [d[0] for d in sorted_data]
v1_attacks = [d[1] for d in sorted_data]
v2_attacks = [d[2] for d in sorted_data]

y_pos = list(range(len(attacks)))
height = 0.4

ax2.barh([y - height/2 for y in y_pos], v1_attacks, height=height,
         color="#888888", label="v1", alpha=0.85, edgecolor="black", linewidth=0.3)
ax2.barh([y + height/2 for y in y_pos], v2_attacks, height=height,
         color="#1f77b4", label="v2 (+RawBoost)", alpha=0.85,
         edgecolor="black", linewidth=0.3)

# Highlight the two big wins
for i, (atk, v1_val, v2_val) in enumerate(sorted_data):
    if atk in ("A17", "A18"):
        improvement = v1_val - v2_val
        ax2.text(v1_val + 0.5, i, f"  −{improvement:.0f}pp ✓",
                 va="center", fontsize=9, fontweight="bold", color="#1f77b4")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(attacks, fontsize=9)
ax2.set_xlabel("EER (%) on 2021 DF eval", fontsize=11)
ax2.set_title("Per-attack EER on 2021 DF (sorted by v1 difficulty)\nA17 & A18 improvements drive the headline",
              fontsize=11, fontweight="bold")
ax2.legend(loc="lower right", fontsize=10)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis="x")
ax2.set_xlim(0, 45)

# ─────────────────────────────────────────
fig.suptitle(
    "AASIST v2 with RawBoost — 2021 DF: 22.95% → 17.20% (-25% relative). "
    "Wins on 3 of 4 datasets; hardest attacks A17/A18 improve dramatically.",
    fontsize=10, y=1.00, color="gray"
)

plt.tight_layout()
out_path = Path(__file__).resolve().parent / "aasist_v1_vs_v2_degradation_curve.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
