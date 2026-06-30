#!/usr/bin/env python
"""Generate AASIST training curve PNG for the sponsor sync deck."""
from pathlib import Path
import matplotlib.pyplot as plt

# Data from /scratch/mhq8ka/aasist/outputs/runs/aasist_fast_13608120/history.json
HISTORY = [
    {"epoch":  1, "tr_loss": 0.6053, "dv_loss": 0.3647, "dv_eer": 0.1232, "dv_auc": 0.9451, "dv_acc": 0.7579},
    {"epoch":  2, "tr_loss": 0.2884, "dv_loss": 0.3406, "dv_eer": 0.0604, "dv_auc": 0.9849, "dv_acc": 0.7618},
    {"epoch":  3, "tr_loss": 0.1769, "dv_loss": 0.2289, "dv_eer": 0.0318, "dv_auc": 0.9931, "dv_acc": 0.8497},
    {"epoch":  4, "tr_loss": 0.1173, "dv_loss": 0.1175, "dv_eer": 0.0298, "dv_auc": 0.9959, "dv_acc": 0.9335},
    {"epoch":  5, "tr_loss": 0.1108, "dv_loss": 0.0921, "dv_eer": 0.0239, "dv_auc": 0.9964, "dv_acc": 0.9517},
    {"epoch":  6, "tr_loss": 0.0967, "dv_loss": 0.1004, "dv_eer": 0.0255, "dv_auc": 0.9966, "dv_acc": 0.9445},
    {"epoch":  7, "tr_loss": 0.0777, "dv_loss": 0.0574, "dv_eer": 0.0201, "dv_auc": 0.9977, "dv_acc": 0.9851},
    {"epoch":  8, "tr_loss": 0.0741, "dv_loss": 0.1143, "dv_eer": 0.0291, "dv_auc": 0.9962, "dv_acc": 0.9363},
    {"epoch":  9, "tr_loss": 0.0620, "dv_loss": 0.0607, "dv_eer": 0.0203, "dv_auc": 0.9978, "dv_acc": 0.9900},
    {"epoch": 10, "tr_loss": 0.0674, "dv_loss": 0.0581, "dv_eer": 0.0188, "dv_auc": 0.9979, "dv_acc": 0.9777},
    {"epoch": 11, "tr_loss": 0.0543, "dv_loss": 0.0528, "dv_eer": 0.0196, "dv_auc": 0.9981, "dv_acc": 0.9838},
    {"epoch": 12, "tr_loss": 0.0504, "dv_loss": 0.0789, "dv_eer": 0.0255, "dv_auc": 0.9968, "dv_acc": 0.9700},
    {"epoch": 13, "tr_loss": 0.0441, "dv_loss": 0.0455, "dv_eer": 0.0130, "dv_auc": 0.9989, "dv_acc": 0.9825},
    {"epoch": 14, "tr_loss": 0.0481, "dv_loss": 0.0471, "dv_eer": 0.0138, "dv_auc": 0.9981, "dv_acc": 0.9900},
    {"epoch": 15, "tr_loss": 0.0443, "dv_loss": 0.3202, "dv_eer": 0.0440, "dv_auc": 0.9903, "dv_acc": 0.9843},
    {"epoch": 16, "tr_loss": 0.0380, "dv_loss": 0.0476, "dv_eer": 0.0149, "dv_auc": 0.9982, "dv_acc": 0.9884},
    {"epoch": 17, "tr_loss": 0.0463, "dv_loss": 0.0310, "dv_eer": 0.0106, "dv_auc": 0.9993, "dv_acc": 0.9912},
    {"epoch": 18, "tr_loss": 0.0418, "dv_loss": 0.1134, "dv_eer": 0.0235, "dv_auc": 0.9970, "dv_acc": 0.9911},
    {"epoch": 19, "tr_loss": 0.0324, "dv_loss": 0.0691, "dv_eer": 0.0165, "dv_auc": 0.9983, "dv_acc": 0.9927},
    {"epoch": 20, "tr_loss": 0.0398, "dv_loss": 0.0874, "dv_eer": 0.0090, "dv_auc": 0.9992, "dv_acc": 0.9502},
    {"epoch": 21, "tr_loss": 0.0299, "dv_loss": 0.0333, "dv_eer": 0.0103, "dv_auc": 0.9993, "dv_acc": 0.9932},
    {"epoch": 22, "tr_loss": 0.0291, "dv_loss": 0.0508, "dv_eer": 0.0134, "dv_auc": 0.9986, "dv_acc": 0.9827},
    {"epoch": 23, "tr_loss": 0.0270, "dv_loss": 0.0788, "dv_eer": 0.0227, "dv_auc": 0.9970, "dv_acc": 0.9813},
    {"epoch": 24, "tr_loss": 0.0296, "dv_loss": 0.1044, "dv_eer": 0.0224, "dv_auc": 0.9971, "dv_acc": 0.9926},
    {"epoch": 25, "tr_loss": 0.0287, "dv_loss": 3.7656, "dv_eer": 0.0752, "dv_auc": 0.9469, "dv_acc": 0.9118},
]

epochs   = [r["epoch"] for r in HISTORY]
tr_loss  = [r["tr_loss"] for r in HISTORY]
dv_loss  = [r["dv_loss"] for r in HISTORY]
dv_eer   = [r["dv_eer"] * 100 for r in HISTORY]  # convert to %
best_idx = min(range(len(dv_eer)), key=lambda i: dv_eer[i])
best_ep  = epochs[best_idx]
best_eer = dv_eer[best_idx]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# --- Top panel: dev EER ---
ax1.plot(epochs, dv_eer, marker="o", color="#1f77b4", linewidth=2, markersize=5, label="Dev EER")
ax1.axhline(y=25, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="PRD gate (25%)")
ax1.scatter([best_ep], [best_eer], color="red", s=120, zorder=5, label=f"Best: ep{best_ep}, {best_eer:.2f}%")
ax1.annotate(f"  {best_eer:.2f}% EER",
             xy=(best_ep, best_eer),
             xytext=(best_ep + 0.5, best_eer + 1.5),
             fontsize=11, color="red", fontweight="bold")
ax1.set_ylabel("Dev EER (%)", fontsize=12)
ax1.set_title("AASIST training on ASVspoof 2019 LA — dev set EER per epoch",
              fontsize=13, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# --- Bottom panel: losses ---
ax2.plot(epochs, tr_loss, marker="o", color="#2ca02c", linewidth=2, markersize=4, label="Train loss")
ax2.plot(epochs, dv_loss, marker="s", color="#d62728", linewidth=2, markersize=4, label="Dev loss")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss (CE, weighted)", fontsize=12)
ax2.set_title("Training and dev loss curves", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale("log")  # log scale handles the ep25 dev_loss spike cleanly

# Layout + save
fig.suptitle("AASIST — 25 epochs, NVIDIA A6000, 3.3 hr — Best dev EER 0.90% (ep20)",
             fontsize=11, y=1.00, color="gray")
plt.tight_layout()
out_path = Path(__file__).resolve().parent / "aasist_training_curve.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
