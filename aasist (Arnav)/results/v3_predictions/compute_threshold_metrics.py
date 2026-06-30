"""Compute EER-threshold metrics + ROC curves for v3 predictions.

Addresses Mustafa's review comment: 'For the final production, we have mention
the threshold as 0.5 which does not sound right considering the EER and other
thresholds.'

For each eval split:
- Finds the EER operating point (threshold where FPR == FNR)
- Recomputes confusion matrix, accuracy, FPR, FNR at that threshold
- Compares to the default 0.5 threshold
- Plots ROC with both threshold points marked

Outputs:
- v3_threshold_metrics.json: per-split metrics at EER threshold and 0.5
- v3_threshold_metrics.md: human-readable summary (paste into report)
- v3_chart_roc_eval_sets.png: ROC curves for the three external eval sets
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

HERE = Path(__file__).parent
OUT = HERE.parent

SPLITS = [
    ("2019 LA dev",  "v3_2019_LA_dev_predictions.csv"),
    ("2019 LA eval", "v3_2019_LA_eval_predictions.csv"),
    ("2021 LA eval", "v3_2021_LA_eval_predictions.csv"),
    ("2021 DF eval", "v3_2021_DF_eval_predictions.csv"),
]

def eer_threshold(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[idx]), float(fpr[idx]), float(fnr[idx]), (fpr, tpr, thr)

def metrics_at(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((tp + tn) / n),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }

def run():
    results = {}
    roc_curves = {}
    for split_name, csv_name in SPLITS:
        df = pd.read_csv(HERE / csv_name)
        y = df["label"].to_numpy()
        s = df["score"].to_numpy()
        eer_t, eer_fpr, eer_fnr, (fpr, tpr, thr) = eer_threshold(y, s)
        eer = (eer_fpr + eer_fnr) / 2.0
        results[split_name] = {
            "n": int(len(df)),
            "n_bonafide": int((y == 0).sum()),
            "n_spoof": int((y == 1).sum()),
            "auc": float(roc_auc_score(y, s)),
            "eer": eer,
            "at_eer_threshold": metrics_at(y, s, eer_t),
            "at_default_0.5":   metrics_at(y, s, 0.5),
        }
        roc_curves[split_name] = (fpr, tpr, eer_fpr, eer_fnr, eer_t)
        print(f"{split_name}: EER={eer*100:.2f}%  thr_eer={eer_t:.4f}  thr_0.5_acc={results[split_name]['at_default_0.5']['accuracy']*100:.2f}%  thr_eer_acc={results[split_name]['at_eer_threshold']['accuracy']*100:.2f}%")

    (OUT / "v3_threshold_metrics.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# AASIST v3 metrics at the EER operating point",
        "",
        "**Context.** The 6/17 progress report reported all confusion-matrix-based metrics at threshold = 0.5, "
        "noting that 'our team has not determined a suitable ruling method.' This file replaces that placeholder "
        "with metrics computed at the EER operating point, the threshold where FPR and FNR are equal. "
        "EER is the standard operating point for anti-spoofing systems where bonafide and spoof errors are equally costly.",
        "",
        "## Per-split summary",
        "",
        "| Eval split | n | AUC | EER | EER threshold | Acc @ EER thr | Acc @ 0.5 (prior) |",
        "|---|---|---|---|---|---|---|",
    ]
    for split_name, csv_name in SPLITS:
        r = results[split_name]
        lines.append(
            f"| {split_name} | {r['n']:,} | {r['auc']:.4f} | {r['eer']*100:.2f}% | "
            f"{r['at_eer_threshold']['threshold']:.4f} | "
            f"{r['at_eer_threshold']['accuracy']*100:.2f}% | "
            f"{r['at_default_0.5']['accuracy']*100:.2f}% |"
        )
    lines += [
        "",
        "## Why this matters",
        "",
        "At threshold = 0.5 the system applies a uniform decision rule that does not account for the score "
        "distribution learned by AASIST v3. On 2021 DF the default threshold misclassifies a larger share of "
        "bonafide audio than necessary (high FPR), and on 2019 LA dev the default underuses the model's "
        "headroom (FPR and FNR are far from balanced). The EER threshold gives a single defensible operating "
        "point that is reproducible per split and comparable across the team's models.",
        "",
        "## For production deployment",
        "",
        "EER is the *symmetric* operating point. For a production fraud system, the operating point should be "
        "tuned to the relative cost of FPR (rejecting a real customer) versus FNR (admitting a spoofed call). "
        "If a fraud-loss-versus-customer-friction tradeoff is provided by the sponsor, the same ROC curves "
        "below allow selection of any operating point along the curve. Until that tradeoff is specified, EER "
        "is the appropriate reporting threshold.",
        "",
        "## Generated artifacts",
        "",
        "- `v3_threshold_metrics.json` - full confusion matrices at both thresholds",
        "- `v3_chart_roc_eval_sets.png` - ROC curves for the three external eval sets, with the EER point and the 0.5 point marked",
        "",
    ]
    (OUT / "v3_threshold_metrics.md").write_text("\n".join(lines))

    eval_splits = ["2019 LA eval", "2021 LA eval", "2021 DF eval"]
    colors = ["#1f6feb", "#d97757", "#8a2be2"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for split_name, color in zip(eval_splits, colors):
        fpr, tpr, eer_fpr, eer_fnr, eer_t = roc_curves[split_name]
        r = results[split_name]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{split_name}  ·  AUC={r['auc']:.3f}  ·  EER={r['eer']*100:.2f}%")
        ax.scatter([eer_fpr], [1 - eer_fnr], color=color, s=70, zorder=5,
                   edgecolor="white", linewidth=1.5)
        d05 = r["at_default_0.5"]
        ax.scatter([d05["fpr"]], [1 - d05["fnr"]], color=color, s=70, marker="x", zorder=5, linewidth=2.2)
    ax.plot([0, 1], [0, 1], color="#9aa3ab", linestyle=":", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("AASIST v3 ROC curves  ·  ● = EER operating point  ·  ✕ = threshold 0.5",
                 fontsize=12, pad=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = OUT / "v3_chart_roc_eval_sets.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"\nwrote {OUT / 'v3_threshold_metrics.json'}")
    print(f"wrote {OUT / 'v3_threshold_metrics.md'}")
    print(f"wrote {p}")

if __name__ == "__main__":
    run()
