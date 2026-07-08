"""Score-level ensemble of AASIST v3 + w2v (Mohini) on 2021 DF and 2021 LA.

Handles the label/score convention flip:
- Arnav's convention:  label=1 -> spoof, score = P(spoof)
- Mohini's convention: label=1 -> bonafide, score = P(bonafide)

Both are converted to Arnav's convention before joining. Verified labels
agree post-flip as a sanity check.

Outputs:
- v3_ensemble_metrics.json - per-weight ensemble EER on both eval sets
- v3_chart_ensemble_2021_df.png - ensemble EER vs w2v weight, with optimum (DF)
- v3_chart_ensemble_2021_la.png - same for LA
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
AASIST_DIR = REPO_ROOT / "aasist" / "results" / "v3_predictions"
W2V_DIR = REPO_ROOT / "w2v"
OUT_DIR = REPO_ROOT / "aasist" / "results"

DATASETS = [
    {
        "name": "2021 DF eval",
        "slug": "2021_df",
        "aasist_csv": AASIST_DIR / "v3_2021_DF_eval_predictions.csv",
        "w2v_csv": W2V_DIR / "test_predictions_fixed2.csv",
    },
    {
        "name": "2021 LA eval",
        "slug": "2021_la",
        "aasist_csv": AASIST_DIR / "v3_2021_LA_eval_predictions.csv",
        "w2v_csv": W2V_DIR / "test_predictions_LA_2021.csv",
    },
]

V3_COLOR = "#1f6feb"
W2V_COLOR = "#d97757"
ENS_COLOR = "#8a2be2"


def eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[idx]), float((fpr[idx] + fnr[idx]) / 2.0)


def load_aasist(path):
    df = pd.read_csv(path).rename(columns={"utterance_id": "file_id"})
    df = df[["file_id", "label", "score"]].copy()
    df.columns = ["file_id", "aasist_label", "aasist_score_spoof"]
    return df


def load_w2v_flip(path):
    df = pd.read_csv(path)[["file_id", "true_label", "score"]].copy()
    df["w2v_label"] = 1 - df["true_label"]
    df["w2v_score_spoof"] = 1.0 - df["score"]
    return df[["file_id", "w2v_label", "w2v_score_spoof"]]


def run_dataset(ds):
    print(f"\n=== {ds['name']} ===")
    aasist = load_aasist(ds["aasist_csv"])
    w2v = load_w2v_flip(ds["w2v_csv"])
    print(f"  AASIST rows: {len(aasist):,}   w2v rows: {len(w2v):,}")

    merged = aasist.merge(w2v, on="file_id", how="inner")
    print(f"  merged (inner): {len(merged):,}")
    disagree = (merged["aasist_label"] != merged["w2v_label"]).sum()
    if disagree > 0:
        raise RuntimeError(f"Labels disagree on {disagree} rows post-flip")

    y = merged["aasist_label"].to_numpy()
    a = merged["aasist_score_spoof"].to_numpy()
    w = merged["w2v_score_spoof"].to_numpy()

    _, aasist_eer = eer(y, a)
    _, w2v_eer = eer(y, w)
    print(f"  standalone AASIST v3: {aasist_eer*100:.2f}%")
    print(f"  standalone w2v:       {w2v_eer*100:.2f}%")

    fine = np.linspace(0.0, 1.0, 201)
    eers = np.array([eer(y, (1 - wt) * a + wt * w)[1] for wt in fine])
    ens_5050 = float(eers[100])
    best_idx = int(np.argmin(eers))
    best_wt = float(fine[best_idx])
    best_eer = float(eers[best_idx])
    print(f"  50/50 ensemble:       {ens_5050*100:.2f}%")
    print(f"  optimal (w2v={best_wt:.2f}): {best_eer*100:.2f}%")
    print(f"  improvement over best standalone: {(min(aasist_eer, w2v_eer) - best_eer)*100:.2f} pp")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(fine, eers * 100, color=ENS_COLOR, lw=2.2, label="Ensemble EER (linear score fusion)")
    ax.axhline(aasist_eer * 100, color=V3_COLOR, lw=1.5, linestyle="--",
               label=f"AASIST v3 alone: {aasist_eer*100:.2f}%")
    ax.axhline(w2v_eer * 100, color=W2V_COLOR, lw=1.5, linestyle="--",
               label=f"w2v alone: {w2v_eer*100:.2f}%")
    ax.scatter([best_wt], [best_eer * 100], color=ENS_COLOR, s=110, zorder=5,
               edgecolor="white", linewidth=2)
    ax.set_xlabel("w2v weight  ·  (AASIST weight = 1 - w2v weight)", fontsize=11)
    ax.set_ylabel(f"EER (%)  on  {ds['name']}  ·  lower is better", fontsize=11)
    ax.set_title(f"Score-level ensemble AASIST v3 + w2v on {ds['name']}",
                 fontsize=12, pad=14)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = OUT_DIR / f"v3_chart_ensemble_{ds['slug']}.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {chart_path.name}")

    return {
        "eval_set": ds["name"],
        "n_joined": int(len(merged)),
        "n_aasist": int(len(aasist)),
        "n_w2v": int(len(w2v)),
        "n_bonafide": int((y == 0).sum()),
        "n_spoof": int((y == 1).sum()),
        "standalone": {
            "aasist_v3": {"eer": aasist_eer, "auc": float(roc_auc_score(y, a))},
            "w2v":       {"eer": w2v_eer,    "auc": float(roc_auc_score(y, w))},
        },
        "ensemble_50_50": {"weight_w2v": 0.5, "eer": ens_5050},
        "ensemble_optimal": {
            "weight_w2v": best_wt, "weight_aasist": 1 - best_wt, "eer": best_eer,
            "improvement_pp_over_best_standalone": float(min(aasist_eer, w2v_eer) - best_eer) * 100.0,
        },
    }


def main():
    all_results = {}
    for ds in DATASETS:
        all_results[ds["slug"]] = run_dataset(ds)
    out = OUT_DIR / "v3_ensemble_metrics.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
