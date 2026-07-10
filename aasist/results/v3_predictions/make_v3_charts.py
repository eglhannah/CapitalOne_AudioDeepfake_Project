"""Generate the v3 presentation charts. Reads only local files."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent

EER = {
    "2019 LA eval": {"v1": 3.33, "v2": 1.87, "v3": 1.67},
    "2021 LA eval": {"v1": 5.67, "v2": 8.01, "v3": 4.67},
    "2021 DF eval": {"v1": 22.95, "v2": 17.20, "v3": 17.00},
}

TEAM_EER = {
    "2019 LA eval": {"aasist_v3": 1.67, "w2v": 1.25, "ensemble": 1.02},
    "2021 LA eval": {"aasist_v3": 4.67, "w2v": 4.49, "ensemble": 3.18},
    "2021 DF eval": {"aasist_v3": 17.01, "w2v": 16.38, "ensemble": 14.87},
}

V1_COLOR = "#9aa3ab"
V2_COLOR = "#88a5c8"
V3_COLOR = "#1f6feb"
W2V_COLOR = "#d97757"
ENS_COLOR = "#8a2be2"

def chart_main():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    benchmarks = list(EER.keys())
    x = np.arange(len(benchmarks))
    w = 0.26
    v1 = [EER[b]["v1"] for b in benchmarks]
    v2 = [EER[b]["v2"] for b in benchmarks]
    v3 = [EER[b]["v3"] for b in benchmarks]
    b1 = ax.bar(x - w, v1, w, label="v1: no augmentation", color=V1_COLOR, edgecolor="white")
    b2 = ax.bar(x,     v2, w, label="v2: RawBoost (incl. synthetic codec)", color=V2_COLOR, edgecolor="white")
    b3 = ax.bar(x + w, v3, w, label="v3: CodecAugment + RawBoost noise", color=V3_COLOR, edgecolor="white")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}%",
                        xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel("EER (%)  ·  lower is better", fontsize=11)
    ax.set_title("AASIST: v3 is best on every benchmark", fontsize=13, pad=14)
    ax.set_ylim(0, max(v1 + v2 + v3) * 1.18)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = OUT / "v3_chart_main_comparison.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"wrote {p}")

def chart_la2021():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    versions = ["v1\nno aug", "v2\nRawBoost\n(synthetic codec)", "v3\nCodecAugment\n(real codecs)"]
    vals = [EER["2021 LA eval"]["v1"], EER["2021 LA eval"]["v2"], EER["2021 LA eval"]["v3"]]
    colors = [V1_COLOR, "#d97757", V3_COLOR]
    bars = ax.bar(versions, vals, color=colors, edgecolor="white", width=0.55)
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}%",
                    xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.annotate("",
                xy=(2, vals[2] + 0.7), xytext=(1, vals[1] + 0.7),
                arrowprops=dict(arrowstyle="->", color="#1f6feb", lw=1.8))
    ax.text(1.5, vals[1] + 1.4, "42% relative\nEER reduction",
            ha="center", va="bottom", fontsize=10, color="#1f6feb", fontweight="bold")
    ax.set_ylabel("EER (%)  on  2021 LA eval", fontsize=11)
    ax.set_title("Real codec augmentation reverses RawBoost's 2021 LA regression",
                 fontsize=12, pad=14)
    ax.set_ylim(0, max(vals) * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = OUT / "v3_chart_la2021_thesis.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"wrote {p}")

def chart_la2021_per_codec():
    summary_path = HERE / "v3_2021_LA_eval_summary.json"
    data = json.loads(summary_path.read_text())
    per_codec = data["per_codec"]
    codecs = ["alaw", "ulaw", "gsm", "pstn", "g722", "opus", "none"]
    eers = [per_codec[c]["eer"] * 100 for c in codecs]
    trained = {"alaw", "ulaw", "g722", "opus"}
    colors = [V3_COLOR if c in trained else V1_COLOR for c in codecs]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(codecs, eers, color=colors, edgecolor="white", width=0.6)
    for r, v in zip(bars, eers):
        ax.annotate(f"{v:.2f}%",
                    xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("EER (%)  on  2021 LA eval", fontsize=11)
    ax.set_title("v3 per-codec breakdown  ·  blue = trained on, gray = unseen at train time",
                 fontsize=11, pad=12)
    ax.set_ylim(0, max(eers) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = OUT / "v3_chart_la2021_per_codec.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"wrote {p}")

def chart_team_comparison():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    benchmarks = list(TEAM_EER.keys())
    x = np.arange(len(benchmarks))
    w = 0.26
    aasist_vals = [TEAM_EER[b]["aasist_v3"] for b in benchmarks]
    w2v_vals = [TEAM_EER[b]["w2v"] for b in benchmarks]
    ens_vals = [TEAM_EER[b]["ensemble"] for b in benchmarks]

    def draw(vals, offset, color, label):
        heights = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + offset, heights, w, label=label, color=color, edgecolor="white")
        for r, v in zip(bars, vals):
            if v is None:
                ax.text(r.get_x() + r.get_width() / 2, 0.3, "pending",
                        ha="center", va="bottom", fontsize=8.5, color="#666", style="italic")
            else:
                ax.annotate(f"{v:.2f}%",
                            xy=(r.get_x() + r.get_width() / 2, v),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
        return bars

    draw(aasist_vals, -w, V3_COLOR, "AASIST v3 (Arnav)")
    draw(w2v_vals,     0, W2V_COLOR, "w2v (Mohini)")
    draw(ens_vals,     w, ENS_COLOR, "Ensemble (50/50 score fusion)")

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel("EER (%)  ·  lower is better", fontsize=11)
    ax.set_title("Team model comparison  ·  ensemble beats best standalone on every benchmark",
                 fontsize=12, pad=14)
    max_val = max(v for v in aasist_vals + w2v_vals + ens_vals if v is not None)
    ax.set_ylim(0, max_val * 1.18)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.text(0.5, -0.02,
             "w2v standalone and ensemble numbers on the intersection of file_ids scored by both models. "
             "Ensemble is 50/50 linear score fusion.",
             ha="center", fontsize=8, color="#666", style="italic")
    fig.tight_layout()
    p = OUT / "v3_chart_team_comparison.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    chart_main()
    chart_la2021()
    chart_la2021_per_codec()
    chart_team_comparison()
