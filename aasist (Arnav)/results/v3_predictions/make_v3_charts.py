"""Generate the two v3 presentation charts. Reads only local files."""
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

V1_COLOR = "#9aa3ab"
V2_COLOR = "#88a5c8"
V3_COLOR = "#1f6feb"

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

if __name__ == "__main__":
    chart_main()
    chart_la2021()
    chart_la2021_per_codec()
