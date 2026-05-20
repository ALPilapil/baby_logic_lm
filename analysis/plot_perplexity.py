"""Figure 1 — Perplexity by paradigm (final CHILDES tasks only)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_data, PARADIGM_ORDER, PALETTE

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _plot_panel(ax, df_scale, scale_label):
    final = df_scale[df_scale["is_final"]].copy()
    final["paradigm"] = pd.Categorical(
        final["paradigm"], categories=PARADIGM_ORDER, ordered=True
    )
    grouped = final.groupby("paradigm", observed=True)["perplexity"]
    means   = grouped.mean().reindex(PARADIGM_ORDER)
    stds    = grouped.std().reindex(PARADIGM_ORDER)

    x      = np.arange(len(PARADIGM_ORDER))
    colors = [PALETTE.get(p, "#999999") for p in PARADIGM_ORDER]

    ax.bar(x, means, yerr=stds, width=0.55, color=colors,
           capsize=4, error_kw=dict(linewidth=1.2), zorder=3)

    rng = np.random.default_rng(42)
    for i, paradigm in enumerate(PARADIGM_ORDER):
        vals = final[final["paradigm"] == paradigm]["perplexity"].values
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(i + jitter, vals, color="white", edgecolors=colors[i],
                   linewidths=1.1, s=24, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(PARADIGM_ORDER, rotation=22, ha="right")
    ax.set_ylabel("Perplexity")
    ax.set_title(scale_label, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def main():
    df = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

    for ax, scale in zip(axes, ["10M", "100M"]):
        _plot_panel(ax, df[df["token_scale"] == scale], f"{scale} tokens")

    fig.suptitle("Perplexity by Training Paradigm", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUT / "fig_perplexity.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
