"""Figure 2 — Cross-entropy loss by paradigm, with warmup-stage bars."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

BAR_W = 0.38
GAP   = 0.04


def _plot_panel(ax, df_scale, scale_label):
    # Final stage bars
    final = df_scale[df_scale["is_final"]].copy()
    final["paradigm"] = pd.Categorical(
        final["paradigm"], categories=PARADIGM_ORDER, ordered=True
    )
    f_grouped = final.groupby("paradigm", observed=True)["CEL"]
    f_means   = f_grouped.mean().reindex(PARADIGM_ORDER)
    f_stds    = f_grouped.std().reindex(PARADIGM_ORDER)

    # Warmup stage bars (NTP before NSP/NUP fine-tuning)
    warmup = df_scale[df_scale["stage"] == "warmup"].copy()
    warmup["paradigm"] = pd.Categorical(
        warmup["paradigm"], categories=PARADIGM_ORDER, ordered=True
    )
    w_grouped = warmup.groupby("paradigm", observed=True)["CEL"]
    w_means   = w_grouped.mean().reindex(PARADIGM_ORDER)
    w_stds    = w_grouped.std().reindex(PARADIGM_ORDER)

    x      = np.arange(len(PARADIGM_ORDER))
    colors = [PALETTE.get(p, "#999999") for p in PARADIGM_ORDER]

    # Final bars (centered left)
    ax.bar(x - (BAR_W + GAP) / 2, f_means, yerr=f_stds, width=BAR_W,
           color=colors, capsize=3, error_kw=dict(linewidth=1.0),
           label="Final stage", zorder=3)

    # Warmup bars (centered right, lighter)
    warmup_colors = [c + "88" for c in colors]  # hex alpha
    has_warmup = w_means.notna()
    ax.bar(x[has_warmup] + (BAR_W + GAP) / 2,
           w_means[has_warmup], yerr=w_stds[has_warmup],
           width=BAR_W, color=[warmup_colors[i] for i in range(len(PARADIGM_ORDER)) if has_warmup.iloc[i]],
           capsize=3, error_kw=dict(linewidth=1.0),
           label="NTP warmup stage", zorder=3)

    rng = np.random.default_rng(42)
    for i, paradigm in enumerate(PARADIGM_ORDER):
        # Final dots
        vals = final[final["paradigm"] == paradigm]["CEL"].values
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i - (BAR_W + GAP) / 2 + jitter, vals,
                   color="white", edgecolors=colors[i],
                   linewidths=1.0, s=20, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(PARADIGM_ORDER, rotation=22, ha="right")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(scale_label, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def main():
    df = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=False)

    for ax, scale in zip(axes, ["10M", "100M"]):
        _plot_panel(ax, df[df["token_scale"] == scale], f"{scale} tokens")

    # Shared legend
    solid_patch  = mpatches.Patch(facecolor="#555555",    label="Final stage")
    light_patch  = mpatches.Patch(facecolor="#55555588",  label="NTP warmup stage")
    axes[1].legend(handles=[solid_patch, light_patch], fontsize=8,
                   loc="upper right", frameon=False)

    fig.suptitle("Cross-Entropy Loss by Training Paradigm",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUT / "fig_cel.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
