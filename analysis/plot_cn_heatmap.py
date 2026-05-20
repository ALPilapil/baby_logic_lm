"""Figure 3 — CN rank-1 rate heatmap (final tasks, averaged across runs)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_data, PARADIGM_ORDER, CN_SHORT_LABELS, CHANCE, GRAMMATICAL_POSITIONS

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


def _build_matrix(df_scale):
    """Return (5 × 12) mean rank-1-rate matrix, rows = paradigms, cols = positions."""
    final = df_scale[df_scale["is_final"] & df_scale["rank1_rates"].notna()]
    mat = np.full((len(PARADIGM_ORDER), 12), np.nan)
    for i, paradigm in enumerate(PARADIGM_ORDER):
        rows = final[final["paradigm"] == paradigm]["rank1_rates"]
        if rows.empty:
            continue
        stacked = np.vstack(rows.values)
        mat[i] = np.nanmean(stacked, axis=0)
    return mat


def _plot_panel(ax, mat, scale_label):
    vmax = max(abs(mat[~np.isnan(mat)] - CHANCE).max(), 0.02)
    vmin_c, vmax_c = CHANCE - vmax, CHANCE + vmax

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=vmin_c, vmax=vmax_c, interpolation="nearest")

    # Annotate cells
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isnan(v):
                continue
            txt_color = "white" if abs(v - CHANCE) > vmax * 0.55 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    # Highlight grammatical columns
    for g in GRAMMATICAL_POSITIONS:
        for r in range(mat.shape[0]):
            ax.add_patch(mpatches.FancyBboxPatch(
                (g - 0.48, r - 0.48), 0.96, 0.96,
                boxstyle="square,pad=0",
                linewidth=1.5, edgecolor="gold", facecolor="none", zorder=5,
            ))

    # Set A / Set B column divider
    ax.axvline(5.5, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.text(2.5, -0.8, "Set A", ha="center", va="center", fontsize=8,
            fontweight="bold", transform=ax.transData)
    ax.text(8.5, -0.8, "Set B", ha="center", va="center", fontsize=8,
            fontweight="bold", transform=ax.transData)

    ax.set_xticks(range(12))
    ax.set_xticklabels(CN_SHORT_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(PARADIGM_ORDER)))
    ax.set_yticklabels(PARADIGM_ORDER)
    ax.set_title(scale_label, fontweight="bold")
    ax.tick_params(length=0)

    return im


def main():
    df = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.88, wspace=0.35, top=0.82, bottom=0.22)
    images = []

    for ax, scale in zip(axes, ["10M", "100M"]):
        mat = _build_matrix(df[df["token_scale"] == scale])
        im  = _plot_panel(ax, mat, f"{scale} tokens")
        images.append(im)

    # Single colorbar in reserved right margin
    cbar_ax = fig.add_axes([0.90, 0.22, 0.015, 0.60])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label("Rank-1 rate", fontsize=9)
    cbar.ax.axhline(CHANCE, color="black", linewidth=1.0, linestyle="--")

    fig.suptitle(
        "CN Rank-1 Rate by Paradigm and Sentence Condition\n"
        "(gold border = grammatically correct form; F = prepose 1st-aux, M = prepose main-aux; "
        "d0/dF/dM = delete none/first/main)",
        fontsize=9.5, y=0.97,
    )

    out = OUT / "fig_cn_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
