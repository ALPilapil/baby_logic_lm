"""BabyLM evaluation results — summary table and bar plots.

Reads babylm_eval_results.csv (produced by run_babylm_eval.sh / collect_babylm_results.py),
maps each model to its paradigm/scale via TASK_META, then outputs:
  results/babylm_summary.csv   — mean ± std per condition across 3 runs
  results/babylm_summary.tex   — same as LaTeX table
  results/babylm_zeroshot.pdf  — bar plots for zero-shot metrics
  results/babylm_finetune.pdf  — bar plots for fine-tune (GLUE) metrics
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import PARADIGM_ORDER, PALETTE, TASK_META

# Strip "_model" suffix to recover task_type
MODEL_TO_TASK = {
    stem + "_model": task
    for task, (paradigm, stage, scale) in TASK_META.items()
    if stage == "final"
    for stem in [task]  # key == task, dir == task + "_model"
}

ZEROSHOT_METRICS = [
    ("blimp_avg",            "BLiMP"),
    ("blimp_supplement_avg", "BLiMP Suppl."),
    ("comps_avg",            "COMPS"),
    ("entity_tracking_avg",  "Entity Track."),
    ("eye_tracking",         "Eye Tracking"),
    ("spr",                  "SPR"),
]

FINETUNE_METRICS = [
    ("boolq",   "BoolQ"),
    ("mnli",    "MNLI"),
    ("mrpc",    "MRPC"),
    ("multirc", "MultiRC"),
    ("qqp",     "QQP"),
    ("rte",     "RTE"),
    ("wsc",     "WSC"),
]

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)


def load_babylm(csv_path: str | Path = "babylm_eval_results.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Map model → task_type → paradigm / scale
    df["task_type"] = df["model"].map(MODEL_TO_TASK)
    missing = df["task_type"].isna()
    if missing.any():
        print(f"  Warning: unmapped models: {df.loc[missing, 'model'].unique().tolist()}")
        df = df[~missing].copy()

    meta = df["task_type"].map(TASK_META)
    df["paradigm"]    = meta.map(lambda x: x[0])
    df["token_scale"] = meta.map(lambda x: x[2])

    # Coerce metric columns to float
    metric_cols = [c for c, _ in ZEROSHOT_METRICS + FINETUNE_METRICS]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c, _ in ZEROSHOT_METRICS + FINETUNE_METRICS]
    rows = []
    for scale in ["10M", "100M"]:
        for paradigm in PARADIGM_ORDER:
            sub = df[(df["token_scale"] == scale) & (df["paradigm"] == paradigm)]
            if sub.empty:
                continue
            row = {"Token Scale": scale, "Paradigm": paradigm}
            for col in metric_cols:
                if col not in sub.columns:
                    row[col + "_mean"] = np.nan
                    row[col + "_std"]  = np.nan
                    continue
                vals = sub[col].dropna()
                row[col + "_mean"] = round(vals.mean(), 3) if len(vals) else np.nan
                row[col + "_std"]  = round(vals.std(),  3) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _latex_summary(tbl: pd.DataFrame, metric_pairs: list[tuple]) -> str:
    display_cols = []
    for col, label in metric_pairs:
        mean_col = col + "_mean"
        std_col  = col + "_std"
        if mean_col in tbl.columns:
            display_cols.append((mean_col, std_col, label))

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{BabyLM evaluation results (mean $\\pm$ std across 3 runs).}",
        "\\label{tab:babylm}",
        "\\begin{tabular}{ll" + "r" * len(display_cols) + "}",
        "\\toprule",
        "Scale & Paradigm & " + " & ".join(label for _, _, label in display_cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in tbl.iterrows():
        cells = [str(row["Token Scale"]), str(row["Paradigm"])]
        for mean_col, std_col, _ in display_cols:
            m, s = row.get(mean_col, np.nan), row.get(std_col, np.nan)
            if pd.isna(m):
                cells.append("—")
            else:
                cells.append(f"{m:.2f}$\\pm${s:.2f}" if not pd.isna(s) else f"{m:.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _bar_panel(df: pd.DataFrame, metrics: list[tuple], title: str, out_path: Path):
    n_metrics = len(metrics)
    scales    = ["10M", "100M"]
    n_scales  = len(scales)

    fig, axes = plt.subplots(
        n_scales, n_metrics,
        figsize=(3.2 * n_metrics, 4.5 * n_scales),
        sharey="row",
    )
    if n_scales == 1:
        axes = axes[np.newaxis, :]
    if n_metrics == 1:
        axes = axes[:, np.newaxis]

    x     = np.arange(len(PARADIGM_ORDER))
    width = 0.6

    for row_i, scale in enumerate(scales):
        sub = df[df["token_scale"] == scale]
        for col_i, (col, label) in enumerate(metrics):
            ax = axes[row_i, col_i]
            means, stds, colors = [], [], []
            for paradigm in PARADIGM_ORDER:
                pdata = sub[sub["paradigm"] == paradigm][col].dropna()
                means.append(pdata.mean() if len(pdata) else np.nan)
                stds.append(pdata.std()   if len(pdata) else np.nan)
                colors.append(PALETTE.get(paradigm, "#999999"))

            bars = ax.bar(x, means, width, yerr=stds, capsize=3, color=colors, error_kw={"lw": 1})
            ax.set_xticks(x)
            ax.set_xticklabels(
                [p.replace(" ", "\n") for p in PARADIGM_ORDER],
                fontsize=7,
            )
            ax.set_title(f"{label}\n({scale})", fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            if col_i == 0:
                ax.set_ylabel("Score", fontsize=8)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    df = load_babylm()

    # Summary table
    tbl = build_summary(df)
    csv_path = OUT / "babylm_summary.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Zero-shot LaTeX table
    tex_zs = OUT / "babylm_zeroshot_summary.tex"
    tex_zs.write_text(_latex_summary(tbl, ZEROSHOT_METRICS))
    print(f"  Saved {tex_zs}")

    # Finetune LaTeX table
    tex_ft = OUT / "babylm_finetune_summary.tex"
    tex_ft.write_text(_latex_summary(tbl, FINETUNE_METRICS))
    print(f"  Saved {tex_ft}")

    # Bar plots
    _bar_panel(df, ZEROSHOT_METRICS, "BabyLM Zero-shot Evaluation",
               OUT / "babylm_zeroshot.pdf")
    _bar_panel(df, FINETUNE_METRICS, "BabyLM Fine-tune Evaluation (GLUE)",
               OUT / "babylm_finetune.pdf")

    print("\nSummary (means across runs):\n")
    display_cols = ["Token Scale", "Paradigm"] + [c + "_mean" for c, _ in ZEROSHOT_METRICS + FINETUNE_METRICS]
    display_cols = [c for c in display_cols if c in tbl.columns]
    print(tbl[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
