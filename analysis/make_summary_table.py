"""Figure 5 — Summary statistics table (CSV + LaTeX) for final tasks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_data, PARADIGM_ORDER

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    final = df[df["is_final"]].copy()
    final["paradigm"] = pd.Categorical(
        final["paradigm"], categories=PARADIGM_ORDER, ordered=True
    )

    rows = []
    for scale in ["10M", "100M"]:
        for paradigm in PARADIGM_ORDER:
            sub = final[(final["token_scale"] == scale) & (final["paradigm"] == paradigm)]
            if sub.empty:
                continue
            rows.append({
                "Token Scale":  scale,
                "Paradigm":     paradigm,
                "Mean PPL":     round(sub["perplexity"].mean(),  3),
                "Std PPL":      round(sub["perplexity"].std(),   3),
                "Mean CEL":     round(sub["CEL"].mean(),         4),
                "Std CEL":      round(sub["CEL"].std(),          4),
                "CN Score":     round(sub["cn_accuracy"].mean(), 4),
                "CN Score Std": round(sub["cn_accuracy"].std(),  4),
            })

    return pd.DataFrame(rows)


def _latex_table(tbl: pd.DataFrame) -> str:
    caption = (
        "Summary statistics (mean $\\pm$ std across 3 runs) for final-stage models. "
        "CN Score = mean rank-difference (mean\\_rank[pF] $-$ mean\\_rank[pM]) averaged over "
        "length-matched sentence pairs; positive values indicate the model assigns lower NLL "
        "to the grammatically correct main-auxiliary preposing."
    )
    latex = tbl.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        column_format="ll" + "r" * (len(tbl.columns) - 2),
    )
    # Wrap in table environment with caption
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{{tab:summary}}",
        latex.strip(),
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    df  = load_data()
    tbl = build_table(df)

    csv_path = OUT / "summary_table.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    tex_path = OUT / "summary_table.tex"
    tex_path.write_text(_latex_table(tbl))
    print(f"Saved {tex_path}")

    print("\n" + tbl.to_string(index=False))


if __name__ == "__main__":
    main()
