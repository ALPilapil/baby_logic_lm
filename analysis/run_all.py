"""Run all analysis scripts in order, generating all figures and tables."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MODULES = [
    "plot_perplexity",
    "plot_cel",
    "plot_cn_heatmap",
    "plot_cn_accuracy",
    "make_summary_table",
]


def main():
    for name in MODULES:
        print(f"\n{'─' * 50}")
        print(f"Running {name} ...")
        mod = importlib.import_module(name)
        mod.main()
    print(f"\n{'─' * 50}")
    print("All outputs written to results/")


if __name__ == "__main__":
    main()
