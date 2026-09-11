"""
export_wandb.py — pulls every run from the wandb project into a local
snapshot (results/wandb_runs.csv + results/wandb_runs.jsonl) you can load
with pandas and reslice into whatever plots you want, without needing to be
online/authenticated at plot time.

Each row has: run name, group (task name), tags, job_type, state, the full
resolved config that was logged at wandb.init() time (flattened, prefixed
"config."), and every key in the run's summary (final/CEL, final/perplexity,
final/BLiMP, final/CN if present, plus whatever the HF Trainer logged).

Usage
-----
    python analysis/export_wandb.py                       # uses your default entity
    python analysis/export_wandb.py --entity my-team
    python analysis/export_wandb.py --project other-project
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb

PROJECT_DEFAULT = "baby-logic-lm"
OUT = Path(__file__).parent.parent / "results"


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def fetch_runs(entity: str | None, project: str) -> list[dict]:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"Fetching runs from {path} ...")

    rows = []
    for run in api.runs(path):
        row = {
            "run_id": run.id,
            "name": run.name,
            "group": run.group,
            "tags": list(run.tags),
            "job_type": run.job_type,
            "state": run.state,
            "url": run.url,
        }
        row.update(_flatten(dict(run.config), prefix="config."))
        row.update(dict(run.summary))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=None, help="wandb entity (default: your default entity)")
    parser.add_argument("--project", default=PROJECT_DEFAULT)
    args = parser.parse_args()

    rows = fetch_runs(args.entity, args.project)
    if not rows:
        print("No runs found.")
        return

    OUT.mkdir(exist_ok=True)

    jsonl_path = OUT / "wandb_runs.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    csv_path = OUT / "wandb_runs.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Wrote {len(rows)} runs to {csv_path} and {jsonl_path}")


if __name__ == "__main__":
    main()
