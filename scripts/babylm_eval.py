"""
babylm_eval.py — wrapper for the babylm-eval strict-track evaluation suite.

Calls babylm-eval shell scripts via subprocess (keeps repos decoupled and avoids
dependency conflicts). Results are parsed from the files babylm-eval writes and
appended to babylm_results.csv, which shares task_name/run keys with training_results.csv.

Usage (standalone — run from baby_logic_lm root):
    python scripts/babylm_eval.py \\
        --model_path models/pythia/ntp_10m_model \\
        --task_name ntp_10m \\
        [--run 1] \\
        [--eval_data_dir /path/to/evaluation_data/full_eval] \\
        [--results_dir /path/to/babylm-eval/strict/results] \\
        [--skip_finetune] \\
        [--output_json babylm_results/ntp_10m.json]

One-time setup (run from babylm-eval/strict before using this script):
    python -m scripts.download_evals
    python -m evaluation_pipeline.ewok.dl_and_filter   # agree to EWoK terms on HF first
    cd evaluation_data/fast_eval && unzip ewok_fast.zip  # password: BabyLM2025
    touch /path/to/babylm-eval/.env                      # required even if empty
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BABYLM_STRICT_DIR = Path("/home/alpilapi/projects/babylm-eval/strict")
BACKEND = "causal"
BABYLM_RESULTS_CSV = "./babylm_results.csv"


# ── Preflight check ────────────────────────────────────────────────────────────

def _check_eval_data(eval_data_dir: Path) -> bool:
    if not eval_data_dir.exists():
        print(f"""
[babylm_eval] ERROR: evaluation_data not found at {eval_data_dir}

Run the following from {BABYLM_STRICT_DIR}:
  1. huggingface-cli login
  2. python -m scripts.download_evals
  3. python -m evaluation_pipeline.ewok.dl_and_filter   # agree to EWoK terms on HF first
  4. cd evaluation_data/fast_eval && unzip ewok_fast.zip  (password: BabyLM2025)
  5. touch {BABYLM_STRICT_DIR.parent}/.env               # needed by eval_finetuning.sh
""")
        return False
    return True


# ── Subprocess runner ──────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path) -> int:
    print(f"\n[babylm_eval] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        print(f"[babylm_eval] WARNING: command returned {result.returncode}")
    return result.returncode


# ── Result parsers ─────────────────────────────────────────────────────────────

def _model_stem(model_path: str) -> str:
    return Path(model_path).stem


def _parse_avg_accuracy(report_path: Path) -> Optional[float]:
    """
    Parse the average accuracy from a best_temperature_report.txt written by
    evaluation_pipeline.sentence_zero_shot.run. Format:
        ### AVERAGE ACCURACY
        75.23
    """
    if not report_path.exists():
        return None
    lines = report_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if "AVERAGE ACCURACY" in line.upper():
            for nxt in lines[i + 1:]:
                nxt = nxt.strip()
                if nxt:
                    try:
                        return float(nxt)
                    except ValueError:
                        pass
    return None


def _parse_finetune_metric(results_path: Path, metric: str = "accuracy") -> Optional[float]:
    """
    Parse a metric from results.txt written by evaluation_pipeline.finetune.run.
    Format: "accuracy: 0.75"
    """
    if not results_path.exists():
        return None
    for line in results_path.read_text().splitlines():
        if line.startswith(metric + ":"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return None


# ── Zero-shot evaluation ───────────────────────────────────────────────────────

def run_zero_shot(model_path: str, eval_data_dir: Path) -> dict:
    abs_model_path = str(Path(model_path).resolve())
    _run(
        ["bash", "scripts/eval_zero_shot.sh", abs_model_path, BACKEND, str(eval_data_dir)],
        cwd=BABYLM_STRICT_DIR,
    )
    return _parse_zero_shot_scores(model_path)


def _parse_zero_shot_scores(model_path: str) -> dict:
    stem = _model_stem(model_path)
    base = BABYLM_STRICT_DIR / "results" / stem / "main" / "zero_shot" / BACKEND
    scores = {}

    task_dataset_pairs = [
        ("babylm_blimp_filtered",    "blimp",           "blimp_filtered"),
        ("babylm_blimp_supplement",  "blimp",           "supplement_filtered"),
        ("babylm_ewok",              "ewok",            "ewok_filtered"),
        ("babylm_entity_tracking",   "entity_tracking", "entity_tracking"),
        ("babylm_comps",             "comps",           "comps"),
    ]
    for key, task, dataset in task_dataset_pairs:
        report = base / task / dataset / "best_temperature_report.txt"
        scores[key] = _parse_avg_accuracy(report)

    # Reading produces no accuracy metric — record the output directory instead
    reading_dir = base / "reading"
    scores["babylm_reading_output"] = str(reading_dir) if reading_dir.exists() else None

    return scores


# ── Fine-tuning evaluation ─────────────────────────────────────────────────────

def run_finetune(model_path: str, seed: int = 42) -> dict:
    abs_model_path = str(Path(model_path).resolve())
    _run(
        ["bash", "scripts/eval_finetuning.sh", "--model_path", abs_model_path, "--seed", str(seed)],
        cwd=BABYLM_STRICT_DIR,
    )
    return _parse_finetune_scores(model_path)


def _parse_finetune_scores(model_path: str) -> dict:
    stem = _model_stem(model_path)
    base = BABYLM_STRICT_DIR / "results" / stem / "main" / "finetune"

    # mrpc and qqp use F1 as the primary metric; the rest use accuracy
    tasks_metric = {
        "boolq":   "accuracy",
        "mnli":    "accuracy",
        "mrpc":    "f1",
        "multirc": "accuracy",
        "qqp":     "f1",
        "rte":     "accuracy",
        "wsc":     "accuracy",
    }
    scores = {}
    for task, metric in tasks_metric.items():
        results_path = base / task / "results.txt"
        scores[f"babylm_glue_{task}"] = _parse_finetune_metric(results_path, metric)
    return scores


# ── CSV / JSON output ──────────────────────────────────────────────────────────

def _append_to_csv(row: dict, filename: str = BABYLM_RESULTS_CSV) -> None:
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[babylm_eval] Results appended to {filename}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained model directory (HF Trainer output)")
    parser.add_argument("--task_name", required=True,
                        help="Task label (matches task_name in training_results.csv)")
    parser.add_argument("--run", type=int, default=1,
                        help="Run number (for joining with training_results.csv)")
    parser.add_argument("--eval_data_dir",
                        default=str(BABYLM_STRICT_DIR / "evaluation_data" / "full_eval"),
                        help="Path to babylm-eval/strict/evaluation_data/full_eval")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Skip the slow GLUE fine-tuning evaluations")
    parser.add_argument("--finetune_seed", type=int, default=42)
    parser.add_argument("--output_json", default=None,
                        help="If set, also write all scores to this JSON file")
    args = parser.parse_args()

    eval_data_dir = Path(args.eval_data_dir)
    if not _check_eval_data(eval_data_dir):
        sys.exit(1)

    all_scores: dict = {
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_name":  args.task_name,
        "run":        args.run,
        "model_path": args.model_path,
    }

    print(f"\n[babylm_eval] === Zero-shot evaluations: {args.task_name} ===")
    all_scores.update(run_zero_shot(args.model_path, eval_data_dir))

    if not args.skip_finetune:
        print(f"\n[babylm_eval] === GLUE fine-tuning evaluations: {args.task_name} ===")
        all_scores.update(run_finetune(args.model_path, seed=args.finetune_seed))
    else:
        for task in ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]:
            all_scores[f"babylm_glue_{task}"] = None

    print("\n[babylm_eval] === Scores ===")
    for k, v in all_scores.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(BABYLM_RESULTS_CSV) or ".", exist_ok=True)
    _append_to_csv(all_scores)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        Path(args.output_json).write_text(json.dumps(all_scores, indent=2))
        print(f"[babylm_eval] JSON snapshot saved to {args.output_json}")

    return all_scores


if __name__ == "__main__":
    main()
