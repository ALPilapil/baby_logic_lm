"""
Collect babylm-eval results for one model run and append a row to the summary CSV.

Zero-shot results read from:
  results/{model}/main/zero_shot/causal/{task}/{subtask}/best_temperature_report.txt

Reading results read from:
  results/{model}/main/zero_shot/causal/reading/report.txt
  Format: "EYE TRACKING SCORE: X.XX\nSELF-PACED READING SCORE: X.XX"

Finetune results read from:
  results/{model}/main/finetune/{task}/results.txt
"""

import argparse
import csv
from pathlib import Path

FINETUNE_METRIC = {
    "boolq":   "accuracy",
    "mnli":    "accuracy",
    "mrpc":    "f1",
    "multirc": "accuracy",
    "qqp":     "f1",
    "rte":     "accuracy",
    "wsc":     "accuracy",
}

# Maps (task_dir, subtask_dir) → CSV column
ZEROSHOT_COL = {
    ("blimp",          "blimp_filtered"):      "blimp_avg",
    ("blimp",          "supplement_filtered"):  "blimp_supplement_avg",

    ("comps",          "comps"):                "comps_avg",
    ("entity_tracking","entity_tracking"):      "entity_tracking_avg",
}

ZERO_SHOT_COLS = [
    "blimp_avg",
    "blimp_supplement_avg",
    "comps_avg",
    "entity_tracking_avg",
    "eye_tracking",
    "spr",
]
FINETUNE_COLS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
HEADER = ["model", "run"] + ZERO_SHOT_COLS + FINETUNE_COLS


def parse_average_score(path: Path) -> float | None:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("### AVERAGE"):
            if i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError:
                    return None
    return None


def parse_reading_report(path: Path) -> tuple[float | None, float | None]:
    eye, spr = None, None
    for line in path.read_text().splitlines():
        if line.startswith("EYE TRACKING SCORE:"):
            try:
                eye = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("SELF-PACED READING SCORE:"):
            try:
                spr = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return eye, spr


def parse_finetune_score(path: Path, metric: str) -> float | None:
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == metric:
            try:
                return float(value.strip()) * 100
            except ValueError:
                return None
    return None


def collect(model: str, stem: str, run: int, results_dir: Path) -> dict:
    row: dict = {"model": stem, "run": run}
    bucket: dict[str, list[float]] = {}

    # Zero-shot: results/{model}/main/zero_shot/causal/{task}/{subtask}/best_temperature_report.txt
    for report in results_dir.glob(f"{model}/main/zero_shot/causal/*/*/best_temperature_report.txt"):
        task    = report.parts[-3]
        subtask = report.parts[-2]
        col = ZEROSHOT_COL.get((task, subtask))
        if col is None:
            continue
        score = parse_average_score(report)
        if score is not None:
            bucket.setdefault(col, []).append(score)

    for col in ["blimp_avg", "blimp_supplement_avg", "comps_avg", "entity_tracking_avg"]:
        scores = bucket.get(col, [])
        row[col] = f"{sum(scores) / len(scores):.4f}" if scores else ""

    # Reading: results/{model}/main/zero_shot/causal/reading/report.txt
    reading_report = results_dir / model / "main" / "zero_shot" / "causal" / "reading" / "report.txt"
    if reading_report.exists():
        eye, spr = parse_reading_report(reading_report)
        row["eye_tracking"] = f"{eye:.4f}" if eye is not None else ""
        row["spr"]          = f"{spr:.4f}" if spr is not None else ""
    else:
        row["eye_tracking"] = ""
        row["spr"]          = ""

    # Finetune: results/{model}/main/finetune/{task}/results.txt
    for task in FINETUNE_COLS:
        results_file = results_dir / model / "main" / "finetune" / task / "results.txt"
        if results_file.exists():
            metric = FINETUNE_METRIC[task]
            score = parse_finetune_score(results_file, metric)
            row[task] = f"{score:.2f}" if score is not None else ""
        else:
            row[task] = ""

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,
                        help="Full model dir name, e.g. ntp_10m_model_run1")
    parser.add_argument("--stem",        required=True,
                        help="Condition name without _runN, e.g. ntp_10m_model")
    parser.add_argument("--run",         type=int, required=True)
    parser.add_argument("--results_dir", required=True,
                        help="Path to the babylm-eval results/ directory")
    parser.add_argument("--output_csv",  required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_csv  = Path(args.output_csv)

    row = collect(args.model, args.stem, args.run, results_dir)

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Appended {args.model} run {args.run} → {output_csv}")


if __name__ == "__main__":
    main()
