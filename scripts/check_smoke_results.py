"""
check_smoke_results.py — verifies scripts/smoke_test.sh's pipeline run
actually logged sane results to training_results.csv.

Reads the most recent tag=="smoke_check" row per task (smoke_eval_pretrain,
smoke_eval_finetune, smoke_split_phase) and checks:
  - CEL / perplexity are finite floats, CEL > 0
  - smoke_eval_finetune: BLiMP in [0, 1]; CN parses into a 12-key dict
    (one full minimal set, since eval_truncation: 12)
  - smoke_eval_pretrain / smoke_split_phase: CN / BLiMP are empty (run_cn /
    run_blimp are off for these fixtures)

Exits 0 and prints a PASS summary if everything checks out, else exits 1
with a per-row failure reason.
"""

import ast
import csv
import logging
import math
import sys

from baby_logic_lm.config_schema import RESULTS_CSV
from baby_logic_lm.logging_utils import setup_logging

logger = logging.getLogger(__name__)

EXPECTED_TASKS = ["smoke_eval_pretrain", "smoke_eval_finetune", "smoke_split_phase"]
TAG = "smoke_check"


def _latest_rows_by_task(filename: str, tag: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("tag") != tag:
                continue
            task = row["task_type"]
            if task not in latest or row["timestamp"] > latest[task]["timestamp"]:
                latest[task] = row
    return latest


def _check_finite_float(row: dict, field: str, errors: list[str], *, positive: bool = False):
    try:
        value = float(row[field])
    except (KeyError, ValueError):
        errors.append(f"{field}={row.get(field)!r} is not a float")
        return None
    if math.isnan(value) or math.isinf(value):
        errors.append(f"{field}={value} is not finite")
    if positive and value <= 0:
        errors.append(f"{field}={value} should be > 0")
    return value


def _check_empty(row: dict, field: str, errors: list[str]):
    if row.get(field, "") not in ("", None):
        errors.append(f"{field}={row[field]!r} expected empty (eval disabled for this task)")


def check_row(task: str, row: dict) -> list[str]:
    errors: list[str] = []
    _check_finite_float(row, "CEL", errors, positive=True)
    _check_finite_float(row, "perplexity", errors, positive=True)

    if task == "smoke_eval_finetune":
        blimp = _check_finite_float(row, "BLiMP", errors)
        if blimp is not None and not (0.0 <= blimp <= 1.0):
            errors.append(f"BLiMP={blimp} outside [0, 1]")
        try:
            cn = ast.literal_eval(row["CN"])
            if not isinstance(cn, dict) or len(cn) != 12:
                errors.append(f"CN parsed but has {len(cn) if isinstance(cn, dict) else 'n/a'} keys, expected 12")
        except (KeyError, ValueError, SyntaxError):
            errors.append(f"CN={row.get('CN')!r} did not parse as a dict")
    else:
        _check_empty(row, "CN", errors)
        _check_empty(row, "BLiMP", errors)

    return errors


def main() -> int:
    try:
        latest = _latest_rows_by_task(RESULTS_CSV, TAG)
    except FileNotFoundError:
        logger.error("%s not found -- did the pipeline run complete?", RESULTS_CSV)
        return 1

    all_ok = True
    for task in EXPECTED_TASKS:
        row = latest.get(task)
        if row is None:
            logger.error("FAIL %-22s no tag=%s row found in %s", task, TAG, RESULTS_CSV)
            all_ok = False
            continue

        errors = check_row(task, row)
        if errors:
            all_ok = False
            logger.error("FAIL %-22s %s", task, "; ".join(errors))
        else:
            logger.info("PASS %-22s CEL=%s perplexity=%s", task, row["CEL"], row["perplexity"])

    if all_ok:
        logger.info("All smoke checks passed.")
    else:
        logger.error("Smoke checks FAILED -- see above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
