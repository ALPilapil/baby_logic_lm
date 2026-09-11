"""save_results() writes both training_results.csv and its training_results.jsonl
sidecar, with CN preserved as a real nested dict in the JSONL (not a
stringified repr like the CSV cell)."""

import ast
import csv
import json
from types import SimpleNamespace

from baby_logic_lm.config_schema import TaskConfig, TrainingConfig
from baby_logic_lm.training.results import save_results


def _fake_evaluation(CN):
    return SimpleNamespace(CEL=1.23, perplexity=4.56, CN=CN, blimp=0.5)


def test_writes_matching_csv_and_jsonl_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    jsonl_path = tmp_path / "results.jsonl"
    task = TaskConfig(name="unit_test", data_path="x", model_save_path="y")
    train_cfg = TrainingConfig()
    cn = {0: {1: 3, 2: 1}, 1: {2: 4}}

    save_results(
        _fake_evaluation(cn), task, train_cfg, run_num=1, train_tokens=1000,
        tag="unit", filename=str(csv_path), jsonl_filename=str(jsonl_path),
    )

    with open(csv_path, newline="") as f:
        csv_row = next(csv.DictReader(f))
    with open(jsonl_path) as f:
        jsonl_row = json.loads(f.readline())

    # Same logical row in both files.
    assert csv_row["task_type"] == jsonl_row["task_type"] == "unit_test"
    assert float(csv_row["CEL"]) == jsonl_row["CEL"] == 1.23

    # CSV stores CN as a Python-repr string that needs ast.literal_eval...
    assert ast.literal_eval(csv_row["CN"]) == cn
    # ...while the JSONL sidecar keeps it as a real (string-keyed) JSON object.
    assert isinstance(jsonl_row["CN"], dict)
    assert jsonl_row["CN"] == {"0": {"1": 3, "2": 1}, "1": {"2": 4}}


def test_appends_multiple_runs(tmp_path):
    csv_path = tmp_path / "results.csv"
    jsonl_path = tmp_path / "results.jsonl"
    task = TaskConfig(name="unit_test", data_path="x", model_save_path="y")
    train_cfg = TrainingConfig()

    for run_num in (1, 2):
        save_results(
            _fake_evaluation(None), task, train_cfg, run_num=run_num, train_tokens=1000,
            filename=str(csv_path), jsonl_filename=str(jsonl_path),
        )

    with open(csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    with open(jsonl_path) as f:
        jsonl_rows = [json.loads(line) for line in f]

    assert len(csv_rows) == len(jsonl_rows) == 2
    assert [r["run"] for r in jsonl_rows] == [1, 2]
