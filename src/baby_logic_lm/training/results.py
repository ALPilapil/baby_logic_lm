import csv
import json
import logging
import os
from datetime import datetime, timezone

from baby_logic_lm.config_schema import (
    BASE_MODEL_ID,
    RESULTS_CSV,
    RESULTS_JSONL,
    TaskConfig,
    TrainingConfig,
)
from baby_logic_lm.evaluation.evaluate import Evaluation

logger = logging.getLogger(__name__)


def save_results(
    evaluation: Evaluation,
    task: TaskConfig,
    train_cfg: TrainingConfig,
    run_num: int,
    train_tokens: int,
    tag: str = "",
    filename: str = RESULTS_CSV,
    jsonl_filename: str = RESULTS_JSONL,
):
    row = {
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tag":          tag,
        "run":          run_num,
        "task_type":    task.name,
        "base_model":   BASE_MODEL_ID,
        "warmup_from":  task.model_load_path or "random_init",
        "epochs":       task.num_train_epochs,
        "train_tokens": train_tokens,
        "total_tokens": train_tokens,
        "learning_rate": train_cfg.learning_rate,
        "batch_size":   train_cfg.per_device_train_batch_size,
        "CEL":          evaluation.CEL,
        "perplexity":   evaluation.perplexity,
        "CN":           evaluation.CN,
        "BLiMP":        evaluation.blimp,
    }
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Results saved to %s", filename)

    # Sidecar: same row, but CN stays a real nested dict (json.dumps handles
    # the int keys by stringifying them, same as any JSON object) instead of
    # a stringified Python repr -- avoids the ast.literal_eval round trip
    # when reslicing/replotting this data later.
    with open(jsonl_filename, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("Results saved to %s", jsonl_filename)
