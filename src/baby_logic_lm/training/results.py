import csv
import logging
import os
from datetime import datetime, timezone

from baby_logic_lm.config_schema import BASE_MODEL_ID, RESULTS_CSV, TaskConfig, TrainingConfig
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
