import argparse
import copy
import os
import sys

sys.path.insert(0, "scripts")

from datasets import load_from_disk

from config import PRETRAIN_CONFIGS, TASK_CONFIGS, TrainingConfig
from train import run_task


def avg_example_length(dataset_path, sample=1000):
    ds = load_from_disk(dataset_path)["train"]
    n = min(sample, len(ds))
    return sum(len(ex["input_ids"]) for ex in ds.select(range(n))) / n


def main():
    parser = argparse.ArgumentParser(
        description="Train baby_logic_lm experiments.",
        epilog=(
            "Example:\n"
            "  python main.py --tasks paren_pretrain paren_then_next_word "
            "--epochs 1 --pretrain-tokens 5000000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        metavar="TASK",
        help=(
            "Task names to run in order. "
            f"TASK_CONFIGS: {sorted(TASK_CONFIGS)}. "
            f"PRETRAIN_CONFIGS: {sorted(PRETRAIN_CONFIGS)}."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="num_train_epochs applied to every task (default: 1).",
    )
    parser.add_argument(
        "--pretrain-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Token budget for pre-training tasks. Sets train_truncation automatically: "
            "N // 512 for paren (fixed 512-token blocks); "
            "N // avg_example_length for pos (variable length)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to repeat the full task sequence (default: 1). Each run uses its index as the random seed.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        metavar="LABEL",
        help="Optional label written to every row of training_results.csv for grouping runs.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    all_configs = {**PRETRAIN_CONFIGS, **TASK_CONFIGS}

    for name in args.tasks:
        if name not in all_configs:
            raise ValueError(
                f"Unknown task '{name}'. Valid tasks: {sorted(all_configs)}"
            )

    produced_in_run = set()
    for name in args.tasks:
        load_path = all_configs[name].model_load_path
        if load_path and load_path not in produced_in_run and not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Task '{name}' requires checkpoint '{load_path}' but it does not exist. "
                "Run the pre-training task that produces it first."
            )
        produced_in_run.add(all_configs[name].model_save_path)

    train_cfg = TrainingConfig()

    for run_num in range(1, args.runs + 1):
        for name in args.tasks:
            task = copy.copy(all_configs[name])
            if not task.lock_epochs:
                task.num_train_epochs = args.epochs

            if name in PRETRAIN_CONFIGS and args.pretrain_tokens:
                if "paren" in name:
                    task.train_truncation = args.pretrain_tokens // 512
                else:
                    avg_len = avg_example_length(task.data_path)
                    task.train_truncation = int(args.pretrain_tokens // avg_len)

            run_task(task, train_cfg, run_num, args.tag)


if __name__ == "__main__":
    main()
