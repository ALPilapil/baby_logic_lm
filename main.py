import argparse
import copy
import os
import sys

sys.path.insert(0, "scripts")

from datasets import load_from_disk

from config import PRETRAIN_CONFIGS, TASK_CONFIGS, TrainingConfig
from train import run_task, run_task_eval_only, run_task_train_only


def avg_example_length(dataset_path, sample=1000):
    ds = load_from_disk(dataset_path)["train"]
    n = min(sample, len(ds))
    return sum(len(ex["input_ids"]) for ex in ds.select(range(n))) / n


def main():
    parser = argparse.ArgumentParser(
        description="Train baby_logic_lm experiments.",
        epilog=(
            "Example:\n"
            "  python main.py --tasks dyck_pretrain dyck_5m_childes "
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
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="'train': train only (no eval/CSV); 'eval': evaluate a saved model; 'both': train then evaluate (default).",
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
        task_cfg = all_configs[name]
        load_path = task_cfg.model_load_path
        if load_path and load_path not in produced_in_run and not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Task '{name}' requires checkpoint '{load_path}' but it does not exist. "
                "Run the pre-training task that produces it first."
            )
        if args.mode == "eval":
            run_paths = (
                [f"{task_cfg.model_save_path}_run{r}" for r in range(1, args.runs + 1)]
                if args.runs > 1
                else [task_cfg.model_save_path]
            )
            missing = [p for p in run_paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    f"Task '{name}' is missing trained models: {missing}. "
                    "Run training first."
                )
        produced_in_run.add(task_cfg.model_save_path)

    train_cfg = TrainingConfig()

    paths_produced_in_command = {all_configs[n].model_save_path for n in args.tasks}

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

            if args.runs > 1:
                task.model_save_path = f"{task.model_save_path}_run{run_num}"
                if task.model_load_path and task.model_load_path in paths_produced_in_command:
                    task.model_load_path = f"{task.model_load_path}_run{run_num}"

            if args.mode == "train":
                run_task_train_only(task, train_cfg, run_num, args.tag)
            elif args.mode == "eval":
                run_task_eval_only(task, train_cfg, run_num, args.tag)
            else:
                run_task(task, train_cfg, run_num, args.tag)


if __name__ == "__main__":
    main()
