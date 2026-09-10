"""
training/runner.py — top-level task pipelines (train+eval / train-only /
eval-only), wired up to wandb.

Each function takes a fully-resolved Config (cfg.model / cfg.training /
cfg.task / cfg.wandb) for a single task -- callers (baby_logic_lm.cli.train
for a single Hydra-driven task, baby_logic_lm.cli.pipeline for a --tasks
sequence) are responsible for composing that Config per task.
"""

import gc
import logging

import torch
from datasets import load_from_disk
from transformers import GPTNeoXForCausalLM, Trainer, TrainingArguments, set_seed

from baby_logic_lm.config_schema import TaskConfig
from baby_logic_lm.evaluation.evaluate import Evaluation
from baby_logic_lm.models.build import build_model, build_tokenizer_and_collator
from baby_logic_lm.tracking import wandb_utils
from baby_logic_lm.training.loop import train
from baby_logic_lm.training.results import save_results

logger = logging.getLogger(__name__)


# ── Shared dataset loading ─────────────────────────────────────────────────────

def _load_datasets(task: TaskConfig):
    dataset  = load_from_disk(task.data_path)
    train_n  = min(task.train_truncation, len(dataset["train"])) if task.train_truncation else None
    eval_n   = min(task.test_truncation,  len(dataset["test"]))  if task.test_truncation  else None
    train_ds = dataset["train"].select(range(train_n)) if train_n else dataset["train"]
    eval_ds  = dataset["test"].select(range(eval_n))   if eval_n  else dataset["test"]
    return train_ds, eval_ds


def _train_tokens_actual(task: TaskConfig, train_ds) -> int:
    """Total tokens consumed across all epochs, accounting for last-epoch truncation."""
    from baby_logic_lm.training.loop import _last_epoch_truncation

    last_trunc = _last_epoch_truncation(task, train_ds)
    tokens_per_epoch = sum(len(ids) for ids in train_ds["input_ids"])

    if last_trunc is None:
        return tokens_per_epoch * task.num_train_epochs

    full_epochs = task.num_train_epochs - 1
    last_epoch_ds = train_ds.select(range(last_trunc))
    last_epoch_tokens = sum(len(ids) for ids in last_epoch_ds["input_ids"])
    return full_epochs * tokens_per_epoch + last_epoch_tokens


def _cleanup(model, tokenizer):
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def evaluate(task: TaskConfig, tokenizer, train_eval_results: dict) -> Evaluation:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTNeoXForCausalLM.from_pretrained(task.model_save_path).to(device)
    logger.info("Running evaluation for: %s (device=%s)", task.name, device)

    evaluation = Evaluation(
        model        = model,
        tokenizer    = tokenizer,
        eval_results = train_eval_results,
        truncation   = task.eval_truncation,
    )
    evaluation.eval(CN=task.run_cn, blimp=task.run_blimp)
    return evaluation


# ── Top-level pipelines ───────────────────────────────────────────────────────

def run_task(cfg, run_num: int = 1, tag: str = ""):
    task, train_cfg = cfg.task, cfg.training
    logger.info("=" * 55)
    logger.info("Task: %s  [run %d]", task.name, run_num)
    logger.info("=" * 55)

    set_seed(run_num)
    wandb_mod = wandb_utils.init_wandb(cfg, task.name, run_num, tag, job_type="train")

    tokenizer, collator = build_tokenizer_and_collator(task)
    model = build_model(cfg.model, task, tokenizer)
    train_ds, eval_ds = _load_datasets(task)

    train_tokens = _train_tokens_actual(task, train_ds)
    train_eval_results = train(model, tokenizer, train_ds, eval_ds,
                               collator, task, train_cfg, seed=run_num)

    evaluation = evaluate(task, tokenizer, train_eval_results)
    save_results(evaluation, task, train_cfg, run_num, train_tokens, tag=tag)
    wandb_utils.log_final_metrics(wandb_mod, evaluation)
    wandb_utils.finish(wandb_mod)

    _cleanup(model, tokenizer)


def run_task_train_only(cfg, run_num: int = 1, tag: str = ""):
    task, train_cfg = cfg.task, cfg.training
    logger.info("=" * 55)
    logger.info("Task (train): %s  [run %d]", task.name, run_num)
    logger.info("=" * 55)

    set_seed(run_num)
    wandb_mod = wandb_utils.init_wandb(cfg, task.name, run_num, tag, job_type="train")

    tokenizer, collator = build_tokenizer_and_collator(task)
    model = build_model(cfg.model, task, tokenizer)
    train_ds, eval_ds = _load_datasets(task)

    train(model, tokenizer, train_ds, eval_ds, collator, task, train_cfg, seed=run_num)
    wandb_utils.finish(wandb_mod)

    _cleanup(model, tokenizer)


def run_task_eval_only(cfg, run_num: int = 1, tag: str = ""):
    task, train_cfg = cfg.task, cfg.training
    logger.info("=" * 55)
    logger.info("Task (eval): %s  [run %d]", task.name, run_num)
    logger.info("=" * 55)

    set_seed(run_num)
    wandb_mod = wandb_utils.init_wandb(cfg, task.name, run_num, tag, job_type="eval")

    tokenizer, collator = build_tokenizer_and_collator(task)
    train_ds, eval_ds = _load_datasets(task)

    train_tokens = _train_tokens_actual(task, train_ds)

    # Load saved model and compute eval loss to stand in for train_eval_results
    model = GPTNeoXForCausalLM.from_pretrained(task.model_save_path)
    eval_args = TrainingArguments(
        output_dir                  = task.model_save_path,
        per_device_eval_batch_size  = train_cfg.per_device_eval_batch_size,
        report_to                   = "none",
        seed                        = run_num,
    )
    trainer = Trainer(
        model=model, args=eval_args, tokenizer=tokenizer,
        eval_dataset=eval_ds, data_collator=collator,
    )
    train_eval_results = trainer.evaluate()

    evaluation = evaluate(task, tokenizer, train_eval_results)
    save_results(evaluation, task, train_cfg, run_num, train_tokens, tag=tag)
    wandb_utils.log_final_metrics(wandb_mod, evaluation)
    wandb_utils.finish(wandb_mod)

    _cleanup(model, tokenizer)
