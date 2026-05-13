"""
train.py — training pipeline.

Hyperparameters and task definitions live in config.py. The entry point is main.py.
"""

import csv
import gc
import glob
import os
from datetime import datetime, timezone

import torch
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoXForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from collator import CustomDataCollator
from config import (
    BASE_MODEL_ID,
    BLIMP_DIR,
    CN_DATA_PATH,
    PRETRAIN_CONFIGS,
    RESULTS_CSV,
    TASK_CONFIGS,
    TaskConfig,
    TrainingConfig,
)
from eval import Evaluation


# ── Model / tokenizer construction ───────────────────────────────────────────

def build_tokenizer_and_collator(task: TaskConfig):
    tokenizer = AutoTokenizer.from_pretrained(task.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    if task.use_custom_collator:
        collator = CustomDataCollator(tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenizer, collator


def build_model(task: TaskConfig, tokenizer) -> GPTNeoXForCausalLM:
    if task.model_load_path is None:
        config = AutoConfig.from_pretrained(BASE_MODEL_ID)
        model = GPTNeoXForCausalLM(config)
        model.apply(model._init_weights)
        print(f"  Initialized fresh model from {BASE_MODEL_ID} config")
    else:
        model = GPTNeoXForCausalLM.from_pretrained(task.model_load_path)
        print(f"  Loaded model weights from {task.model_load_path}")

    model.resize_token_embeddings(len(tokenizer))
    return model


# ── Token-limit helpers ───────────────────────────────────────────────────────

def _last_epoch_truncation(task: TaskConfig, train_ds) -> int | None:
    """Return the number of examples to use for the last epoch to stay under
    task.token_limit. Returns None when no limit is set."""
    if task.token_limit is None:
        return None

    tokens_per_epoch = sum(len(ids) for ids in train_ds["input_ids"])
    full_epochs = task.num_train_epochs - 1
    last_epoch_budget = task.token_limit - full_epochs * tokens_per_epoch
    avg_tokens = tokens_per_epoch / len(train_ds)
    truncation = int(last_epoch_budget // avg_tokens)
    return max(0, min(truncation, len(train_ds)))


def _latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the most recent checkpoint saved under output_dir."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


# ── Training ──────────────────────────────────────────────────────────────────

def _make_training_args(task: TaskConfig, cfg: TrainingConfig, seed: int,
                        num_epochs: int, report_to: str | None = None) -> TrainingArguments:
    return TrainingArguments(
        output_dir                   = task.model_save_path,
        num_train_epochs             = num_epochs,
        per_device_train_batch_size  = cfg.per_device_train_batch_size,
        per_device_eval_batch_size   = cfg.per_device_eval_batch_size,
        learning_rate                = cfg.learning_rate,
        lr_scheduler_type            = cfg.lr_scheduler_type,
        warmup_ratio                 = cfg.warmup_ratio,
        adam_beta1                   = cfg.adam_beta1,
        adam_beta2                   = cfg.adam_beta2,
        weight_decay                 = cfg.weight_decay,
        eval_strategy                = cfg.eval_strategy,
        eval_steps                   = cfg.eval_steps,
        logging_strategy             = "steps",
        logging_steps                = cfg.logging_steps,
        save_steps                   = cfg.save_steps,
        save_total_limit             = cfg.save_total_limit,
        report_to                    = report_to if report_to is not None else cfg.report_to,
        seed                         = seed,
    )


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    collator,
    task: TaskConfig,
    cfg: TrainingConfig,
    seed: int = 1,
) -> dict:
    """Train the model, optionally splitting into two phases to enforce task.token_limit."""
    last_trunc = _last_epoch_truncation(task, train_dataset)
    full_epochs = task.num_train_epochs - 1

    if last_trunc is not None and full_epochs > 0:
        # Phase 1: N-1 full epochs
        phase1_args = _make_training_args(task, cfg, seed, num_epochs=full_epochs)
        Trainer(
            model=model, args=phase1_args, tokenizer=tokenizer,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=collator,
        ).train()

        # Phase 2: 1 truncated last epoch, resuming from the latest checkpoint
        checkpoint = _latest_checkpoint(task.model_save_path)
        last_epoch_ds = train_dataset.select(range(last_trunc))
        phase2_args = _make_training_args(task, cfg, seed, num_epochs=1)
        trainer = Trainer(
            model=model, args=phase2_args, tokenizer=tokenizer,
            train_dataset=last_epoch_ds, eval_dataset=eval_dataset,
            data_collator=collator,
        )
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(task.model_save_path)
        return trainer.evaluate()

    elif last_trunc is not None:
        # num_train_epochs == 1: single truncated epoch, no resume needed
        last_epoch_ds = train_dataset.select(range(last_trunc))
        trainer = Trainer(
            model=model, args=_make_training_args(task, cfg, seed, num_epochs=1),
            tokenizer=tokenizer, train_dataset=last_epoch_ds,
            eval_dataset=eval_dataset, data_collator=collator,
        )
        trainer.train()
        trainer.save_model(task.model_save_path)
        return trainer.evaluate()

    else:
        # No token limit: original single-call path
        trainer = Trainer(
            model=model,
            args=_make_training_args(task, cfg, seed, num_epochs=task.num_train_epochs),
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(task.model_save_path)
        return trainer.evaluate()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(task: TaskConfig, tokenizer, train_eval_results: dict) -> Evaluation:
    model = GPTNeoXForCausalLM.from_pretrained(task.model_save_path)
    print(f"  Running evaluation for: {task.name}")

    evaluation = Evaluation(
        model        = model,
        tokenizer    = tokenizer,
        eval_results = train_eval_results,
        truncation   = task.eval_truncation,
    )
    evaluation.eval(CN=task.run_cn, blimp=task.run_blimp)
    return evaluation


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
        "total_tokens": train_tokens * task.num_train_epochs,
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
    print(f"  Results saved to {filename}")


# ── Shared dataset loading ─────────────────────────────────────────────────────

def _load_datasets(task: TaskConfig):
    dataset  = load_from_disk(task.data_path)
    train_ds = (dataset["train"].select(range(task.train_truncation))
                if task.train_truncation else dataset["train"])
    eval_ds  = (dataset["test"].select(range(task.test_truncation))
                if task.test_truncation else dataset["test"])
    return train_ds, eval_ds


def _train_tokens_actual(task: TaskConfig, train_ds) -> int:
    """Total tokens consumed across all epochs, accounting for last-epoch truncation."""
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


# ── Top-level pipelines ───────────────────────────────────────────────────────

def run_task(task: TaskConfig, train_cfg: TrainingConfig, run_num: int = 1, tag: str = ""):
    print(f"\n{'='*55}")
    print(f"  Task: {task.name}  [run {run_num}]")
    print(f"{'='*55}")

    set_seed(run_num)

    tokenizer, collator = build_tokenizer_and_collator(task)
    model = build_model(task, tokenizer)
    train_ds, eval_ds = _load_datasets(task)

    train_tokens = _train_tokens_actual(task, train_ds)
    train_eval_results = train(model, tokenizer, train_ds, eval_ds,
                               collator, task, train_cfg, seed=run_num)

    evaluation = evaluate(task, tokenizer, train_eval_results)
    save_results(evaluation, task, train_cfg, run_num, train_tokens, tag=tag)

    _cleanup(model, tokenizer)


def run_task_train_only(task: TaskConfig, train_cfg: TrainingConfig, run_num: int = 1, tag: str = ""):
    print(f"\n{'='*55}")
    print(f"  Task (train): {task.name}  [run {run_num}]")
    print(f"{'='*55}")

    set_seed(run_num)

    tokenizer, collator = build_tokenizer_and_collator(task)
    model = build_model(task, tokenizer)
    train_ds, eval_ds = _load_datasets(task)

    train(model, tokenizer, train_ds, eval_ds, collator, task, train_cfg, seed=run_num)

    _cleanup(model, tokenizer)


def run_task_eval_only(task: TaskConfig, train_cfg: TrainingConfig, run_num: int = 1, tag: str = ""):
    print(f"\n{'='*55}")
    print(f"  Task (eval): {task.name}  [run {run_num}]")
    print(f"{'='*55}")

    set_seed(run_num)

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

    _cleanup(model, tokenizer)
