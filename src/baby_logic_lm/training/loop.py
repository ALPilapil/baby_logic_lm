"""
training/loop.py — the core Trainer-building / training-loop logic.

Handles the split-phase training path used when TaskConfig.token_limit is
set (the 100M-token conditions): N-1 full epochs, then a truncated final
epoch resumed from the latest checkpoint, so the token budget isn't
exceeded by a fractional final epoch.
"""

import glob
import os
from typing import Optional

from transformers import Trainer, TrainingArguments

from baby_logic_lm.config_schema import TaskConfig, TrainingConfig


def _last_epoch_truncation(task: TaskConfig, train_ds) -> Optional[int]:
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


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path of the most recent checkpoint saved under output_dir."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _make_training_args(task: TaskConfig, cfg: TrainingConfig, seed: int,
                        num_epochs: int, report_to: Optional[str] = None) -> TrainingArguments:
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
