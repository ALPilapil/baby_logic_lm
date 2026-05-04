"""
train.py — training pipeline.

To train a different model variation, edit `tasks_to_run` in main().
Everything else (hyperparams, paths, tokenizers) is in config.py.
"""

import csv
import gc
import math
import os

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


# ── Training ──────────────────────────────────────────────────────────────────

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
    args = TrainingArguments(
        output_dir                   = task.model_save_path,
        num_train_epochs             = task.num_train_epochs,
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
        report_to                    = cfg.report_to,
        seed                         = seed,
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        tokenizer     = tokenizer,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = collator,
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


def save_results(evaluation: Evaluation, task_name: str, run_num: int = 1, filename: str = RESULTS_CSV):
    row = {
        "run":        run_num,
        "task_type":  task_name,
        "CEL":        evaluation.CEL,
        "perplexity": evaluation.perplexity,
        "CN":         evaluation.CN,
        "BLiMP":      evaluation.blimp,
    }
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Results saved to {filename}")


# ── Top-level pipeline ────────────────────────────────────────────────────────

def run_task(task: TaskConfig, train_cfg: TrainingConfig, run_num: int = 1):
    print(f"\n{'='*55}")
    print(f"  Task: {task.name}  [run {run_num}]")
    print(f"{'='*55}")

    set_seed(run_num)

    tokenizer, collator = build_tokenizer_and_collator(task)
    model = build_model(task, tokenizer)

    dataset  = load_from_disk(task.data_path)
    train_ds = (dataset["train"].select(range(task.train_truncation))
                if task.train_truncation else dataset["train"])
    eval_ds  = (dataset["test"].select(range(task.test_truncation))
                if task.test_truncation else dataset["test"])

    train_eval_results = train(model, tokenizer, train_ds, eval_ds,
                               collator, task, train_cfg, seed=run_num)

    evaluation = evaluate(task, tokenizer, train_eval_results)
    save_results(evaluation, task.name, run_num)

    # Free GPU memory before the next task
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    train_cfg = TrainingConfig()

    # ── Pick which tasks to run ──────────────────────────────────────────────
    # For two-stage pre-training conditions, run the PRETRAIN_CONFIGS stage
    # first (once) to produce the checkpoint, then run the TASK_CONFIGS stage.
    tasks_to_run = [
        TASK_CONFIGS["next_word"],
        # PRETRAIN_CONFIGS["pos_pretrain"],   # run once to produce checkpoint
        # TASK_CONFIGS["pos_then_next_word"],
        # PRETRAIN_CONFIGS["paren_pretrain"], # run once to produce checkpoint
        # TASK_CONFIGS["paren_then_next_word"],
        # TASK_CONFIGS["next_word_then_nsp"],
        # TASK_CONFIGS["next_word_then_nup"],
    ]
    # ────────────────────────────────────────────────────────────────────────

    for task in tasks_to_run:
        run_task(task, train_cfg)


if __name__ == "__main__":
    main()
