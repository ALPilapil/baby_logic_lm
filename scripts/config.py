"""
config.py — single source of truth for all training variants.

To train a new model variation, add a TaskConfig to TASK_CONFIGS and
add its name to `tasks_to_run` in train.py. Nothing else needs to change.
"""

from dataclasses import dataclass, field
from typing import Optional

# ── Shared constants ──────────────────────────────────────────────────────────

BASE_MODEL_ID   = "EleutherAI/pythia-160m"
RESULTS_CSV     = "./training_results.csv"
CN_DATA_PATH    = "./evals/cn/crain-and-nakayama-breakdown.txt.data"
BLIMP_DIR       = "./evals/blimp_tests"

# ── Training hyperparameters ──────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Hyperparameters shared across all tasks. Override per-task if needed."""
    num_train_epochs:             int   = 1
    per_device_train_batch_size:  int   = 8
    per_device_eval_batch_size:   int   = 8
    learning_rate:                float = 2.5e-4
    lr_scheduler_type:            str   = "cosine"
    warmup_ratio:                 float = 0.05
    adam_beta1:                   float = 0.9
    adam_beta2:                   float = 0.999
    weight_decay:                 float = 0.01
    eval_strategy:                str   = "steps"
    eval_steps:                   int   = 500
    logging_steps:                int   = 500
    save_steps:                   int   = 5000
    save_total_limit:             int   = 10
    report_to:                    str   = "wandb"

# ── Per-task configuration ────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    """
    Everything that differs between training runs.

    Fields
    ------
    name              : human-readable label used in logs and the results CSV
    data_path         : path to a HuggingFace DatasetDict saved with save_to_disk()
    model_save_path   : where to write the trained model (also used as output_dir)
    tokenizer_path    : HF model id or local path (defaults to base model)
    model_load_path   : local path to load weights from; None = fresh random init
    use_custom_collator : True for sequence-pair tasks (NSP / NUP)
    train_truncation  : cap training examples (None = full dataset)
    test_truncation   : cap eval examples    (None = full dataset)
    eval_truncation   : cap CN evaluation    (None = full suite)
    run_cn            : whether to run the CN evaluation
    run_blimp         : whether to run the BLiMP evaluation
    """
    name:                str
    data_path:           str
    model_save_path:     str
    tokenizer_path:      str           = BASE_MODEL_ID
    model_load_path:     Optional[str] = None
    use_custom_collator: bool          = False
    train_truncation:    Optional[int] = None
    test_truncation:     Optional[int] = None
    eval_truncation:     Optional[int] = None
    run_cn:              bool          = True
    run_blimp:           bool          = True

# ── Preset task registry ──────────────────────────────────────────────────────

TASK_CONFIGS: dict[str, TaskConfig] = {

    # 1. Vanilla next-token prediction on CHILDES
    "next_word": TaskConfig(
        name             = "next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/nt-model",
    ),

    # 2. POS-tag pre-training on C4
    "pos_pretrain": TaskConfig(
        name             = "pos_pretrain",
        data_path        = "./data/pos_dataset",
        model_save_path  = "./models/pythia/pos-model",
        tokenizer_path   = "./tokenizers/pos_tokenizer",
    ),

    # 3. Dyck / parentheses pre-training
    "paren_pretrain": TaskConfig(
        name             = "paren_pretrain",
        data_path        = "./data/paren/nt_dataset",
        model_save_path  = "./models/pythia/paren-model",
        tokenizer_path   = "./tokenizers/paren_tokenizer",
    ),

    # 4. Next-sentence prediction (fine-tune from scratch)
    "nsp": TaskConfig(
        name                = "nsp",
        data_path           = "./data/base/nsp_dataset",
        model_save_path     = "./models/pythia/nsp-model",
        use_custom_collator = True,
    ),

    # 5. Next-utterance prediction (fine-tune from scratch)
    "nup": TaskConfig(
        name                = "nup",
        data_path           = "./data/base/nup_dataset",
        model_save_path     = "./models/pythia/nup-model",
        use_custom_collator = True,
    ),

    # 6. POS pre-train → next-word fine-tune (two-stage)
    "pos_then_next_word": TaskConfig(
        name             = "pos_then_next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/pos-nt-model",
        model_load_path  = "./models/pythia/pos-model",  # warm-start
    ),

    # 7. Paren pre-train → next-word fine-tune (two-stage)
    "paren_then_next_word": TaskConfig(
        name             = "paren_then_next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/paren-nt-model",
        model_load_path  = "./models/pythia/paren-model",
    ),

    # 8. Next-word + NUP (train on NUP after next-word)
    "next_word_then_nup": TaskConfig(
        name                = "next_word_then_nup",
        data_path           = "./data/base/nup_dataset",
        model_save_path     = "./models/pythia/nt-nup-model",
        model_load_path     = "./models/pythia/nt-model",
        use_custom_collator = True,
    ),
}
