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

# ── Training hyperparameters (shared across all tasks) ────────────────────────

@dataclass
class TrainingConfig:
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

    Token-count strategy
    --------------------
    CHILDES tasks (TASK_CONFIGS):
        Always train on the full dataset. Control exposure via num_train_epochs.
        train_truncation should remain None.

    Pre-training tasks (PRETRAIN_CONFIGS):
        Datasets are large (C4 / generated). Control both token budget and
        epochs to match CHILDES exposure.

        train_truncation caps the number of training *examples*. Convert a
        target token budget to an example cap as follows:

            paren_pretrain  — examples are exactly 512 tokens (fixed blocks):
                examples = target_tokens // 512

            pos_pretrain    — examples are variable length (≤ 512 tokens);
                use avg tokens per example from the dataset as an estimate:
                examples ≈ target_tokens // avg_tokens_per_example

    Fields
    ------
    name                : label used in logs and the results CSV
    data_path           : HuggingFace DatasetDict saved with save_to_disk()
    model_save_path     : where to write the trained model
    num_train_epochs    : epochs over the (truncated) training set
    tokenizer_path      : HF model id or local path
    model_load_path     : weights to warm-start from; None = random init
    use_custom_collator : True for sequence-pair tasks (NSP / NUP)
    train_truncation    : cap training examples (None = full dataset)
    test_truncation     : cap eval examples    (None = full dataset)
    eval_truncation     : cap BLiMP test files (None = all 67)
    run_cn              : whether to run CN evaluation
    run_blimp           : whether to run BLiMP evaluation
    """
    name:                str
    data_path:           str
    model_save_path:     str
    num_train_epochs:    int           = 1
    tokenizer_path:      str           = BASE_MODEL_ID
    model_load_path:     Optional[str] = None
    use_custom_collator: bool          = False
    train_truncation:    Optional[int] = None
    test_truncation:     Optional[int] = None
    eval_truncation:     Optional[int] = None
    run_cn:              bool          = True
    run_blimp:           bool          = True

# ── Pre-training helpers ──────────────────────────────────────────────────────
# Run these first (once) to produce the checkpoints loaded by
# pos_then_next_word and paren_then_next_word.
#
# Set train_truncation to match the CHILDES NTP token budget:
#   paren:  train_truncation = <childes_tokens> // 512
#   pos:    train_truncation = <childes_tokens> // <avg_pos_tokens_per_example>

PRETRAIN_CONFIGS: dict[str, TaskConfig] = {

    "pos_pretrain": TaskConfig(
        name             = "pos_pretrain",
        data_path        = "./data/pos_dataset",
        model_save_path  = "./models/pythia/pos-model",
        tokenizer_path   = "./tokenizers/pos_tokenizer",
        num_train_epochs = 1,
        train_truncation = None,   # TODO: set to match CHILDES token budget
    ),

    "paren_pretrain": TaskConfig(
        name             = "paren_pretrain",
        data_path        = "./data/paren/nt_dataset",
        model_save_path  = "./models/pythia/paren-model",
        tokenizer_path   = "./tokenizers/paren_tokenizer",
        num_train_epochs = 1,
        train_truncation = None,   # TODO: set to <childes_tokens> // 512
    ),
}

# ── Experimental conditions ───────────────────────────────────────────────────
# All 5 conditions include NTP on CHILDES as a training stage.
# train_truncation is always None here — train on the full CHILDES dataset.
# Adjust num_train_epochs to control total CHILDES exposure.

TASK_CONFIGS: dict[str, TaskConfig] = {

    # 1. NTP only (baseline)
    "next_word": TaskConfig(
        name             = "next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/nt-model",
        num_train_epochs = 1,
    ),

    # 2. POS pre-train → NTP fine-tune
    "pos_then_next_word": TaskConfig(
        name             = "pos_then_next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/pos-nt-model",
        model_load_path  = "./models/pythia/pos-model",
        num_train_epochs = 1,
    ),

    # 3. Paren pre-train → NTP fine-tune
    "paren_then_next_word": TaskConfig(
        name             = "paren_then_next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/paren-nt-model",
        model_load_path  = "./models/pythia/paren-model",
        num_train_epochs = 1,
    ),

    # 4. NTP → NSP fine-tune
    "next_word_then_nsp": TaskConfig(
        name                = "next_word_then_nsp",
        data_path           = "./data/base/nsp_dataset",
        model_save_path     = "./models/pythia/nsp-model",
        model_load_path     = "./models/pythia/nt-model",
        num_train_epochs    = 1,
        use_custom_collator = True,
    ),

    # 5. NTP → NUP fine-tune
    "next_word_then_nup": TaskConfig(
        name                = "next_word_then_nup",
        data_path           = "./data/base/nup_dataset",
        model_save_path     = "./models/pythia/nt-nup-model",
        model_load_path     = "./models/pythia/nt-model",
        num_train_epochs    = 1,
        use_custom_collator = True,
    ),
}
