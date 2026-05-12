"""
config.py — single source of truth for all training variants.

To train a new model variation, add a TaskConfig to TASK_CONFIGS and
add its name to `tasks_to_run` in train.py. Nothing else needs to change.
"""

from dataclasses import dataclass, field
from typing import Optional

# ── Shared constants ──────────────────────────────────────────────────────────

BASE_MODEL_ID      = "EleutherAI/pythia-160m"
RESULTS_CSV        = "./training_results.csv"
CN_DATA_PATH       = "./evals/cn/crain-and-nakayama-breakdown.txt.data"
BLIMP_DIR          = "./evals/blimp_tests"
BABYLM_EVAL_DIR    = "/home/alpilapi/projects/babylm-eval/strict"
BABYLM_RESULTS_CSV = "./babylm_results.csv"

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
    lock_epochs:         bool          = False  # if True, --epochs does not override num_train_epochs
    run_babylm_eval:     bool          = False  # if True, runs babylm-eval zero-shot suite after this task

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
        train_truncation = None,   # set at runtime via --pretrain-tokens
    ),

    # ── 100M pre-training stages ─────────────────────────────────────────────
    # train_truncation set at runtime via --pretrain-tokens 39600000

    "dyck_pretrain_100m": TaskConfig(
        name             = "dyck_pretrain_100m",
        data_path        = "./data/paren/nt_dataset",
        model_save_path  = "./models/pythia/dyck_100m_model",
        tokenizer_path   = "./tokenizers/paren_tokenizer",
        num_train_epochs = 1,
        train_truncation = None,
    ),

    "pos_pretrain_100m": TaskConfig(
        name             = "pos_pretrain_100m",
        data_path        = "./data/pos_dataset",
        model_save_path  = "./models/pythia/pos_100m_model",
        tokenizer_path   = "./tokenizers/pos_tokenizer",
        num_train_epochs = 1,
        train_truncation = None,
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
        run_babylm_eval  = True,
    ),

    # 2. POS pre-train → NTP fine-tune
    "pos_then_next_word": TaskConfig(
        name             = "pos_then_next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/pos-nt-model",
        model_load_path  = "./models/pythia/pos-model",
        num_train_epochs = 1,
        run_babylm_eval  = True,
    ),

    # 3. Paren pre-train → NTP fine-tune
    "paren_then_next_word": TaskConfig(
        name             = "paren_then_next_word",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/paren-nt-model",
        model_load_path  = "./models/pythia/paren-model",
        num_train_epochs = 1,
        run_babylm_eval  = True,
    ),

    # 4. NTP → NSP fine-tune
    "next_word_then_nsp": TaskConfig(
        name                = "next_word_then_nsp",
        data_path           = "./data/base/nsp_dataset",
        model_save_path     = "./models/pythia/nsp-model",
        model_load_path     = "./models/pythia/nt-model",
        num_train_epochs    = 1,
        use_custom_collator = True,
        run_babylm_eval     = True,
    ),

    # 5. NTP → NUP fine-tune
    "next_word_then_nup": TaskConfig(
        name                = "next_word_then_nup",
        data_path           = "./data/base/nup_dataset",
        model_save_path     = "./models/pythia/nt-nup-model",
        model_load_path     = "./models/pythia/nt-model",
        num_train_epochs    = 1,
        use_custom_collator = True,
        run_babylm_eval     = True,
    ),

    # ── 10M conditions (all 1 epoch, split datasets) ──────────────────────────
    # Datasets in data/split/ created by scripts/make_split_datasets.py

    # Baseline
    "ntp_10m": TaskConfig(
        name             = "ntp_10m",
        data_path        = "./data/split/nt_10m",
        model_save_path  = "./models/pythia/ntp_10m_model",
        num_train_epochs = 1,
        run_babylm_eval  = True,
    ),

    # Post-training NSP: stage 1 (NTP on first 5M), stage 2 (NSP on second 5M)
    "ntp_10m_for_nsp": TaskConfig(
        name             = "ntp_10m_for_nsp",
        data_path        = "./data/split/nt_5m_a",
        model_save_path  = "./models/pythia/ntp_10m_nsp_model",
        num_train_epochs = 1,
    ),
    "nsp_10m": TaskConfig(
        name                = "nsp_10m",
        data_path           = "./data/split/nsp_5m_b",
        model_save_path     = "./models/pythia/nsp_10m_model",
        model_load_path     = "./models/pythia/ntp_10m_nsp_model",
        num_train_epochs    = 1,
        use_custom_collator = True,
        run_babylm_eval     = True,
    ),

    # Post-training NUP: stage 1 (NTP on first 5M), stage 2 (NUP on second 5M)
    "ntp_10m_for_nup": TaskConfig(
        name             = "ntp_10m_for_nup",
        data_path        = "./data/split/nt_5m_a",
        model_save_path  = "./models/pythia/ntp_10m_nup_model",
        num_train_epochs = 1,
    ),
    "nup_10m": TaskConfig(
        name                = "nup_10m",
        data_path           = "./data/split/nup_5m_b",
        model_save_path     = "./models/pythia/nup_10m_model",
        model_load_path     = "./models/pythia/ntp_10m_nup_model",
        num_train_epochs    = 1,
        use_custom_collator = True,
        run_babylm_eval     = True,
    ),

    # Dyck pre-training: 5M paren → random 5M CHILDES (train_truncation ≈ 9765 examples)
    "dyck_5m_childes": TaskConfig(
        name             = "dyck_5m_childes",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/dyck_5m_childes_model",
        model_load_path  = "./models/pythia/paren-model",
        num_train_epochs = 1,
        train_truncation = 9765,   # 9765 × 512 ≈ 5M tokens
        run_babylm_eval  = True,
    ),

    # POS pre-training: 5M POS → random 5M CHILDES (train_truncation ≈ 9765 examples)
    "pos_5m_childes": TaskConfig(
        name             = "pos_5m_childes",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/pos_5m_childes_model",
        model_load_path  = "./models/pythia/pos-model",
        num_train_epochs = 1,
        train_truncation = 9765,   # 9765 × 512 ≈ 5M tokens
        run_babylm_eval  = True,
    ),

    # ── 100M conditions (lock_epochs=True; epochs set in config, not --epochs) ─
    # Hard cap: ≤100M tokens per condition.
    # train_truncation sets examples/epoch so that epochs × truncation × 512 = target.
    #   baseline:      4 × 48828 × 512 = 99,999,744  (~100M)
    #   post-training: (4 × 24414 × 512) × 2 stages  = 99,997,696  (~100M)
    #   pre-training:  50M pre-train + 2 × 48828 × 512 = 49,999,872  (~50M CHILDES)

    # Baseline: 4 epochs, truncated to 48828 examples → 99,999,744 tokens
    "ntp_100m": TaskConfig(
        name             = "ntp_100m",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/ntp_100m_model",
        num_train_epochs = 4,
        train_truncation = 48828,
        lock_epochs      = True,
        run_babylm_eval  = True,
    ),

    # Post-training NSP: 4 epochs × 24414 examples each stage → 50M + 50M = ~100M
    "ntp_100m_for_nsp": TaskConfig(
        name             = "ntp_100m_for_nsp",
        data_path        = "./data/split/nt_half_a",
        model_save_path  = "./models/pythia/ntp_100m_nsp_model",
        num_train_epochs = 4,
        train_truncation = 24414,
        lock_epochs      = True,
    ),
    "nsp_100m": TaskConfig(
        name                = "nsp_100m",
        data_path           = "./data/split/nsp_half_b",
        model_save_path     = "./models/pythia/nsp_100m_model",
        model_load_path     = "./models/pythia/ntp_100m_nsp_model",
        num_train_epochs    = 4,
        train_truncation    = 24414,
        use_custom_collator = True,
        lock_epochs         = True,
        run_babylm_eval     = True,
    ),

    # Post-training NUP: 4 epochs × 24414 examples each stage → 50M + 50M = ~100M
    "ntp_100m_for_nup": TaskConfig(
        name             = "ntp_100m_for_nup",
        data_path        = "./data/split/nt_half_a",
        model_save_path  = "./models/pythia/ntp_100m_nup_model",
        num_train_epochs = 4,
        train_truncation = 24414,
        lock_epochs      = True,
    ),
    "nup_100m": TaskConfig(
        name                = "nup_100m",
        data_path           = "./data/split/nup_half_b",
        model_save_path     = "./models/pythia/nup_100m_model",
        model_load_path     = "./models/pythia/ntp_100m_nup_model",
        num_train_epochs    = 4,
        train_truncation    = 24414,
        use_custom_collator = True,
        lock_epochs         = True,
        run_babylm_eval     = True,
    ),

    # Dyck 100M: 50M paren → 2 epochs × 48828 examples CHILDES → 49,999,872 tokens (~50M)
    "dyck_100m_childes": TaskConfig(
        name             = "dyck_100m_childes",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/dyck_100m_childes_model",
        model_load_path  = "./models/pythia/dyck_100m_model",
        num_train_epochs = 2,
        train_truncation = 48828,
        lock_epochs      = True,
        run_babylm_eval  = True,
    ),

    # POS 100M: 50M POS → 2 epochs × 48828 examples CHILDES → 49,999,872 tokens (~50M)
    "pos_100m_childes": TaskConfig(
        name             = "pos_100m_childes",
        data_path        = "./data/base/nt_dataset",
        model_save_path  = "./models/pythia/pos_100m_childes_model",
        model_load_path  = "./models/pythia/pos_100m_model",
        num_train_epochs = 2,
        train_truncation = 48828,
        lock_epochs      = True,
        run_babylm_eval  = True,
    ),
}
