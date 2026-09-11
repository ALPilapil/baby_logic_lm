"""
config_schema.py — structured Hydra config schema; single source of truth
for the *shape* of every experiment config.

Concrete values live in configs/{model,training,wandb,task}/*.yaml. Adding a
new experimental condition means adding one YAML file under configs/task/
that sets whatever TaskConfig fields differ from the defaults below —
nothing in this file needs to change.
"""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

BASE_MODEL_ID = "EleutherAI/pythia-160m"
RESULTS_CSV = "./training_results.csv"
# Sidecar to RESULTS_CSV: one JSON object per run, same fields, but CN is
# kept as a real nested structure instead of a stringified dict -- easier to
# reslice/replot later without an ast.literal_eval round trip. Written
# alongside the CSV by save_results(); the CSV remains the schema analysis/
# depends on, this is purely additive.
RESULTS_JSONL = "./training_results.jsonl"
CN_DATA_PATH = "./evals/cn/crain-and-nakayama-breakdown.txt.data"
BLIMP_DIR = "./evals/blimp_tests"


@dataclass
class ModelConfig:
    """GPTNeoX architecture, pinned locally instead of fetched from the Hub
    at train time. Values verified against EleutherAI/pythia-160m's real
    config.json (see tests/test_model_build.py)."""

    name: str = "pythia-160m"
    vocab_size: int = 50304
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    rotary_pct: float = 0.25
    rotary_emb_base: int = 10000
    max_position_embeddings: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_parallel_residual: bool = True
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0


@dataclass
class TrainingConfig:
    """Optimizer/schedule hyperparameters shared across every task."""

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2.5e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 500
    save_steps: int = 5000
    save_total_limit: int = 10
    report_to: str = "wandb"


@dataclass
class TaskConfig:
    """Everything that differs between training runs.

    Fields
    ------
    name                : label used in logs, training_results.csv, and wandb run names
    data_path           : HuggingFace DatasetDict saved with save_to_disk()
    model_save_path     : where to write the trained model
    num_train_epochs    : epochs over the (truncated) training set
    tokenizer_path      : HF model id or local path
    model_load_path     : weights to warm-start from; None = random init
    use_custom_collator : True for sequence-pair tasks (NSP / NUP)
    train_truncation    : cap training examples (None = full dataset)
    test_truncation     : cap eval examples (None = full dataset)
    eval_truncation     : cap BLiMP test files / CN test-set size (None = all)
    run_cn              : whether to run CN evaluation
    run_blimp           : whether to run BLiMP evaluation (False for
                          pre-training-only checkpoints not meant to be
                          evaluated as English models)
    lock_epochs         : if True, --epochs does not override num_train_epochs
    token_limit         : if set, the last epoch is truncated to stay under this
    is_pretrain         : True for intermediate pre-training checkpoints
                          (replaces the old "name in PRETRAIN_CONFIGS" dict check)
    """

    name: str = "???"
    data_path: str = "???"
    model_save_path: str = "???"
    num_train_epochs: int = 1
    tokenizer_path: str = BASE_MODEL_ID
    model_load_path: Optional[str] = None
    use_custom_collator: bool = False
    train_truncation: Optional[int] = None
    test_truncation: Optional[int] = None
    eval_truncation: Optional[int] = None
    run_cn: bool = True
    run_blimp: bool = True
    lock_epochs: bool = False
    token_limit: Optional[int] = None
    is_pretrain: bool = False


@dataclass
class WandbConfig:
    project: str = "baby-logic-lm"
    entity: Optional[str] = None
    mode: str = "online"  # "online" | "offline" | "disabled"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 1
    tag: str = ""


def register_configs() -> None:
    """Register the structured-config schema with Hydra's ConfigStore.
    Must run once before any hydra.compose()/@hydra.main call."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    cs.store(group="model", name="schema", node=ModelConfig)
    cs.store(group="training", name="schema", node=TrainingConfig)
    cs.store(group="task", name="schema", node=TaskConfig)
    cs.store(group="wandb", name="schema", node=WandbConfig)
