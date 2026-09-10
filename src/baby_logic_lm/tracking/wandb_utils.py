"""
tracking/wandb_utils.py — Weights & Biases integration.

init_wandb() must be called before Trainer(...) is constructed: HF's
WandbCallback (triggered by TrainingConfig.report_to == "wandb", already
threaded into every TrainingArguments) attaches to an already-open
wandb.run instead of creating its own, so this is both necessary and
sufficient for the two to coexist.

training_results.csv (see training/results.py) remains the mandatory,
batch-analysis source of truth that analysis/ reads -- wandb tracking here
is purely additive (live dashboards + full resolved-config provenance).
"""

import logging
from typing import Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def init_wandb(cfg: DictConfig, task_name: str, run_num: int, tag: str, job_type: str):
    """Returns the wandb module (with an open run) if tracking is enabled,
    otherwise None. Safe to call unconditionally."""
    if cfg.training.report_to != "wandb" or cfg.wandb.mode == "disabled":
        return None

    import wandb

    run_name = f"{tag + '-' if tag else ''}{task_name}_run{run_num}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        group=task_name,
        tags=[tag] if tag else [],
        job_type=job_type,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return wandb


def log_final_metrics(wandb_mod, evaluation) -> None:
    if wandb_mod is None:
        return
    wandb_mod.log({
        "final/CEL": evaluation.CEL,
        "final/perplexity": evaluation.perplexity,
        "final/BLiMP": evaluation.blimp,
    })
    if evaluation.CN is not None:
        wandb_mod.log({"final/CN": evaluation.CN})


def finish(wandb_mod) -> None:
    if wandb_mod is not None:
        wandb_mod.finish()
