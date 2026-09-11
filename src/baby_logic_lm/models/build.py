"""
models/build.py — model / tokenizer / collator construction.

Architecture hyperparameters come from a local, pinned ModelConfig (see
config_schema.py) instead of an implicit AutoConfig.from_pretrained() Hub
fetch, so a from-scratch model's exact architecture is versioned in-repo.
Warm-starting from a checkpoint (model_load_path set) is unaffected -- it
already loads locally-saved weights.
"""

import logging

from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from baby_logic_lm.collators.pair_collator import CustomDataCollator
from baby_logic_lm.config_schema import ModelConfig, TaskConfig

logger = logging.getLogger(__name__)


def build_tokenizer_and_collator(task: TaskConfig):
    tokenizer = AutoTokenizer.from_pretrained(task.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    if task.use_custom_collator:
        collator = CustomDataCollator(tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenizer, collator


def build_model(model_cfg: ModelConfig, task: TaskConfig, tokenizer) -> GPTNeoXForCausalLM:
    if task.model_load_path is None:
        # model_cfg may be a plain ModelConfig or a Hydra DictConfig backed by
        # that schema (methods aren't accessible on the latter), so convert
        # explicitly rather than relying on ModelConfig.to_hf_kwargs().
        model_kwargs = {
            k: v for k, v in OmegaConf.to_container(model_cfg, resolve=True).items()
            if k != "name"
        }
        hf_config = GPTNeoXConfig(**model_kwargs)
        model = GPTNeoXForCausalLM(hf_config)  # __init__ already runs _init_weights via post_init()
        logger.info("Initialized fresh model from local architecture config '%s'", model_cfg.name)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(task.model_load_path)
        logger.info("Loaded model weights from %s", task.model_load_path)

    model.resize_token_embeddings(len(tokenizer))
    return model
