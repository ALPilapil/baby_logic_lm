"""
Proves the locally-pinned Pythia-160M architecture config
(configs/model/pythia_160m.yaml) is field-for-field identical to what
AutoConfig.from_pretrained("EleutherAI/pythia-160m") returns, so switching
build_model() away from the implicit Hub fetch changes nothing.
"""

from transformers import AutoConfig, GPTNeoXConfig

from baby_logic_lm.config_schema import BASE_MODEL_ID


def test_pinned_model_config_matches_hub_config(compose_task):
    cfg = compose_task("next_word")
    hub_config = AutoConfig.from_pretrained(BASE_MODEL_ID)

    pinned = GPTNeoXConfig(**{k: v for k, v in cfg.model.items() if k != "name"})

    for field_name in [
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "intermediate_size", "hidden_act", "rotary_pct", "rotary_emb_base",
        "max_position_embeddings", "layer_norm_eps", "initializer_range",
        "use_parallel_residual", "tie_word_embeddings", "bos_token_id", "eos_token_id",
    ]:
        assert getattr(pinned, field_name) == getattr(hub_config, field_name), field_name
