"""
Regression test locking the Hydra config tree (configs/task/*.yaml) to a
frozen snapshot of the field values from the original scripts/config.py
TASK_CONFIGS/PRETRAIN_CONFIGS dicts (captured before that file was deleted
during the Hydra migration -- see the migration plan). Guards against
accidental drift in the YAML configs.
"""

import pytest

# Frozen snapshot of the pre-migration scripts/config.py dicts, field-for-field.
LEGACY_CONFIGS = {
    "dyck_100m_childes": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": "./models/pythia/dyck_100m_model",
        "model_save_path": "./models/pythia/dyck_100m_childes_model",
        "name": "dyck_100m_childes", "num_train_epochs": 2, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 50000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 48828, "use_custom_collator": False,
    },
    "dyck_5m_childes": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/paren-model",
        "model_save_path": "./models/pythia/dyck_5m_childes_model",
        "name": "dyck_5m_childes", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 9765, "use_custom_collator": False,
    },
    "dyck_pretrain": {
        "data_path": "./data/paren/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/paren-model",
        "name": "dyck_pretrain", "num_train_epochs": 1, "run_blimp": False, "run_cn": False,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "./tokenizers/paren_tokenizer",
        "train_truncation": None, "use_custom_collator": False,
    },
    "dyck_pretrain_100m": {
        "data_path": "./data/paren/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/dyck_100m_model",
        "name": "dyck_pretrain_100m", "num_train_epochs": 1, "run_blimp": False, "run_cn": False,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "./tokenizers/paren_tokenizer",
        "train_truncation": None, "use_custom_collator": False,
    },
    "next_word": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/nt-model",
        "name": "next_word", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": False,
    },
    "next_word_then_nsp": {
        "data_path": "./data/base/nsp_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/nt-model", "model_save_path": "./models/pythia/nsp-model",
        "name": "next_word_then_nsp", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": True,
    },
    "next_word_then_nup": {
        "data_path": "./data/base/nup_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/nt-model", "model_save_path": "./models/pythia/nt-nup-model",
        "name": "next_word_then_nup", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": True,
    },
    "nsp_100m": {
        "data_path": "./data/split/nsp_half_b", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": "./models/pythia/ntp_100m_nsp_model",
        "model_save_path": "./models/pythia/nsp_100m_model",
        "name": "nsp_100m", "num_train_epochs": 4, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 50000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 24414, "use_custom_collator": True,
    },
    "nsp_10m": {
        "data_path": "./data/split/nsp_5m_b", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/ntp_10m_nsp_model",
        "model_save_path": "./models/pythia/nsp_10m_model",
        "name": "nsp_10m", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": True,
    },
    "ntp_100m": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": None, "model_save_path": "./models/pythia/ntp_100m_model",
        "name": "ntp_100m", "num_train_epochs": 4, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 100000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 48828, "use_custom_collator": False,
    },
    "ntp_100m_for_nsp": {
        "data_path": "./data/split/nt_half_a", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": None, "model_save_path": "./models/pythia/ntp_100m_nsp_model",
        "name": "ntp_100m_for_nsp", "num_train_epochs": 4, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 50000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 24414, "use_custom_collator": False,
    },
    "ntp_100m_for_nup": {
        "data_path": "./data/split/nt_half_a", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": None, "model_save_path": "./models/pythia/ntp_100m_nup_model",
        "name": "ntp_100m_for_nup", "num_train_epochs": 4, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 50000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 24414, "use_custom_collator": False,
    },
    "ntp_10m": {
        "data_path": "./data/split/nt_10m", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/ntp_10m_model",
        "name": "ntp_10m", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": False,
    },
    "ntp_10m_for_nsp": {
        "data_path": "./data/split/nt_5m_a", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/ntp_10m_nsp_model",
        "name": "ntp_10m_for_nsp", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": False,
    },
    "ntp_10m_for_nup": {
        "data_path": "./data/split/nt_5m_a", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/ntp_10m_nup_model",
        "name": "ntp_10m_for_nup", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": False,
    },
    "nup_100m": {
        "data_path": "./data/split/nup_half_b", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": "./models/pythia/ntp_100m_nup_model",
        "model_save_path": "./models/pythia/nup_100m_model",
        "name": "nup_100m", "num_train_epochs": 4, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 50000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 24414, "use_custom_collator": True,
    },
    "nup_10m": {
        "data_path": "./data/split/nup_5m_b", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/ntp_10m_nup_model",
        "model_save_path": "./models/pythia/nup_10m_model",
        "name": "nup_10m", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": True,
    },
    "paren_then_next_word": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/paren-model",
        "model_save_path": "./models/pythia/paren-nt-model",
        "name": "paren_then_next_word", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": False,
    },
    "pos_100m_childes": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": True,
        "model_load_path": "./models/pythia/pos_100m_model",
        "model_save_path": "./models/pythia/pos_100m_childes_model",
        "name": "pos_100m_childes", "num_train_epochs": 2, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": 50000000, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 48828, "use_custom_collator": False,
    },
    "pos_5m_childes": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/pos-model",
        "model_save_path": "./models/pythia/pos_5m_childes_model",
        "name": "pos_5m_childes", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": 9765, "use_custom_collator": False,
    },
    "pos_pretrain": {
        "data_path": "./data/pos_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/pos-model",
        "name": "pos_pretrain", "num_train_epochs": 1, "run_blimp": False, "run_cn": False,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "./tokenizers/pos_tokenizer",
        "train_truncation": None, "use_custom_collator": False,
    },
    "pos_pretrain_100m": {
        "data_path": "./data/pos_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": None, "model_save_path": "./models/pythia/pos_100m_model",
        "name": "pos_pretrain_100m", "num_train_epochs": 1, "run_blimp": False, "run_cn": False,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "./tokenizers/pos_tokenizer",
        "train_truncation": None, "use_custom_collator": False,
    },
    "pos_then_next_word": {
        "data_path": "./data/base/nt_dataset", "eval_truncation": None, "lock_epochs": False,
        "model_load_path": "./models/pythia/pos-model",
        "model_save_path": "./models/pythia/pos-nt-model",
        "name": "pos_then_next_word", "num_train_epochs": 1, "run_blimp": True, "run_cn": True,
        "test_truncation": None, "token_limit": None, "tokenizer_path": "EleutherAI/pythia-160m",
        "train_truncation": None, "use_custom_collator": False,
    },
}

PRETRAIN_NAMES = {"dyck_pretrain", "dyck_pretrain_100m", "pos_pretrain", "pos_pretrain_100m"}

COMPARABLE_FIELDS = [
    "name", "data_path", "model_save_path", "num_train_epochs", "tokenizer_path",
    "model_load_path", "use_custom_collator", "train_truncation", "test_truncation",
    "eval_truncation", "run_cn", "run_blimp", "lock_epochs", "token_limit",
]


@pytest.mark.parametrize("task_name", sorted(LEGACY_CONFIGS))
def test_task_config_matches_frozen_snapshot(compose_task, task_name):
    old = LEGACY_CONFIGS[task_name]
    cfg = compose_task(task_name)
    new = cfg.task

    for field_name in COMPARABLE_FIELDS:
        old_value = old[field_name]
        new_value = new[field_name] if field_name in new else None
        assert old_value == new_value, (
            f"{task_name}.{field_name}: legacy={old_value!r} vs hydra={new_value!r}"
        )


@pytest.mark.parametrize("task_name", sorted(PRETRAIN_NAMES))
def test_pretrain_tasks_flagged(compose_task, task_name):
    assert compose_task(task_name).task.is_pretrain is True


@pytest.mark.parametrize("task_name", sorted(set(LEGACY_CONFIGS) - PRETRAIN_NAMES))
def test_non_pretrain_tasks_not_flagged(compose_task, task_name):
    assert compose_task(task_name).task.is_pretrain is False


def test_all_task_yaml_files_are_covered():
    from pathlib import Path

    task_yaml_dir = Path(__file__).resolve().parents[1] / "configs" / "task"
    yaml_names = {p.stem for p in task_yaml_dir.glob("*.yaml")}
    smoke_fixtures = {n for n in yaml_names if n.startswith("smoke")}
    assert yaml_names - smoke_fixtures == set(LEGACY_CONFIGS)
