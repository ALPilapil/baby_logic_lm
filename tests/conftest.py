from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from baby_logic_lm.config_schema import register_configs

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

register_configs()


@pytest.fixture
def compose_task():
    """Returns a function(task_name, overrides=()) -> resolved Config for that task."""

    def _compose(task_name: str, overrides=()):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="config", overrides=[f"task={task_name}", *overrides])

    return _compose
