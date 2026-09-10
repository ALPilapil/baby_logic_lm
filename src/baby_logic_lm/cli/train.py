"""
cli/train.py — single-task Hydra entry point.

    python -m baby_logic_lm.cli.train task=ntp_10m training.num_train_epochs=5

Also unlocks real Hydra multirun sweeps for free, e.g.:

    python -m baby_logic_lm.cli.train -m task=ntp_10m,ntp_100m \
        training.learning_rate=1e-4,2.5e-4

For the checkpoint-chained --tasks A B ... sequences used by the full
experiment suites, use baby_logic_lm.cli.pipeline instead -- Hydra's
multirun is for independent trials, not stateful task chains.

Hydra already configures logging (with per-run log-file routing) before
this function runs -- do not call baby_logic_lm.logging_utils.setup_logging()
here, it would clobber that.
"""

import hydra
from omegaconf import DictConfig

from baby_logic_lm.config_schema import register_configs
from baby_logic_lm.training.runner import run_task

register_configs()


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_task(cfg, run_num=cfg.seed, tag=cfg.tag)


if __name__ == "__main__":
    main()
