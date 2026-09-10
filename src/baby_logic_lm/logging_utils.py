"""
logging_utils.py — shared logging setup for non-Hydra entry points.

Do NOT call setup_logging() from Hydra-driven code (baby_logic_lm.cli.train)
-- Hydra already configures the root logger (with per-run log file routing)
before the app function runs, and calling this would clobber that. Only call
it from cli/pipeline.py's outer argparse flow and the data-prep scripts.
"""

import logging

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True
