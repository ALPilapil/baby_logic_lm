"""Thin wrapper: `python scripts/make_split_datasets.py` -> baby_logic_lm.data.make_split_datasets.main()"""

from baby_logic_lm.data.make_split_datasets import main
from baby_logic_lm.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
