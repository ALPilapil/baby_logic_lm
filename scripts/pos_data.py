"""Thin wrapper: `python scripts/pos_data.py [--no-tok]` -> baby_logic_lm.data.pos_data.main()"""

import argparse

from baby_logic_lm.data.pos_data import main
from baby_logic_lm.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer creation (use existing)")
    args = parser.parse_args()
    main(create_tokenizer=not args.no_tok)
