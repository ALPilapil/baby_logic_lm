"""Thin wrapper: `python scripts/make_paren.py [limit]` -> baby_logic_lm.data.make_paren.main()"""

from baby_logic_lm.data.make_paren import main
from baby_logic_lm.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
