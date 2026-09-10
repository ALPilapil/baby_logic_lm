"""Thin wrapper: `python scripts/format.py` -> baby_logic_lm.data.format.main()"""

from baby_logic_lm.data.format import main
from baby_logic_lm.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
