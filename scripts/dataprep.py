"""Thin wrapper: `python scripts/dataprep.py [--paren]` -> baby_logic_lm.data.dataprep.main()"""

from baby_logic_lm.data.dataprep import main
from baby_logic_lm.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
