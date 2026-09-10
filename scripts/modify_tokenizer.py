"""Thin wrapper: `python scripts/modify_tokenizer.py` -> baby_logic_lm.data.modify_tokenizer"""

from baby_logic_lm.data.modify_tokenizer import load_unique_paren_tokens
from baby_logic_lm.logging_utils import setup_logging

if __name__ == "__main__":
    import logging

    setup_logging()
    logger = logging.getLogger(__name__)
    dyck_path = "pre-predata/shuff_dyck/dyck_sequences.txt"
    tokens = load_unique_paren_tokens(dyck_path)
    logger.info("Unique paren tokens (%d): %s ...", len(tokens), tokens[:10])
