"""
make_paren.py — prepares the Dyck/parentheses pre-training data and tokenizer.

Steps
-----
1. Convert raw dyck_sequences.txt (integers) → special-token format (<0>, <1>, …)
2. Create and save the paren tokenizer.

Usage
-----
    python -m baby_logic_lm.data.make_paren            # process all lines
    python -m baby_logic_lm.data.make_paren 50000       # limit to first 50 000 lines
"""

import logging
import os
import re
import sys

from transformers import AutoTokenizer

from baby_logic_lm.data import modify_tokenizer

logger = logging.getLogger(__name__)

INPUT_PATH  = "pre-predata/shuff_dyck/dyck_sequences.txt"
OUTPUT_PATH = "pre-predata/tokenized_paren/tokenized_paren.txt"
TOK_SAVE    = "tokenizers/paren_tokenizer"
CHUNK_SIZE  = 10_000


def convert_paren_file(input_path: str, output_path: str, limit: int | None = None):
    """
    Replace bare integers with special-token format: 3 → <3>.
    Processes the file in chunks to stay memory-efficient.
    """
    lines_done = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        buffer = []

        for line in infile:
            buffer.append(re.sub(r"\b(\d+)\b", r"<\1>", line))
            lines_done += 1

            if len(buffer) >= CHUNK_SIZE:
                outfile.writelines(buffer)
                buffer = []
                logger.info("Processed %s lines ...", f"{lines_done:,}")

            if limit is not None and lines_done >= limit:
                break

        if buffer:
            outfile.writelines(buffer)

    logger.info("Done — %s lines -> %s", f"{lines_done:,}", output_path)


def build_paren_tokenizer(dyck_path: str = INPUT_PATH):
    """Add Dyck integer tokens to the base tokenizer and save it."""
    unique_strings = modify_tokenizer.load_unique_paren_tokens(dyck_path)
    logger.info("Found %d unique paren tokens", len(unique_strings))

    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    modify_tokenizer.main(
        tokenizer      = base_tokenizer,
        unique_strings = unique_strings,
        save_dir       = TOK_SAVE,
    )


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        logger.info("Processing first %s lines", f"{limit:,}")

    logger.info("Step 1: converting integers to special tokens ...")
    convert_paren_file(INPUT_PATH, OUTPUT_PATH, limit=limit)

    logger.info("Step 2: building paren tokenizer ...")
    build_paren_tokenizer(dyck_path=INPUT_PATH)


if __name__ == "__main__":
    from baby_logic_lm.logging_utils import setup_logging

    setup_logging()
    main()
