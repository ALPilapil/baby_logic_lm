"""
make_paren.py — prepares the Dyck/parentheses pre-training data and tokenizer.

Steps
-----
1. Convert raw dyck_sequences.txt (integers) → special-token format (<0>, <1>, …)
2. Create and save the paren tokenizer.

Usage
-----
    python scripts/make_paren.py            # process all lines
    python scripts/make_paren.py 50000      # limit to first 50 000 lines
"""

import re
import sys
import os

from transformers import AutoTokenizer

import modify_tokenizer

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
                print(f"  Processed {lines_done:,} lines …", end="\r")

            if limit is not None and lines_done >= limit:
                break

        if buffer:
            outfile.writelines(buffer)

    print(f"\n  Done — {lines_done:,} lines → {output_path}")


def build_paren_tokenizer(dyck_path: str = INPUT_PATH):
    """Add Dyck integer tokens to the base tokenizer and save it."""
    unique_strings = modify_tokenizer.load_unique_paren_tokens(dyck_path)
    print(f"  Found {len(unique_strings)} unique paren tokens")

    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    modify_tokenizer.main(
        tokenizer      = base_tokenizer,
        unique_strings = unique_strings,
        save_dir       = TOK_SAVE,
    )


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        print(f"Processing first {limit:,} lines")

    print("Step 1: converting integers to special tokens …")
    convert_paren_file(INPUT_PATH, OUTPUT_PATH, limit=limit)

    print("Step 2: building paren tokenizer …")
    build_paren_tokenizer(dyck_path=INPUT_PATH)


if __name__ == "__main__":
    main()
