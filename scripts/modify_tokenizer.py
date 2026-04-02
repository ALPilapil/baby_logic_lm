"""
modify_tokenizer.py — adds special tokens to a tokenizer and saves it.

This module is imported by pos_data.py and make_paren.py.
It can also be run standalone to inspect what tokens would be added.
"""

from transformers import AutoTokenizer


def add_tokens_and_save(tokenizer, unique_strings: list[str], save_dir: str) -> int:
    """
    Add `unique_strings` as special tokens to `tokenizer` and save to `save_dir`.

    Returns the number of tokens actually added (0 if all already present).
    """
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": unique_strings}
    )
    tokenizer.save_pretrained(save_dir)
    print(f"  Added {n_added} tokens → saved tokenizer to {save_dir}")
    return n_added


def load_unique_paren_tokens(file_path: str) -> list[str]:
    """
    Read a Dyck sequence file and return the unique integer tokens
    formatted as special tokens: 0 → '<0>', 1 → '<1>', …
    """
    unique_nums: set[int] = set()

    try:
        with open(file_path, "r") as f:
            for line in f:
                for token in line.split():
                    try:
                        unique_nums.add(int(token))
                    except ValueError:
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Dyck sequence file not found: {file_path}")

    return [f"<{n}>" for n in sorted(unique_nums)]


# Keep `main` as the public API used by pos_data.py / make_paren.py
def main(tokenizer, unique_strings: list[str], save_dir: str) -> int:
    return add_tokens_and_save(tokenizer, unique_strings, save_dir)


if __name__ == "__main__":
    # Quick sanity check: show what tokens would be added for the paren tokenizer
    dyck_path = "pre-predata/shuff_dyck/dyck_sequences.txt"
    tokens = load_unique_paren_tokens(dyck_path)
    print(f"Unique paren tokens ({len(tokens)}): {tokens[:10]} …")
