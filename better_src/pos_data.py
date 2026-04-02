"""
pos_data.py — generates the POS-tag dataset from C4.

Optionally creates (or reuses) the POS tokenizer first.

Usage
-----
    python scripts/pos_data.py              # create tokenizer + dataset
    python scripts/pos_data.py --no-tok     # skip tokenizer creation
"""

import argparse
import os

import nltk
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

import modify_tokenizer

# Penn Treebank POS tags (exhaustive; already known at design time)
POS_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
    "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$",
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
    "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "``", "''",
]

TOK_SAVE_DIR  = "./tokenizers/pos_tokenizer"
DATA_SAVE_DIR = "./data/pos_dataset"
TARGET_TOKENS = 100_000_000
TRAIN_RATIO   = 0.85
MAX_LENGTH    = 512


def text_to_pos(text: str) -> tuple[str, int]:
    """Convert raw text to a space-separated string of POS tags."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    pos_string = " ".join(tag for _, tag in tagged)
    return pos_string, len(tagged)


def build_pos_dataset(tokenizer) -> DatasetDict:
    train_threshold = int(TARGET_TOKENS * TRAIN_RATIO)
    train_data, test_data = [], []
    running_count = 0

    print(f"Target: {TARGET_TOKENS:,} tokens  "
          f"(train ≤ {train_threshold:,}, test = remainder)")

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    for example in dataset:
        pos_text, _ = text_to_pos(example["text"])

        tokenized = tokenizer(
            pos_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        count = len(tokenized["input_ids"])
        record = {
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        if running_count < train_threshold:
            train_data.append(record)
        elif running_count < TARGET_TOKENS:
            test_data.append(record)
        else:
            break

        running_count += count

        if running_count % 1_000_000 == 0:
            print(f"  {running_count:,} / {TARGET_TOKENS:,} tokens  "
                  f"(train {len(train_data):,}, test {len(test_data):,})",
                  end="\r")

    print()  # newline after \r progress

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test":  Dataset.from_list(test_data),
    })


def main(create_tokenizer: bool = True):
    if create_tokenizer:
        base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        print("Creating POS tokenizer …")
        modify_tokenizer.main(
            tokenizer      = base_tokenizer,
            unique_strings = POS_TAGS,
            save_dir       = TOK_SAVE_DIR,
        )

    tokenizer = AutoTokenizer.from_pretrained(TOK_SAVE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    print("Building POS dataset …")
    dataset_dict = build_pos_dataset(tokenizer)

    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    dataset_dict.save_to_disk(DATA_SAVE_DIR)
    print(f"Saved to {DATA_SAVE_DIR}  "
          f"(train {len(dataset_dict['train']):,}, "
          f"test {len(dataset_dict['test']):,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer creation (use existing)")
    args = parser.parse_args()
    main(create_tokenizer=not args.no_tok)
