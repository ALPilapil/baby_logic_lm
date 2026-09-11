"""
pos_data.py — generates the POS-tag dataset from C4.

Optionally creates (or reuses) the POS tokenizer first.

Usage
-----
    python -m baby_logic_lm.data.pos_data              # create tokenizer + dataset
    python -m baby_logic_lm.data.pos_data --no-tok      # skip tokenizer creation
"""

import argparse
import logging
import os

import nltk
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from baby_logic_lm.data import modify_tokenizer

logger = logging.getLogger(__name__)

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

    logger.info(
        "Target: %s tokens (train <= %s, test = remainder)",
        f"{TARGET_TOKENS:,}", f"{train_threshold:,}",
    )

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
            logger.info(
                "%s / %s tokens (train %s, test %s)",
                f"{running_count:,}", f"{TARGET_TOKENS:,}",
                f"{len(train_data):,}", f"{len(test_data):,}",
            )

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test":  Dataset.from_list(test_data),
    })


def main(create_tokenizer: bool = True):
    if create_tokenizer:
        base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        logger.info("Creating POS tokenizer ...")
        modify_tokenizer.main(
            tokenizer      = base_tokenizer,
            unique_strings = POS_TAGS,
            save_dir       = TOK_SAVE_DIR,
        )

    tokenizer = AutoTokenizer.from_pretrained(TOK_SAVE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Building POS dataset ...")
    dataset_dict = build_pos_dataset(tokenizer)

    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    dataset_dict.save_to_disk(DATA_SAVE_DIR)
    logger.info(
        "Saved to %s (train %s, test %s)",
        DATA_SAVE_DIR, f"{len(dataset_dict['train']):,}", f"{len(dataset_dict['test']):,}",
    )


if __name__ == "__main__":
    from baby_logic_lm.logging_utils import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer creation (use existing)")
    args = parser.parse_args()
    main(create_tokenizer=not args.no_tok)
