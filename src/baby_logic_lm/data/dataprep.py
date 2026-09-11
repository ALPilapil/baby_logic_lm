"""
dataprep.py — tokenizes and saves all three datasets to disk.

Run after format.py has produced the raw text / jsonl files (and, for
--paren, after make_paren.py has produced tokenized_paren.txt).

Usage
-----
    python -m baby_logic_lm.data.dataprep              # use base pythia tokenizer
    python -m baby_logic_lm.data.dataprep --paren       # use paren tokenizer

Datasets are saved to tokenizer-specific subdirectories so that running
with different tokenizers never overwrites each other:

    base tokenizer  →  ./data/base/nt_dataset, ./data/base/nsp_dataset, ...
    paren tokenizer →  ./data/paren/nt_dataset   (Dyck sequences only —
                        no NSP/NUP counterpart exists for this condition)
"""

import argparse
import json
import logging
from itertools import chain

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ── Next-token dataset ────────────────────────────────────────────────────────

def make_nt_dataset(read_path: str, save_path: str, tokenizer, block_size: int = 512):
    """
    Reads plain text, cleans it, chunks into fixed-length blocks, saves to disk.
    """
    dataset = load_dataset("text", data_files=read_path, split="train")
    logger.info("Loaded %s lines from %s", f"{len(dataset):,}", read_path)

    def clean_and_tokenize(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    tokenized = dataset.map(
        clean_and_tokenize,
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=["text"],
        keep_in_memory=False,
        desc="Cleaning and tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples}
        total = (len(concatenated["input_ids"]) // block_size) * block_size
        return {k: [concatenated[k][i : i + block_size]
                    for i in range(0, total, block_size)]
                for k in concatenated}

    chunked = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        keep_in_memory=False,
        desc=f"Chunking into {block_size}-token blocks",
    )

    logger.info("Created %s chunks", f"{len(chunked):,}")
    split = chunked.train_test_split(test_size=0.1, seed=42)
    split.save_to_disk(save_path)
    logger.info("Saved NT dataset to %s", save_path)
    return split


# ── Sequence-pair datasets (NSP / NUP) ───────────────────────────────────────

def make_pair_dataset(
    jsonl_path: str,
    save_path: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.1,
):
    """
    Tokenizes {"s1", "s2"} pairs for next-sentence / next-utterance prediction.
    Labels mask s1 so the model only predicts s2.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(pairs)
    logger.info("Loaded %s pairs from %s", f"{len(dataset):,}", jsonl_path)

    half = max_length // 2

    def tokenize_pairs(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for s1, s2 in zip(examples["s1"], examples["s2"]):
            t1 = tokenizer(s1, truncation=True, max_length=half,
                           add_special_tokens=False)["input_ids"]
            t2 = tokenizer(s2, truncation=True, max_length=half,
                           add_special_tokens=False)["input_ids"]

            # [s1] <eos> [s2] <eos>
            ids   = t1 + [tokenizer.eos_token_id] + t2 + [tokenizer.eos_token_id]
            masks = [1] * len(ids)
            # Mask s1 + its eos from the loss
            labels = [-100] * (len(t1) + 1) + t2 + [tokenizer.eos_token_id]

            # Hard truncate if still too long
            ids, masks, labels = (x[:max_length] for x in (ids, masks, labels))

            input_ids_list.append(ids)
            attention_mask_list.append(masks)
            labels_list.append(labels)

        return {"input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "labels": labels_list}

    processed = dataset.map(
        tokenize_pairs,
        batched=True,
        batch_size=1000,
        remove_columns=["s1", "s2"],
        num_proc=1,
        keep_in_memory=False,
        desc="Tokenizing pairs",
    )

    split = processed.train_test_split(test_size=test_size, seed=42)
    DatasetDict({"train": split["train"], "test": split["test"]}).save_to_disk(save_path)
    logger.info(
        "Saved pair dataset to %s (train %s / test %s)",
        save_path, f"{len(split['train']):,}", f"{len(split['test']):,}",
    )
    return split


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paren", action="store_true",
                        help="Use the paren tokenizer instead of the base model tokenizer")
    args = parser.parse_args()

    if args.paren:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizers/paren_tokenizer")
        logger.info("Using tokenizer: ./tokenizers/paren_tokenizer")
        logger.info("Saving dataset to: ./data/paren/")

        # Dyck pre-training trains on the special-token-converted integer
        # sequences (see make_paren.py), not on CHILDES text.
        make_nt_dataset(
            "./pre-predata/tokenized_paren/tokenized_paren.txt",
            "./data/paren/nt_dataset",
            tokenizer,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    logger.info("Using tokenizer: EleutherAI/pythia-160m")
    logger.info("Saving datasets to: ./data/base/")

    make_nt_dataset("./data/nt_text.txt", "./data/base/nt_dataset", tokenizer)
    make_pair_dataset("./data/nsp_text.jsonl", "./data/base/nsp_dataset", tokenizer)
    make_pair_dataset("./data/nup_text.jsonl", "./data/base/nup_dataset", tokenizer)


if __name__ == "__main__":
    from baby_logic_lm.logging_utils import setup_logging

    setup_logging()
    main()
