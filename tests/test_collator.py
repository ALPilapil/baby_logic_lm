"""Regression fixtures for CustomDataCollator's right-padding / -100 label masking."""

from types import SimpleNamespace

from baby_logic_lm.collators.pair_collator import CustomDataCollator


def _fake_tokenizer(pad_token_id=0):
    return SimpleNamespace(eos_token="<eos>", pad_token_id=pad_token_id)


def test_pads_to_max_length_in_batch():
    collator = CustomDataCollator(_fake_tokenizer(pad_token_id=99))
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    batch = collator(examples)

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 3], [-100, 5, -100]]


def test_no_padding_needed_when_lengths_equal():
    collator = CustomDataCollator(_fake_tokenizer(pad_token_id=0))
    examples = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [-100, 4]},
    ]
    batch = collator(examples)

    assert batch["input_ids"].tolist() == [[1, 2], [3, 4]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 1]]
    assert batch["labels"].tolist() == [[-100, 2], [-100, 4]]
