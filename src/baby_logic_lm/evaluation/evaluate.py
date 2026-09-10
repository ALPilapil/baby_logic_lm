import logging
import math
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import torch

from baby_logic_lm.config_schema import BLIMP_DIR, CN_DATA_PATH

logger = logging.getLogger(__name__)


# ── CN helper functions ──────────────────────────────────────────────────────

def read_syntax_data(filepath):
    """
    Read syntax evaluation data line by line, preserving the tab-separated format.

    Args:
        filepath (str): Path to the data file

    Returns:
        list: List of tuples (condition, sentence) where condition is the
              syntactic manipulation and sentence is the test sentence
    """
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            parts = line.split('\t')

            if len(parts) == 2:
                condition, sentence = parts
                data.append((condition.strip(), sentence.strip()))
            elif len(parts) == 1:
                # Handle lines that might be cut off (like the last line)
                logger.warning("Line %d appears incomplete: %s", line_num, line)
                condition = parts[0].strip()
                data.append((condition, ""))
            elif len(parts) == 3:
                condition = parts[0]
                sentence = parts[2]
                data.append((condition.strip(), sentence.strip()))
            else:
                logger.warning("Line %d has unexpected format: %s, parts: %s", line_num, line, parts)

    return data


def CN_format(scores_log):
    """
    Format the CN results into a cleaner format: for each sentence position,
    track how many times the model ranked it 1st, 2nd, etc. by NLL.
    """
    position_rank_counts = defaultdict(lambda: defaultdict(int))

    for sublist in scores_log:
        # rankings[0] is the position of the smallest NLL, rankings[1] the
        # 2nd-smallest, etc.
        rankings = np.argsort(sublist)
        for rank, position in enumerate(rankings):
            position_rank_counts[position][rank + 1] += 1  # 1-indexed ranks

    return {
        int(position): dict(position_rank_counts[position])
        for position in sorted(position_rank_counts)
    }


def process_blimp_score(total_score):
    """total_score is a list of (good_nll, bad_nll) pairs; return the fraction
    where the model assigns lower NLL to the grammatical (good) sentence."""
    correct = sum(1 for good, bad in total_score if good < bad)
    return correct / len(total_score)


# ── Main eval class ──────────────────────────────────────────────────────────

class Evaluation:
    def __init__(self, model, tokenizer, eval_results, truncation=None, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_results = eval_results
        self.truncation = truncation
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        # Right-padding is required: causal attention lets trailing pad tokens
        # not affect earlier real-token predictions, so per-example NLL stays
        # correct regardless of batch composition.
        self.tokenizer.padding_side = "right"

    def nll_batch(self, sentences):
        """
        Compute total NLL for each sentence via batched forward passes on
        self.device. Returns a list of floats aligned with `sentences`.
        """
        results = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = sentences[i:i + self.batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]
            shift_mask = inputs["attention_mask"][:, 1:].float()

            token_nll = torch.nn.functional.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, reduction="none"
            )
            nll_per_example = (token_nll * shift_mask).sum(dim=1)
            results.extend(nll_per_example.tolist())

        return results

    def CN_test(self, file_path):
        """
        Read the CN test sentences and record, for each position in the
        12-way minimal set, how often the model ranks it 1st, 2nd, etc. by NLL.
        """
        test_set = read_syntax_data(file_path)
        if self.truncation:
            test_set = test_set[:self.truncation]

        # Batch every 12 lines (one full minimal set per batch)
        candidates = []
        scores_log = []

        for i, (_, sentence) in enumerate(test_set, 1):
            candidates.append(sentence)
            if i % 12 == 0:
                scores_log.append(self.nll_batch(candidates))
                candidates = []

        return CN_format(scores_log)

    # ---------------- BLiMP ----------------

    def run_test(self, file_path):
        """Return the fraction of good vs. bad sentences the model picks correctly."""
        good_sentences, bad_sentences = [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                good_sentences.append(data["sentence_good"])
                bad_sentences.append(data["sentence_bad"])

        good_scores = self.nll_batch(good_sentences)
        bad_scores = self.nll_batch(bad_sentences)
        return process_blimp_score(list(zip(good_scores, bad_scores)))

    def run_blimp(self, path):
        """
        Each jsonl file in the blimp_tests folder is one test case (67 total).
        Returns a dict of {test case name: good-vs-bad ratio}.
        """
        folder = Path(path)
        test_files_paths = list(folder.glob("*.jsonl"))
        if self.truncation:
            test_files_paths = test_files_paths[:self.truncation]

        return {
            file_path.stem: self.run_test(file_path)
            for file_path in test_files_paths
        }

    def eval(self, CN, blimp):
        self.blimp = None
        self.CN = None

        if self.eval_results is None:
            self.perplexity = None
            self.CEL = None
        else:
            self.perplexity = math.exp(self.eval_results["eval_loss"])
            self.CEL = self.eval_results["eval_loss"]

        if CN:
            logger.info("Running CN")
            self.CN = self.CN_test(CN_DATA_PATH)

        if blimp:
            logger.info("Running BLiMP")
            blimps = self.run_blimp(BLIMP_DIR)
            self.blimp = sum(blimps.values()) / len(blimps)
            logger.info("BLiMP: %s", self.blimp)
