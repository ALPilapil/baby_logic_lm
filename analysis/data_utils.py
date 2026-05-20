"""Shared data loading and metric utilities for the analysis pipeline."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paradigm metadata ────────────────────────────────────────────────────────

PARADIGM_ORDER = [
    "Baseline NTP",
    "Post-train NSP",
    "Post-train NUP",
    "Dyck Pre-train",
    "POS Pre-train",
]

# task_type → (paradigm, stage, token_scale)
TASK_META = {
    "ntp_10m":            ("Baseline NTP",   "final",    "10M"),
    "nsp_10m":            ("Post-train NSP", "final",    "10M"),
    "nup_10m":            ("Post-train NUP", "final",    "10M"),
    "dyck_5m_childes":    ("Dyck Pre-train", "final",    "10M"),
    "pos_5m_childes":     ("POS Pre-train",  "final",    "10M"),
    "ntp_10m_for_nsp":    ("Post-train NSP", "warmup",   "10M"),
    "ntp_10m_for_nup":    ("Post-train NUP", "warmup",   "10M"),
    "dyck_pretrain":      ("Dyck Pre-train", "pretrain", "10M"),
    "pos_pretrain":       ("POS Pre-train",  "pretrain", "10M"),
    "ntp_100m":           ("Baseline NTP",   "final",    "100M"),
    "nsp_100m":           ("Post-train NSP", "final",    "100M"),
    "nup_100m":           ("Post-train NUP", "final",    "100M"),
    "dyck_100m_childes":  ("Dyck Pre-train", "final",    "100M"),
    "pos_100m_childes":   ("POS Pre-train",  "final",    "100M"),
    "ntp_100m_for_nsp":   ("Post-train NSP", "warmup",   "100M"),
    "ntp_100m_for_nup":   ("Post-train NUP", "warmup",   "100M"),
    "dyck_pretrain_100m": ("Dyck Pre-train", "pretrain", "100M"),
    "pos_pretrain_100m":  ("POS Pre-train",  "pretrain", "100M"),
}

# ── CN condition metadata ────────────────────────────────────────────────────

# Positions 0-11 = 2 cycles of these 6 conditions (set A: 0-5, set B: 6-11)
_CN_CYCLE = [
    "prepose_first_and_delete_first",
    "prepose_first_and_delete_main",
    "prepose_first_and_delete_none",
    "prepose_main_and_delete_first",
    "prepose_main_and_delete_main",
    "prepose_main_and_delete_none",
]
CN_CONDITIONS = _CN_CYCLE + _CN_CYCLE

# Short column labels for heatmap x-axis
#   F = prepose first (wrong), M = prepose main (correct)
#   -F = delete first, -M = delete main, -N = delete none
CN_SHORT_LABELS = ["F-F", "F-M", "F-N", "M-F", "M-M", "M-N"] * 2

# Positions of the grammatically correct form (prepose_main_and_delete_none)
GRAMMATICAL_POSITIONS = [5, 11]

# Chance level for 12-way ranking
CHANCE = 1 / 12

# Matched pairs (pF_pos, pM_pos) sharing the same deletion condition.
# pF = prepose first (ungrammatical), pM = prepose main (grammatical).
# Comparing within-pair controls for sentence length.
#
#   deletion    set A           set B
#   delete_first  (0, 3)         (6, 9)
#   delete_main   (1, 4)         (7, 10)
#   delete_none   (2, 5)         (8, 11)
CN_MATCHED_PAIRS = [(0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)]

# ── Colour palette ───────────────────────────────────────────────────────────

PALETTE = {
    "Baseline NTP":    "#377eb8",
    "Post-train NSP":  "#e41a1c",
    "Post-train NUP":  "#ff7f00",
    "Dyck Pre-train":  "#4daf4a",
    "POS Pre-train":   "#984ea3",
}

# ── CN parsing and metrics ───────────────────────────────────────────────────

def parse_cn(cn_str) -> dict | None:
    """Parse the CN dict from a CSV cell string, return None if absent."""
    if cn_str is None:
        return None
    if isinstance(cn_str, float) and np.isnan(cn_str):
        return None
    s = str(cn_str).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def compute_cn_metrics(cn_dict: dict) -> dict:
    """
    Given a parsed CN dict {position: {rank: count}}, return:
      rank1_rates  – list[float, 12]: proportion of batches where this position ranked 1st
      mean_ranks   – list[float, 12]: mean rank assigned to each position
    """
    rank1_rates, mean_ranks = [], []
    for pos in range(12):
        dist = cn_dict.get(pos, {})
        total = sum(dist.values())
        if total == 0:
            rank1_rates.append(np.nan)
            mean_ranks.append(np.nan)
            continue
        rank1_rates.append(dist.get(1, 0) / total)
        mean_ranks.append(sum(r * c for r, c in dist.items()) / total)
    return {"rank1_rates": rank1_rates, "mean_ranks": mean_ranks}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(
    csv_path: str | Path = "training_results.csv",
    min_date: str = "2026-05-18",
) -> pd.DataFrame:
    """
    Load training_results.csv, filter to rows on or after min_date, and
    annotate with paradigm/scale/stage columns and derived CN metrics.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = pd.Timestamp(min_date, tz="UTC")
    df = df[df["timestamp"] >= cutoff].copy()

    # Paradigm / scale / stage annotations
    def _meta(task):
        m = TASK_META.get(task)
        return m if m else (None, None, None)

    meta_series = df["task_type"].map(_meta)
    df["paradigm"]    = meta_series.map(lambda x: x[0])
    df["stage"]       = meta_series.map(lambda x: x[1])
    df["token_scale"] = meta_series.map(lambda x: x[2])
    df["is_final"]    = df["stage"] == "final"

    # CN parsing
    cn_parsed  = df["CN"].map(parse_cn)
    cn_metrics = cn_parsed.map(lambda d: compute_cn_metrics(d) if d is not None else None)

    df["rank1_rates"] = cn_metrics.map(lambda m: m["rank1_rates"] if m else None)
    df["mean_ranks"]  = cn_metrics.map(lambda m: m["mean_ranks"]  if m else None)

    # CN grammar score: mean(mean_rank[pF] - mean_rank[pM]) across matched pairs.
    # Positive = model assigns higher rank (lower NLL) to pM (grammatical) = GOOD.
    # Negative = model prefers pF (ungrammatical).
    # Controls for sentence length by comparing only within length-matched pairs.
    def _cn_grammar_score(mean_ranks):
        if mean_ranks is None:
            return np.nan
        diffs = []
        for pF, pM in CN_MATCHED_PAIRS:
            mF, mM = mean_ranks[pF], mean_ranks[pM]
            if not (np.isnan(mF) or np.isnan(mM)):
                diffs.append(mF - mM)  # positive when model prefers pM (lower mean rank)
        return float(np.mean(diffs)) if diffs else np.nan

    df["cn_accuracy"] = df["mean_ranks"].map(_cn_grammar_score)
    return df
