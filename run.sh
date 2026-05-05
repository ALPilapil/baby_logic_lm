#!/bin/bash
set -e

# ── Experiment parameters ─────────────────────────────────────────────────────
RUNS=3
EPOCHS=1
PRETRAIN_TOKENS=10000000   # 10M — token-matched to CHILDES for fair comparison
TAG="initial runs"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Baseline NTP + post-training conditions
#    next_word must precede next_word_then_nsp/nup — they warm-start from its checkpoint.
python main.py \
    --tasks next_word next_word_then_nsp next_word_then_nup \
    --epochs "$EPOCHS" \
    --runs "$RUNS" \
    --tag "$TAG"

# 2. POS pre-training → NTP fine-tune
# python main.py \
#     --tasks pos_pretrain pos_then_next_word \
#     --epochs "$EPOCHS" \
#     --pretrain-tokens "$PRETRAIN_TOKENS" \
#     --runs "$RUNS" \
#     --tag "$TAG"

# # 3. Paren pre-training → NTP fine-tune
# python main.py \
#     --tasks paren_pretrain paren_then_next_word \
#     --epochs "$EPOCHS" \
#     --pretrain-tokens "$PRETRAIN_TOKENS" \
#     --runs "$RUNS" \
#     --tag "$TAG"
