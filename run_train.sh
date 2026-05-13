#!/bin/bash
set -e

# ── Experiment parameters ─────────────────────────────────────────────────────
RUNS=3
TAG_10M="10m"
TAG_100M="100m"
# ─────────────────────────────────────────────────────────────────────────────

# ── Data prep (run once before training) ─────────────────────────────────────
# python scripts/make_split_datasets.py

# ══ 10M suite (≈10M tokens per condition) ════════════════════════════════════

# 1. Baseline NTP — 10M tokens, 1 epoch
python main.py \
    --tasks ntp_10m \
    --epochs 1 \
    --runs "$RUNS" \
    --tag "$TAG_10M" \
    --mode train

# 2. Post-training NSP — 5M NTP (stage 1) + 5M NSP (stage 2)
python main.py \
    --tasks ntp_10m_for_nsp nsp_10m \
    --epochs 1 \
    --runs "$RUNS" \
    --tag "$TAG_10M" \
    --mode train

# 3. Post-training NUP — 5M NTP (stage 1) + 5M NUP (stage 2)
python main.py \
    --tasks ntp_10m_for_nup nup_10m \
    --epochs 1 \
    --runs "$RUNS" \
    --tag "$TAG_10M" \
    --mode train

# 4. Dyck pre-training — 5M paren (pre-train) + random 5M CHILDES NTP (fine-tune)
python main.py \
    --tasks dyck_pretrain dyck_5m_childes \
    --epochs 1 \
    --pretrain-tokens 5000000 \
    --runs "$RUNS" \
    --tag "$TAG_10M" \
    --mode train

# 5. POS pre-training — 5M POS (pre-train) + random 5M CHILDES NTP (fine-tune)
python main.py \
    --tasks pos_pretrain pos_5m_childes \
    --epochs 1 \
    --pretrain-tokens 5000000 \
    --runs "$RUNS" \
    --tag "$TAG_10M" \
    --mode train

# ══ 100M suite (≤100M tokens per condition) ═══════════════════════════════════
# lock_epochs=True in config controls epoch counts for 100M tasks;
# --epochs 1 applies only to pre-training stages.
# token_limit in config enforces the strict 100M cap via last-epoch truncation.

# 1. Baseline NTP — 4 epochs × full CHILDES (≤100M tokens)
python main.py \
    --tasks ntp_100m \
    --runs "$RUNS" \
    --tag "$TAG_100M" \
    --mode train

# 2. Post-training NSP — 4 epochs × first half NTP (≤50M) + 4 epochs × second half NSP (≤50M)
python main.py \
    --tasks ntp_100m_for_nsp nsp_100m \
    --runs "$RUNS" \
    --tag "$TAG_100M" \
    --mode train

# 3. Post-training NUP — 4 epochs × first half NTP (≤50M) + 4 epochs × second half NUP (≤50M)
python main.py \
    --tasks ntp_100m_for_nup nup_100m \
    --runs "$RUNS" \
    --tag "$TAG_100M" \
    --mode train

# 4. Dyck pre-training — 50M paren (pre-train) + 2 epochs × full CHILDES (≤50M)
python main.py \
    --tasks dyck_pretrain_100m dyck_100m_childes \
    --epochs 1 \
    --pretrain-tokens 50000000 \
    --runs "$RUNS" \
    --tag "$TAG_100M" \
    --mode train

# 5. POS pre-training — 50M POS (pre-train) + 2 epochs × full CHILDES (≤50M)
python main.py \
    --tasks pos_pretrain_100m pos_100m_childes \
    --epochs 1 \
    --pretrain-tokens 50000000 \
    --runs "$RUNS" \
    --tag "$TAG_100M" \
    --mode train
