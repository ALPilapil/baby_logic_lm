#!/bin/bash
set -e

# Run babylm-eval zero-shot and fine-tuning evaluations on all final model checkpoints.
# CN evaluation is handled separately by run_eval.sh (main.py --mode eval).
# Results are written to ../babylm-eval/strict/results/<model_stem>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BABYLM_STRICT="$(cd "$SCRIPT_DIR/../babylm-eval/strict" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models/pythia"
ARCH="causal"

# Final CHILDES-trained models — intermediate pre-training checkpoints excluded
MODELS=(
    # ── 10M suite ─────────────────────────────────────────────────────────────
    "ntp_10m_model"
    "nsp_10m_model"
    "nup_10m_model"
    "dyck_5m_childes_model"
    "pos_5m_childes_model"
    # ── 100M suite ────────────────────────────────────────────────────────────
    "ntp_100m_model"
    "nsp_100m_model"
    "nup_100m_model"
    "dyck_100m_childes_model"
    "pos_100m_childes_model"
)

cd "$BABYLM_STRICT"

for model in "${MODELS[@]}"; do
    model_path="$MODELS_DIR/$model"

    if [ ! -d "$model_path" ]; then
        echo "Skipping $model — not found at $model_path"
        continue
    fi

    echo ""
    echo "══════════════════════════════════════════"
    echo "  Evaluating: $model"
    echo "══════════════════════════════════════════"

    echo "  [1/2] Zero-shot (BLiMP, EWoK, COMPS, entity tracking, reading)"
    bash scripts/eval_zero_shot.sh "$model_path" "$ARCH"

    echo "  [2/2] Fine-tuning (GLUE)"
    bash scripts/eval_finetuning.sh --model_path "$model_path"
done

echo ""
echo "Done. Results in: $BABYLM_STRICT/results/"
