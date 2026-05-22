#!/bin/bash
set -e

# Run babylm-eval zero-shot and fine-tuning evaluations on all final model checkpoints,
# iterating over all 3 runs per condition. Results from the eval pipeline go to
# ../babylm-eval/strict/results/<model>/. A summary row per run is appended to
# babylm_eval_results.csv in this directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BABYLM_STRICT="$(cd "$SCRIPT_DIR/../babylm-eval/strict" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models/pythia"
ARCH="causal"
RESULTS_CSV="$SCRIPT_DIR/babylm_eval_results.csv"

# Model stems — actual directories are <stem>_run1 / _run2 / _run3
MODEL_STEMS=(
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

for stem in "${MODEL_STEMS[@]}"; do
    for run in 1 2 3; do
        model="${stem}_run${run}"
        model_path="$MODELS_DIR/$model"

        if [ ! -d "$model_path" ]; then
            echo "Skipping $model — not found at $model_path"
            continue
        fi

        echo ""
        echo "══════════════════════════════════════════"
        echo "  Evaluating: $model"
        echo "══════════════════════════════════════════"

        echo "  [1/2] Zero-shot (BLiMP, COMPS, entity tracking, reading)"
        bash scripts/eval_zero_shot.sh "$model_path" "$ARCH"

        echo "  [2/2] Fine-tuning (GLUE)"
        bash scripts/eval_finetuning.sh --model_path "$model_path"

        echo "  [collecting results → $RESULTS_CSV]"
        python3 "$SCRIPT_DIR/scripts/collect_babylm_results.py" \
            --model  "$model" \
            --stem   "$stem" \
            --run    "$run" \
            --results_dir "results" \
            --output_csv  "$RESULTS_CSV"
    done
done

echo ""
echo "Done. Results in: $BABYLM_STRICT/results/"
echo "Summary CSV: $RESULTS_CSV"
