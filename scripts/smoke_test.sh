#!/bin/bash
set -e

# Pre-flight check before running a real (multi-hour) suite: trains 3 tiny
# fixtures that exercise checkpoint-chaining, the split-phase/token_limit
# training path used by every 100M-suite condition, and real CN/BLiMP
# scoring -- then verifies training_results.csv logged sane values.
#
# Uses whatever configs/wandb/default.yaml currently has (mode: online by
# default), so this writes real runs (tagged "smoke_check") to your wandb
# account -- that's the point: open the dashboard afterward and confirm the
# CN/BLiMP charts look right.
#
# Cleans up the 3 disposable model checkpoints it creates; leaves the
# training_results.csv rows and wandb runs (tagged "smoke_check") in place
# so you can inspect them.

MODELS=(
    "./models/pythia/smoke_eval_pretrain_model"
    "./models/pythia/smoke_eval_finetune_model"
    "./models/pythia/smoke_split_phase_model"
)

cleanup() {
    rm -rf "${MODELS[@]}"
}
trap cleanup EXIT

echo "=== Running smoke fixtures (smoke_eval_pretrain -> smoke_eval_finetune, smoke_split_phase) ==="
poetry run python -m baby_logic_lm.cli.pipeline \
    --tasks smoke_eval_pretrain smoke_eval_finetune smoke_split_phase \
    --tag smoke_check \
    --mode both

echo ""
echo "=== Checking training_results.csv ==="
if poetry run python scripts/check_smoke_results.py; then
    status=0
else
    status=$?
fi

echo ""
if [ "$status" -eq 0 ]; then
    echo "Smoke test passed. Check your wandb dashboard (project baby-logic-lm, tag smoke_check)"
    echo "to confirm the 3 runs are grouped/tagged correctly and the CN/BLiMP charts render."
else
    echo "Smoke test FAILED -- see errors above."
fi

exit "$status"
