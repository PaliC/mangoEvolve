#!/bin/bash
# Get results from a MangoEvolve experiment
# Usage: ./scripts/get_results.sh <experiment_dir>
#        ./scripts/get_results.sh experiments/symbolic_regression_gemini_flash_20260114_155112

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 experiments/symbolic_regression_gemini_flash_20260114_155112"
    exit 1
fi

EXPERIMENT_DIR="$1"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory '$EXPERIMENT_DIR' not found"
    exit 1
fi

echo "========================================"
echo "Results for: $(basename "$EXPERIMENT_DIR")"
echo "========================================"
echo

# Summary
echo "## Summary"
TOTAL_TRIALS=$(find "$EXPERIMENT_DIR/generations" -name "trial_*.json" 2>/dev/null | wc -l | tr -d ' ')
SUCCESSFUL=$(find "$EXPERIMENT_DIR/generations" -name "trial_*.json" -exec jq -r 'select(.metrics.valid == true) | .trial_id' {} \; 2>/dev/null | wc -l | tr -d ' ')
echo "Total trials: $TOTAL_TRIALS"
echo "Successful: $SUCCESSFUL"
echo

# All trials table
echo "## All Trials (sorted by score)"
echo "trial_id        mse_train       mse_test        score"
echo "--------        ---------       --------        -----"
for f in "$EXPERIMENT_DIR"/generations/*/trial_*.json; do
    if [ -f "$f" ]; then
        jq -r 'if .metrics.valid then
            [.trial_id, (.metrics.mse_train // "N/A" | tostring), (.metrics.mse_test // "N/A" | tostring), (.metrics.score // 0 | tostring)]
            | @tsv
        else
            [.trial_id, "FAILED", "FAILED", "0"] | @tsv
        end' "$f" 2>/dev/null
    fi
done | sort -t$'\t' -k4 -rn | column -t -s$'\t'
echo

# Best trial
echo "## Best Trial"
BEST_FILE=$(for f in "$EXPERIMENT_DIR"/generations/*/trial_*.json; do
    if [ -f "$f" ]; then
        jq -r --arg f "$f" 'select(.metrics.valid == true) | [$f, (.metrics.score // 0)] | @tsv' "$f" 2>/dev/null
    fi
done | sort -t$'\t' -k2 -rn | head -1 | cut -f1)

if [ -n "$BEST_FILE" ] && [ -f "$BEST_FILE" ]; then
    jq '{
        trial_id,
        mse_train: .metrics.mse_train,
        mse_test: .metrics.mse_test,
        score: .metrics.score,
        optimized_params: .metrics.optimized_params
    }' "$BEST_FILE"

    echo
    echo "## Best Trial Code"
    jq -r '.code' "$BEST_FILE"
else
    echo "No successful trials found"
fi
