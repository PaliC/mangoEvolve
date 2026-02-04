#!/bin/bash
# Run LLM-SRBench benchmark across domains
#
# Usage:
#   ./scripts/run_llm_srbench.sh [domain] [options]
#
# Examples:
#   ./scripts/run_llm_srbench.sh phys                    # Run all physics problems
#   ./scripts/run_llm_srbench.sh all --max-problems 3    # Run 3 problems per domain
#   ./scripts/run_llm_srbench.sh chem,phys               # Run chemistry and physics
#   ./scripts/run_llm_srbench.sh --list                  # List available domains
#   ./scripts/run_llm_srbench.sh phys --dry-run          # Preview without running
#   ./scripts/run_llm_srbench.sh --show-results          # Show results from last run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_DIR/problems/symbolic_regression/experiments/benchmark"

cd "$PROJECT_DIR"

# Function to display results from saved benchmark_results.json
show_results() {
    local results_file="${1:-$BENCHMARK_DIR/benchmark_results.json}"

    if [ ! -f "$results_file" ]; then
        echo "No results file found at: $results_file"
        echo "Run a benchmark first or specify a path with --show-results <path>"
        exit 1
    fi

    echo ""
    echo "================================================================================"
    echo "BENCHMARK RESULTS"
    echo "================================================================================"
    echo ""

    # Parse JSON and display table using Python (since jq may not handle nested structures well)
    uv run python << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

results_file = sys.argv[1] if len(sys.argv) > 1 else "problems/symbolic_regression/experiments/benchmark/benchmark_results.json"

try:
    with open(results_file) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Results file not found: {results_file}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Invalid JSON in: {results_file}")
    sys.exit(1)

results = data.get("results", [])
if not results:
    print("No results in file.")
    sys.exit(0)

# Header
print(f"{'Problem':<25} {'Status':<10} {'Best Score':<12} {'MSE Train':<14} {'MSE Test':<14}")
print("-" * 80)

# Sort by domain then problem index
sorted_results = sorted(results, key=lambda x: (x.get("domain", ""), x.get("problem_index", 0)))

total_cost = 0.0
successful = 0
failed = 0

for r in sorted_results:
    domain = r.get("domain", "?")
    idx = r.get("problem_index", 0)
    problem_name = f"{domain}_{idx:03d}"

    if r.get("success"):
        successful += 1
        summary = r.get("summary", {})
        best_score = summary.get("best_score", "N/A")
        cost = summary.get("total_cost", 0.0)
        total_cost += cost if isinstance(cost, (int, float)) else 0

        # Get best trial metrics
        exp_dir = r.get("dir")
        mse_train = "N/A"
        mse_test = "N/A"

        if exp_dir:
            exp_path = Path(exp_dir)
            trial_files = list(exp_path.glob("generations/*/trial_*.json"))
            best_trial_score = float("-inf")
            for tf in trial_files:
                try:
                    with open(tf) as f:
                        trial = json.load(f)
                    if trial.get("metrics", {}).get("valid"):
                        score = trial["metrics"].get("score", 0)
                        if score > best_trial_score:
                            best_trial_score = score
                            mse_train = trial["metrics"].get("mse_train", "N/A")
                            mse_test = trial["metrics"].get("mse_test", "N/A")
                except:
                    pass

        # Format values
        if isinstance(best_score, (int, float)):
            best_score_str = f"{best_score:.4f}"
        else:
            best_score_str = str(best_score)

        if isinstance(mse_train, (int, float)):
            mse_train_str = f"{mse_train:.6e}"
        else:
            mse_train_str = str(mse_train)

        if isinstance(mse_test, (int, float)):
            mse_test_str = f"{mse_test:.6e}"
        else:
            mse_test_str = str(mse_test)

        print(f"{problem_name:<25} {'OK':<10} {best_score_str:<12} {mse_train_str:<14} {mse_test_str:<14}")
    else:
        failed += 1
        error = str(r.get("error", "Unknown"))[:30]
        print(f"{problem_name:<25} {'FAILED':<10} {'-':<12} {'-':<14} {error:<14}")

print("-" * 80)
print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed} | Cost: ${total_cost:.4f}")
print("=" * 80)
PYTHON_SCRIPT
}

# Check if LLM-SRBench data exists
DATA_DIR="$PROJECT_DIR/problems/symbolic_regression/data/llm_srbench"

# Handle special flags
if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    echo "Available LLM-SRBench domains:"
    echo ""
    for domain in bio chem matsci phys transform; do
        if [ -d "$DATA_DIR/$domain" ]; then
            count=$(ls -d "$DATA_DIR/$domain"/*_* 2>/dev/null | wc -l | tr -d ' ')
            echo "  $domain: $count problems"
        else
            echo "  $domain: not downloaded"
        fi
    done
    exit 0
fi

if [ "$1" == "--show-results" ]; then
    shift
    show_results "$@"
    exit 0
fi

# Default to showing help if no args
if [ $# -eq 0 ]; then
    echo "LLM-SRBench Benchmark Runner"
    echo ""
    echo "Usage: $0 [domain] [options]"
    echo ""
    echo "Domains:"
    echo "  all       - Run all domains"
    echo "  phys      - Physics (oscillation)"
    echo "  chem      - Chemistry (reactions)"
    echo "  bio       - Biology (population growth)"
    echo "  matsci    - Materials Science"
    echo "  transform - Transformed physical models"
    echo ""
    echo "Options:"
    echo "  --max-problems N    Run at most N problems per domain"
    echo "  --problems 0,1,2    Run specific problem indices"
    echo "  --output-dir DIR    Output directory for results"
    echo "  --base-config FILE  Base config for LLM settings"
    echo "  --dry-run           Show what would run without executing"
    echo "  --list, -l          List available domains"
    echo "  --show-results      Show results from last benchmark run"
    echo ""
    echo "Examples:"
    echo "  $0 phys --max-problems 5"
    echo "  $0 all --max-problems 2 --dry-run"
    echo "  $0 chem,phys --problems 0,1,2"
    echo "  $0 --show-results"
    echo ""
    echo "Press Ctrl+C during a run to stop early - results will still be displayed."
    exit 0
fi

# Check if data exists
if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/index.json" ]; then
    echo "Error: LLM-SRBench data not found at: $DATA_DIR"
    echo ""
    echo "To download the dataset:"
    echo "  1. Create a Hugging Face account: https://huggingface.co/join"
    echo "  2. Accept the dataset terms: https://huggingface.co/datasets/nnheui/llm-srbench"
    echo "  3. Create an access token: https://huggingface.co/settings/tokens"
    echo "  4. Add the token to your .env file: HF_TOKEN=hf_your_token_here"
    echo "  5. Run: python -m problems.symbolic_regression.scripts.setup_llm_srbench"
    exit 1
fi

# Run the Python benchmark script
# The Python script handles SIGINT itself and will display results on interrupt
exec uv run python -m problems.symbolic_regression.scripts.run_benchmark --domain "$@"
