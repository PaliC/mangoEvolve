#!/usr/bin/env python3
"""
Run LLM-SRBench benchmark across multiple domains and problems.

This script generates configs on the fly and runs MangoEvolve for each problem.

Usage:
    # Run all problems in physics domain
    python -m problems.symbolic_regression.scripts.run_benchmark --domain phys

    # Run all domains
    python -m problems.symbolic_regression.scripts.run_benchmark --domain all

    # Run specific domains
    python -m problems.symbolic_regression.scripts.run_benchmark --domain phys,chem

    # Run specific problems
    python -m problems.symbolic_regression.scripts.run_benchmark --domain phys --problems 0,1,2

    # Dry run (generate configs without running)
    python -m problems.symbolic_regression.scripts.run_benchmark --domain phys --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

# Domain info
DOMAINS = {
    "bio": {"name": "Biology (population growth)", "num_problems": 24},
    "chem": {"name": "Chemistry (reactions)", "num_problems": 36},
    "matsci": {"name": "Materials Science", "num_problems": 25},
    "phys": {"name": "Physics (oscillation)", "num_problems": 44},
    "transform": {"name": "Transformed physical models", "num_problems": 111},
}

# Global state for interrupt handling
_benchmark_state = {
    "output_dir": None,
    "results": [],
    "interrupted": False,
}


def get_domain_info(domain: str) -> dict:
    """Get domain info, loading from downloaded metadata if available."""
    data_dir = Path(__file__).parent.parent / "data" / "llm_srbench" / domain
    metadata_path = data_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return {
            "name": DOMAINS.get(domain, {}).get("name", domain),
            "num_problems": metadata.get("num_problems", DOMAINS.get(domain, {}).get("num_problems", 0)),
        }
    return DOMAINS.get(domain, {"name": domain, "num_problems": 0})


def generate_config(
    domain: str,
    problem_index: int,
    output_dir: Path,
    base_config: dict | None = None,
) -> dict:
    """Generate a config for a specific problem."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = base_config.copy() if base_config else {}

    # Experiment settings
    config["experiment"] = {
        "name": f"llm_srbench_{domain}_{problem_index:03d}_{timestamp}",
        "output_dir": str(output_dir),
    }

    # Default LLM settings if not provided
    if "root_llm" not in config:
        config["root_llm"] = {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "cost_per_million_input_tokens": 0.15,
            "cost_per_million_output_tokens": 0.60,
            "reasoning": {"enabled": True, "effort": "medium"},
        }

    if "child_llms" not in config:
        config["child_llms"] = [
            {
                "alias": "flash",
                "provider": "google",
                "model": "gemini-2.5-flash",
                "cost_per_million_input_tokens": 0.15,
                "cost_per_million_output_tokens": 0.60,
            }
        ]

    if "default_child_llm_alias" not in config:
        config["default_child_llm_alias"] = "flash"

    # Evolution settings
    if "evolution" not in config:
        config["evolution"] = {
            "max_generations": 3,
            "max_children_per_generation": 5,
        }

    # Calibration
    if "calibration" not in config:
        config["calibration"] = {"enabled": False}

    # Budget
    if "budget" not in config:
        config["budget"] = {"max_cost_usd": 1.0}

    # Evaluator - always set for LLM-SRBench
    config["evaluation"] = {
        "evaluator_fn": "problems.symbolic_regression.evaluator:SymbolicRegressionEvaluator",
        "evaluator_kwargs": {
            "benchmark": "llm_srbench",
            "domain": domain,
            "problem_index": problem_index,
            "n_params": 5,
            "timeout_seconds": 30.0,
        },
    }

    return config


def run_experiment(config: dict, dry_run: bool = False) -> dict | None:
    """Run a single experiment with the given config."""
    experiment_name = config["experiment"]["name"]

    if dry_run:
        print(f"  [DRY RUN] Would run: {experiment_name}")
        return None

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    try:
        print(f"  Running: {experiment_name}")
        print()  # Blank line before evolution output

        # Run with output streaming (not captured) so progress bars show in real-time
        result = subprocess.run(
            ["uv", "run", "python", "-m", "mango_evolve", "--config", config_path],
            text=True,
        )

        print()  # Blank line after evolution output

        if result.returncode != 0:
            return {"success": False, "error": f"Exit code {result.returncode}"}

        # Parse output for results
        output_dir = Path(config["experiment"]["output_dir"])
        # Find the experiment directory (most recent matching name pattern)
        exp_dirs = sorted(output_dir.glob(f"{config['experiment']['name'].rsplit('_', 1)[0]}*"))
        if exp_dirs:
            exp_dir = exp_dirs[-1]
            summary_path = exp_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    return {"success": True, "summary": json.load(f), "dir": str(exp_dir)}

        return {"success": True, "dir": str(output_dir)}

    finally:
        Path(config_path).unlink()


def save_incremental_results(output_dir: Path, results: list[dict]) -> None:
    """Save results incrementally to allow recovery on interrupt."""
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump({"results": results, "timestamp": datetime.now().isoformat()}, f, indent=2, default=str)


def print_summary_table(results: list[dict], interrupted: bool = False) -> None:
    """Print a summary table of benchmark results."""
    if not results:
        print("\nNo results to display.")
        return

    print("\n")
    print("=" * 80)
    if interrupted:
        print("BENCHMARK INTERRUPTED - Partial Results")
    else:
        print("BENCHMARK RESULTS")
    print("=" * 80)

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
            total_cost += cost

            # Get best trial metrics from experiment directory
            exp_dir = r.get("dir")
            mse_train = "N/A"
            mse_test = "N/A"

            if exp_dir:
                exp_path = Path(exp_dir)
                # Find best trial
                trial_files = list(exp_path.glob("generations/*/trial_*.json"))
                best_mse = float("inf")
                for tf in trial_files:
                    try:
                        with open(tf) as f:
                            trial = json.load(f)
                        if trial.get("metrics", {}).get("valid"):
                            score = trial["metrics"].get("score", 0)
                            if score > best_mse or mse_train == "N/A":
                                if score > best_mse:
                                    best_mse = score
                                mse_train = trial["metrics"].get("mse_train", "N/A")
                                mse_test = trial["metrics"].get("mse_test", "N/A")
                    except (json.JSONDecodeError, KeyError):
                        pass

            # Format values
            if isinstance(best_score, float):
                best_score_str = f"{best_score:.4f}"
            else:
                best_score_str = str(best_score)

            if isinstance(mse_train, float):
                mse_train_str = f"{mse_train:.6e}"
            else:
                mse_train_str = str(mse_train)

            if isinstance(mse_test, float):
                mse_test_str = f"{mse_test:.6e}"
            else:
                mse_test_str = str(mse_test)

            print(f"{problem_name:<25} {'OK':<10} {best_score_str:<12} {mse_train_str:<14} {mse_test_str:<14}")
        else:
            failed += 1
            error = r.get("error", "Unknown error")[:30]
            print(f"{problem_name:<25} {'FAILED':<10} {'-':<12} {'-':<14} {error:<14}")

    # Summary
    print("-" * 80)
    print(f"Total: {len(results)} experiments | Successful: {successful} | Failed: {failed} | Cost: ${total_cost:.4f}")
    print("=" * 80)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    _benchmark_state["interrupted"] = True
    print("\n\n*** Interrupted by user ***")
    print_summary_table(_benchmark_state["results"], interrupted=True)

    # Save final results
    if _benchmark_state["output_dir"]:
        save_incremental_results(Path(_benchmark_state["output_dir"]), _benchmark_state["results"])
        print(f"\nPartial results saved to: {_benchmark_state['output_dir']}/benchmark_results.json")

    sys.exit(130)  # Standard exit code for SIGINT


def check_llm_srbench_data() -> tuple[bool, str]:
    """Check if LLM-SRBench data is downloaded."""
    data_dir = Path(__file__).parent.parent / "data" / "llm_srbench"
    index_path = data_dir / "index.json"

    if not data_dir.exists() or not index_path.exists():
        return False, f"""
LLM-SRBench dataset not found at: {data_dir}

To download the dataset:
  1. Create a Hugging Face account: https://huggingface.co/join
  2. Accept the dataset terms: https://huggingface.co/datasets/nnheui/llm-srbench
  3. Create an access token: https://huggingface.co/settings/tokens
  4. Add the token to your .env file: HF_TOKEN=hf_your_token_here
  5. Run: python -m problems.symbolic_regression.scripts.setup_llm_srbench
"""

    # Check if any domain has data
    try:
        with open(index_path) as f:
            index = json.load(f)
        if index.get("total_problems", 0) == 0:
            return False, "LLM-SRBench data exists but contains no problems. Re-run setup script."
    except (json.JSONDecodeError, KeyError):
        return False, "LLM-SRBench index.json is corrupted. Re-run setup script."

    return True, ""


def run_benchmark(
    domains: list[str],
    problems: list[int] | None = None,
    output_dir: Path | None = None,
    base_config_path: Path | None = None,
    dry_run: bool = False,
    max_problems_per_domain: int | None = None,
) -> dict:
    """Run benchmark across specified domains and problems."""
    # Check if LLM-SRBench data exists
    data_ok, error_msg = check_llm_srbench_data()
    if not data_ok:
        print(error_msg)
        sys.exit(1)

    # Load base config if provided
    base_config = None
    if base_config_path and base_config_path.exists():
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        # Remove experiment-specific fields
        base_config.pop("experiment", None)
        base_config.pop("evaluation", None)

    # Default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "experiments" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set global state for interrupt handling
    _benchmark_state["output_dir"] = str(output_dir)
    _benchmark_state["results"] = []

    results = {"domains": {}, "total_experiments": 0, "successful": 0, "failed": 0}

    for domain in domains:
        if _benchmark_state["interrupted"]:
            break

        domain_info = get_domain_info(domain)
        num_problems = domain_info["num_problems"]

        if num_problems == 0:
            print(f"\nSkipping {domain}: no problems found (run setup script first)")
            continue

        print(f"\n{'='*60}")
        print(f"Domain: {domain} - {domain_info['name']}")
        print(f"Problems: {num_problems}")
        print("=" * 60)

        # Determine which problems to run
        if problems is not None:
            problem_indices = [p for p in problems if p < num_problems]
        elif max_problems_per_domain is not None:
            problem_indices = list(range(min(num_problems, max_problems_per_domain)))
        else:
            problem_indices = list(range(num_problems))

        domain_results = []
        for idx in problem_indices:
            if _benchmark_state["interrupted"]:
                break

            config = generate_config(domain, idx, output_dir, base_config)
            result = run_experiment(config, dry_run)
            results["total_experiments"] += 1

            if result is None:  # dry run
                continue

            # Add metadata for summary
            result["domain"] = domain
            result["problem_index"] = idx

            if result.get("success"):
                results["successful"] += 1
            else:
                results["failed"] += 1

            domain_results.append({"problem_index": idx, "result": result})

            # Track for interrupt handling
            _benchmark_state["results"].append(result)

            # Save incremental results
            save_incremental_results(output_dir, _benchmark_state["results"])

        results["domains"][domain] = domain_results

    return results


def main():
    # Set up signal handler for graceful interrupt
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Run LLM-SRBench benchmark across domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run first 3 problems in physics domain
    python -m problems.symbolic_regression.scripts.run_benchmark --domain phys --max-problems 3

    # Run all domains with max 5 problems each
    python -m problems.symbolic_regression.scripts.run_benchmark --domain all --max-problems 5

    # Run specific problems in chemistry
    python -m problems.symbolic_regression.scripts.run_benchmark --domain chem --problems 0,5,10

    # Use custom base config
    python -m problems.symbolic_regression.scripts.run_benchmark --domain phys --base-config my_config.yaml
""",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain(s) to run: 'all', single domain, or comma-separated list (bio,chem,matsci,phys,transform)",
    )
    parser.add_argument(
        "--problems",
        type=str,
        help="Specific problem indices to run (comma-separated, e.g., '0,1,2')",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        help="Maximum number of problems to run per domain",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        help="Base config file to use for LLM settings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs without running experiments",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domains and exit",
    )

    args = parser.parse_args()

    if args.list_domains:
        print("Available domains:")
        for domain, info in DOMAINS.items():
            domain_info = get_domain_info(domain)
            status = f"{domain_info['num_problems']} problems" if domain_info["num_problems"] > 0 else "not downloaded"
            print(f"  {domain}: {info['name']} ({status})")
        return

    # Parse domains
    if args.domain.lower() == "all":
        domains = list(DOMAINS.keys())
    else:
        domains = [d.strip() for d in args.domain.split(",")]
        invalid = [d for d in domains if d not in DOMAINS]
        if invalid:
            print(f"Error: Invalid domain(s): {invalid}")
            print(f"Valid domains: {list(DOMAINS.keys())}")
            sys.exit(1)

    # Parse problems
    problems = None
    if args.problems:
        problems = [int(p.strip()) for p in args.problems.split(",")]

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Base config
    base_config_path = Path(args.base_config) if args.base_config else None

    print("=" * 60)
    print("LLM-SRBench Benchmark Runner")
    print("=" * 60)
    print(f"Domains: {domains}")
    if problems:
        print(f"Problems: {problems}")
    if args.max_problems:
        print(f"Max problems per domain: {args.max_problems}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    results = run_benchmark(
        domains=domains,
        problems=problems,
        output_dir=output_dir,
        base_config_path=base_config_path,
        dry_run=args.dry_run,
        max_problems_per_domain=args.max_problems,
    )

    # Print summary table
    if not args.dry_run:
        print_summary_table(_benchmark_state["results"])

    # Save final results
    final_output_dir = output_dir or Path(__file__).parent.parent / "experiments" / "benchmark"
    if not args.dry_run:
        results_path = final_output_dir / "benchmark_results.json"
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
