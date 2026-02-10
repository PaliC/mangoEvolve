#!/usr/bin/env python3
"""
Ablation Study Runner for MangoEvolve.

This script validates and runs ablation experiments to test the impact of
different features (scratchpad, trial reasoning, query_llm) on evolution.

Usage:
    # Validate all configs (no API calls)
    python scripts/run_ablations.py --validate

    # Run quick validation tests (minimal API calls, 1 gen, 2 children)
    python scripts/run_ablations.py --quick-test

    # Run a specific ablation at full scale
    python scripts/run_ablations.py --run configs/ablations/no_scratchpad.yaml

    # Run all ablations at full scale (sequential)
    python scripts/run_ablations.py --all

    # Run all ablations at full scale with a specific output directory
    python scripts/run_ablations.py --all --output-dir ./experiments/ablation_study_1
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, try loading manually
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mango_evolve.config import Config, config_from_dict, load_config
from mango_evolve.llm.prompts import (
    build_root_system_prompt_static,
    get_calibration_system_prompt_parts,
)


ABLATION_CONFIGS = [
    # "configs/ablations/baseline.yaml",
    # "configs/ablations/no_scratchpad.yaml",
    # "configs/ablations/no_trial_reasoning.yaml",
    # "configs/ablations/no_query_llm.yaml",
    "configs/ablations/all_ablations.yaml",
]


def validate_config(config_path: str) -> tuple[bool, Config | None, str]:
    """
    Validate a config file loads correctly and has expected ablation flags.

    Returns:
        Tuple of (success, config, message)
    """
    try:
        config = load_config(config_path)

        # Check ablation flags are present and valid booleans
        flags = {
            "hide_scratchpad": config.hide_scratchpad,
            "hide_trial_reasoning": config.hide_trial_reasoning,
            "disable_query_llm": config.disable_query_llm,
        }

        for name, value in flags.items():
            if not isinstance(value, bool):
                return False, None, f"Flag {name} is not a boolean: {value}"

        return True, config, f"OK - flags: {flags}"

    except Exception as e:
        return False, None, f"FAILED: {e}"


def validate_prompt_generation(config: Config) -> tuple[bool, str]:
    """
    Validate that prompts are generated correctly with ablation flags.

    Returns:
        Tuple of (success, message)
    """

    # Create a mock problem spec
    class MockSpec:
        name = "Test Problem"
        description = "A test problem"
        best_known_solution = 2.0
        objective = "maximize"
        metric_name = "score"
        entry_function = "solve"
        helper_functions = []
        return_description = "Return a value"
        allowed_modules = ["numpy"]
        constraints = []
        example_code = None
        reference_code = None
        reference_context = None
        secondary_metrics = []

    spec = MockSpec()

    try:
        # Test root system prompt
        prompt = build_root_system_prompt_static(
            spec, disable_query_llm=config.disable_query_llm
        )

        # Verify query_llm is properly included/excluded
        has_query_llm = "query_llm" in prompt
        if config.disable_query_llm and has_query_llm:
            return False, "query_llm should be removed from prompt but is present"
        if not config.disable_query_llm and not has_query_llm:
            return False, "query_llm should be in prompt but is missing"

        return True, "Prompt generation OK"

    except Exception as e:
        return False, f"Prompt generation failed: {e}"


def validate_evolution_api(config: Config) -> tuple[bool, str]:
    """
    Validate that EvolutionAPI respects ablation flags.

    Returns:
        Tuple of (success, message)
    """
    from unittest.mock import Mock

    from mango_evolve.config import ChildLLMConfig
    from mango_evolve.evolution_api import EvolutionAPI, ScratchpadProxy

    class MockEvaluator:
        def evaluate(self, code):
            return {"valid": True, "score": 1.0}

    class MockProblemSpec:
        name = "Test"

    class MockLogger:
        def save_scratchpad(self, **kwargs):
            pass

        base_dir = "/tmp/test"

    mock_cost_tracker = Mock()
    mock_cost_tracker.raise_if_over_budget = Mock()

    try:
        api = EvolutionAPI(
            evaluator=MockEvaluator(),
            problem_spec=MockProblemSpec(),
            child_llm_configs={
                "default": Mock(spec=ChildLLMConfig, calibration_calls=0)
            },
            cost_tracker=mock_cost_tracker,
            logger=MockLogger(),
            hide_scratchpad=config.hide_scratchpad,
            hide_trial_reasoning=config.hide_trial_reasoning,
            disable_query_llm=config.disable_query_llm,
        )

        issues = []

        # Test scratchpad behavior
        proxy = ScratchpadProxy(api)
        if config.hide_scratchpad:
            if "unavailable" not in proxy.content:
                issues.append("Scratchpad should return placeholder when hidden")
        else:
            if "unavailable" in proxy.content and proxy.content != "":
                issues.append("Scratchpad should not return placeholder when not hidden")

        # Test query_llm in namespace
        funcs = api.get_api_functions()
        if config.disable_query_llm:
            if "query_llm" in funcs:
                issues.append("query_llm should be excluded from namespace")
        else:
            if "query_llm" not in funcs:
                issues.append("query_llm should be in namespace")

        if issues:
            return False, "; ".join(issues)

        return True, "EvolutionAPI OK"

    except Exception as e:
        return False, f"EvolutionAPI test failed: {e}"


def run_quick_test(config_path: str) -> tuple[bool, str, dict | None]:
    """
    Run a quick integration test with minimal API calls.

    This runs evolution for just 1 generation with 2 children to verify
    the ablation flags are working end-to-end.

    Returns:
        Tuple of (success, message, result_dict)
    """
    import tempfile

    import yaml

    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY not set - skipping API test", None

    try:
        # Load and modify config for quick test
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Reduce to minimal settings for quick validation
        config_data["evolution"]["max_generations"] = 1
        config_data["evolution"]["max_children_per_generation"] = 2
        config_data["budget"]["max_total_cost"] = 0.50

        # Use temp output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data["experiment"]["output_dir"] = tmpdir
            config_data["experiment"]["name"] = f"quick_test_{Path(config_path).stem}"

            # Remove calibration notes file if present
            config_data.pop("calibration_notes_file", None)

            # Set calibration calls to 0 to skip calibration
            for child in config_data.get("child_llms", []):
                child["calibration_calls"] = 0

            config = config_from_dict(config_data)

            # Run orchestrator
            from mango_evolve.root_llm import RootLLMOrchestrator

            orchestrator = RootLLMOrchestrator(config)
            result = orchestrator.run()

            # Validate result
            if result.total_trials == 0:
                return False, "No trials were spawned", None

            # Check that ablation flags affected behavior
            flags_msg = (
                f"scratchpad={'hidden' if config.hide_scratchpad else 'enabled'}, "
                f"reasoning={'hidden' if config.hide_trial_reasoning else 'enabled'}, "
                f"query_llm={'disabled' if config.disable_query_llm else 'enabled'}"
            )

            result_dict = {
                "config": Path(config_path).stem,
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "best_score": result.best_score,
                "num_generations": result.num_generations,
                "reason": result.reason,
                "cost": result.cost_summary,
                "flags": {
                    "hide_scratchpad": config.hide_scratchpad,
                    "hide_trial_reasoning": config.hide_trial_reasoning,
                    "disable_query_llm": config.disable_query_llm,
                },
            }

            return (
                True,
                f"Quick test passed: {result.total_trials} trials, "
                f"best={result.best_score:.4f}, {flags_msg}",
                result_dict,
            )

    except Exception as e:
        import traceback

        return False, f"Quick test failed: {e}\n{traceback.format_exc()}", None


def run_full_ablation(
    config_path: str, output_dir: str | None = None
) -> tuple[bool, str, dict | None]:
    """
    Run a full ablation experiment using the config as-is.

    Args:
        config_path: Path to the ablation config file
        output_dir: Optional override for output directory

    Returns:
        Tuple of (success, message, result_dict)
    """
    import yaml

    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY not set - cannot run ablation", None

    try:
        # Load config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Override output directory if provided
        if output_dir:
            config_data["experiment"]["output_dir"] = output_dir

        # Add timestamp to experiment name for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = config_data["experiment"]["name"]
        config_data["experiment"]["name"] = f"{original_name}_{timestamp}"

        # Remove calibration notes file if present (start fresh)
        config_data.pop("calibration_notes_file", None)

        config = config_from_dict(config_data)

        print(f"\n    Experiment: {config_data['experiment']['name']}")
        print(f"    Output: {config_data['experiment']['output_dir']}")
        print(f"    Generations: {config.evolution.max_generations}")
        print(f"    Children/gen: {config.evolution.max_children_per_generation}")
        print(f"    Budget: ${config.budget.max_total_cost:.2f}")
        print()

        # Run orchestrator
        from mango_evolve.root_llm import RootLLMOrchestrator

        orchestrator = RootLLMOrchestrator(config)
        result = orchestrator.run()

        # Build result dict
        result_dict = {
            "config": Path(config_path).stem,
            "experiment_name": config_data["experiment"]["name"],
            "total_trials": result.total_trials,
            "successful_trials": result.successful_trials,
            "best_score": result.best_score,
            "num_generations": result.num_generations,
            "reason": result.reason,
            "cost": result.cost_summary,
            "flags": {
                "hide_scratchpad": config.hide_scratchpad,
                "hide_trial_reasoning": config.hide_trial_reasoning,
                "disable_query_llm": config.disable_query_llm,
            },
        }

        flags_msg = (
            f"scratchpad={'hidden' if config.hide_scratchpad else 'enabled'}, "
            f"reasoning={'hidden' if config.hide_trial_reasoning else 'enabled'}, "
            f"query_llm={'disabled' if config.disable_query_llm else 'enabled'}"
        )

        msg = (
            f"Completed: {result.num_generations} generations, "
            f"{result.total_trials} trials, "
            f"best={result.best_score:.6f}, "
            f"cost=${result.cost_summary.get('total_cost', 0):.2f}, "
            f"{flags_msg}"
        )

        return True, msg, result_dict

    except Exception as e:
        import traceback

        return False, f"Ablation failed: {e}\n{traceback.format_exc()}", None


def save_study_results(results: list[dict], output_dir: str) -> str:
    """Save ablation study results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"ablation_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "num_configs": len(results),
                "results": results,
            },
            f,
            indent=2,
        )

    return str(results_file)


def print_study_summary(results: list[dict]) -> None:
    """Print a summary table of ablation study results."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    # Header
    print(
        f"{'Config':<25} {'Scratchpad':<12} {'Reasoning':<12} {'QueryLLM':<12} "
        f"{'Best Score':<12} {'Cost':<10}"
    )
    print("-" * 80)

    for r in results:
        flags = r.get("flags", {})
        cost = r.get("cost", {}).get("total_cost", 0) if r.get("cost") else 0
        print(
            f"{r['config']:<25} "
            f"{'hidden' if flags.get('hide_scratchpad') else 'enabled':<12} "
            f"{'hidden' if flags.get('hide_trial_reasoning') else 'enabled':<12} "
            f"{'disabled' if flags.get('disable_query_llm') else 'enabled':<12} "
            f"{r.get('best_score', 0):<12.6f} "
            f"${cost:<9.2f}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for MangoEvolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --validate                    Validate configs only (no API calls)
  %(prog)s --quick-test                  Quick test all configs (1 gen, 2 children)
  %(prog)s --run configs/ablations/baseline.yaml   Run single config at full scale
  %(prog)s --all                         Run all configs at full scale
  %(prog)s --all --output-dir ./my_study Run all with custom output directory
        """,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all configs without running (no API calls)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick validation tests (1 generation, 2 children each)",
    )
    parser.add_argument(
        "--run",
        type=str,
        metavar="CONFIG",
        help="Run a specific config file at full scale",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all ablation experiments at full scale (sequential)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/ablations",
        help="Output directory for experiment results (default: ./experiments/ablations)",
    )

    args = parser.parse_args()

    if not any([args.validate, args.quick_test, args.run, args.all]):
        parser.print_help()
        return 1

    # Determine which configs to process
    if args.run:
        configs = [args.run]
    else:
        configs = ABLATION_CONFIGS

    print("=" * 60)
    print("MangoEvolve Ablation Study")
    print("=" * 60)

    if args.all:
        print(f"\nRunning FULL ablation study ({len(configs)} configs)")
        print(f"Output directory: {args.output_dir}")
    elif args.run:
        print(f"\nRunning FULL ablation: {args.run}")
        print(f"Output directory: {args.output_dir}")
    elif args.quick_test:
        print("\nRunning QUICK tests (1 generation, 2 children each)")
    else:
        print("\nValidating configs only (no API calls)")

    all_passed = True
    study_results = []

    for config_path in configs:
        print(f"\n{'=' * 60}")
        print(f"Config: {config_path}")
        print("=" * 60)

        # Step 1: Validate config loads
        print("\n[1] Validating config file...")
        success, config, msg = validate_config(config_path)
        print(f"    {msg}")
        if not success:
            all_passed = False
            continue

        # Step 2: Validate prompt generation
        print("\n[2] Validating prompt generation...")
        success, msg = validate_prompt_generation(config)
        print(f"    {msg}")
        if not success:
            all_passed = False

        # Step 3: Validate EvolutionAPI behavior
        print("\n[3] Validating EvolutionAPI behavior...")
        success, msg = validate_evolution_api(config)
        print(f"    {msg}")
        if not success:
            all_passed = False

        # Step 4: Run experiment (if requested)
        if args.quick_test:
            print("\n[4] Running quick integration test...")
            success, msg, result = run_quick_test(config_path)
            print(f"    {msg}")
            if result:
                study_results.append(result)
            if not success and "GEMINI_API_KEY not set" not in msg:
                all_passed = False

        elif args.run or args.all:
            print("\n[4] Running full ablation experiment...")
            success, msg, result = run_full_ablation(config_path, args.output_dir)
            print(f"    {msg}")
            if result:
                study_results.append(result)
            if not success:
                all_passed = False

    # Print summary if we ran experiments
    if study_results:
        print_study_summary(study_results)

        # Save results to file
        if args.all or args.run:
            results_file = save_study_results(study_results, args.output_dir)
            print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 60)
    if all_passed:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
