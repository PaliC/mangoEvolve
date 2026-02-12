#!/usr/bin/env python3
"""
Parse root_llm_log.jsonl files to extract:
1. All query_llm calls (prompt, model, generation, response summary)
2. Scratchpad content per generation
3. REPL-defined functions
4. Convergence data (best score per generation)
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ABLATIONS_DIR = REPO_ROOT / "saved_experiments" / "ablations"
OPENEVOLVE_DIR = (
    REPO_ROOT
    / "saved_experiments"
    / "openevolve_config_gemini_flash_20260114_120026"
)

# Experiment directories with short names
EXPERIMENTS = {
    "baseline": ABLATIONS_DIR / "ablation_baseline_20260210_225140_20260210_225140",
    "no_query_llm": ABLATIONS_DIR / "ablation_no_query_llm_20260210_225140_20260210_225140",
    "no_scratchpad": ABLATIONS_DIR / "ablation_no_scratchpad_20260210_225140_20260210_225140",
    "no_trial_reasoning": ABLATIONS_DIR / "ablation_no_trial_reasoning_20260210_225140_20260210_225140",
    "all_disabled": ABLATIONS_DIR / "ablation_all_disabled_20260210_225140_20260210_225140",
    "shinka_evolve": ABLATIONS_DIR / "shinka_evolve_circle_packing_20260210_225140_20260210_225140",
    "openevolve": OPENEVOLVE_DIR,
}

# Which experiments have which features enabled
QUERY_LLM_ENABLED = {
    "baseline": True,
    "no_query_llm": False,
    "no_scratchpad": True,
    "no_trial_reasoning": True,
    "all_disabled": False,
    "shinka_evolve": True,
    "openevolve": True,
}

SCRATCHPAD_ENABLED = {
    "baseline": True,
    "no_query_llm": True,
    "no_scratchpad": False,
    "no_trial_reasoning": True,
    "all_disabled": False,
    "shinka_evolve": True,
    "openevolve": True,
}


def categorize_query_llm_call(prompt_text):
    """Categorize a query_llm call based on its prompt content."""
    prompt_lower = prompt_text.lower()

    if any(w in prompt_lower for w in ["analyze", "analysis", "examine", "inspect", "look at"]):
        if any(w in prompt_lower for w in ["trial", "solution", "code", "approach"]):
            return "trial_analysis"
        if any(w in prompt_lower for w in ["error", "fail", "bug"]):
            return "error_diagnosis"
        return "general_analysis"

    if any(w in prompt_lower for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"

    if any(w in prompt_lower for w in ["strategy", "plan", "next", "direction", "suggest", "recommend"]):
        return "strategy_planning"

    if any(w in prompt_lower for w in ["review", "improve", "refine", "optimize"]):
        return "code_review"

    if any(w in prompt_lower for w in ["summarize", "summary", "recap"]):
        return "summarization"

    if any(w in prompt_lower for w in ["error", "fail", "bug", "fix", "debug"]):
        return "error_diagnosis"

    if any(w in prompt_lower for w in ["explain", "why", "how", "what"]):
        return "explanation"

    return "other"


def parse_log_for_query_llm(log_path):
    """Parse JSONL log and extract all query_llm calls."""
    calls = []
    if not log_path.exists():
        return calls

    current_generation = 0

    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Track generation from messages
            content = ""
            if isinstance(entry, dict):
                content = entry.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(c.get("text", "") if isinstance(c, dict) else c)
                        for c in content
                    )
                elif not isinstance(content, str):
                    content = str(content)

                # Detect generation markers
                gen_match = re.search(r"Generation (\d+)", content)
                if gen_match:
                    current_generation = int(gen_match.group(1))

            # Find query_llm calls in content
            if "query_llm" in content:
                # Extract the prompt from the query_llm call
                # Pattern: query_llm([{"prompt": "...", ...}])
                query_prompts = re.findall(
                    r'query_llm\s*\(\s*\[([^\]]*)\]',
                    content,
                    re.DOTALL,
                )

                # Also match multiline patterns
                if not query_prompts:
                    query_prompts = re.findall(
                        r'query_llm\s*\(',
                        content,
                    )

                for qp in query_prompts:
                    # Try to extract prompt text
                    prompt_matches = re.findall(
                        r'"prompt"\s*:\s*["\'](.+?)["\']',
                        qp if qp else content,
                        re.DOTALL,
                    )
                    # Also match f-string patterns
                    if not prompt_matches:
                        prompt_matches = re.findall(
                            r'"prompt"\s*:\s*f["\'](.+?)["\']',
                            qp if qp else content,
                            re.DOTALL,
                        )

                    prompt_text = prompt_matches[0][:200] if prompt_matches else "(prompt not extractable)"

                    # Try to extract model
                    model_matches = re.findall(
                        r'"model"\s*:\s*"(.+?)"',
                        qp if qp else content,
                    )
                    model = model_matches[0] if model_matches else "default"

                    category = categorize_query_llm_call(prompt_text)

                    calls.append({
                        "generation": current_generation,
                        "prompt_excerpt": prompt_text,
                        "model": model,
                        "category": category,
                        "line_num": line_num,
                    })

    return calls


def parse_log_for_repl_functions(log_path):
    """Parse JSONL log and extract function definitions in REPL code blocks."""
    functions = []
    if not log_path.exists():
        return functions

    current_generation = 0

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = ""
            if isinstance(entry, dict):
                content = entry.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(c.get("text", "") if isinstance(c, dict) else c)
                        for c in content
                    )
                elif not isinstance(content, str):
                    content = str(content)

                gen_match = re.search(r"Generation (\d+)", content)
                if gen_match:
                    current_generation = int(gen_match.group(1))

            # Find function definitions in code blocks
            # Look for def statements that are NOT inside child prompts
            code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
            for block in code_blocks:
                # Skip if this looks like a child prompt (contains spawn_children)
                if "spawn_children" in block:
                    # Still check for helper functions defined before spawn
                    pass

                func_defs = re.findall(r'def\s+(\w+)\s*\(([^)]*)\)', block)
                for func_name, params in func_defs:
                    # Skip common child-code functions
                    if func_name in ("run_packing", "construct_packing", "objective",
                                     "constraints_func", "jacobian_func", "polish"):
                        continue
                    # This is likely a REPL helper function
                    functions.append({
                        "generation": current_generation,
                        "name": func_name,
                        "params": params.strip(),
                    })

    return functions


def read_scratchpads(exp_dir):
    """Read all scratchpad files for an experiment."""
    scratchpads = {}
    gen_dir = exp_dir / "generations"
    if not gen_dir.exists():
        return scratchpads

    for gen_folder in sorted(gen_dir.iterdir()):
        if gen_folder.is_dir() and gen_folder.name.startswith("gen_"):
            gen_num = int(gen_folder.name.split("_")[1])
            sp_file = gen_folder / "scratchpad.txt"
            if sp_file.exists():
                content = sp_file.read_text()
                # Check if empty
                is_empty = "(Empty)" in content or len(content.strip().split("\n")) <= 4
                word_count = len(content.split()) if not is_empty else 0

                # Extract themes
                themes = []
                if "strategy" in content.lower():
                    themes.append("strategy")
                if "analysis" in content.lower() or "insight" in content.lower():
                    themes.append("analysis")
                if "```" in content:
                    themes.append("code_snippets")
                if "error" in content.lower() or "fail" in content.lower():
                    themes.append("error_tracking")
                if "generation" in content.lower() and "summary" in content.lower():
                    themes.append("generation_summary")
                if "lp" in content.lower() or "linear programming" in content.lower():
                    themes.append("LP_optimization")
                if "slsqp" in content.lower():
                    themes.append("SLSQP_refinement")
                if "shaking" in content.lower() or "perturbation" in content.lower():
                    themes.append("shaking_perturbation")
                if "basin" in content.lower():
                    themes.append("basin_hopping")

                scratchpads[gen_num] = {
                    "word_count": word_count,
                    "is_empty": is_empty,
                    "themes": themes,
                    "content_excerpt": content[:500] if not is_empty else "(Empty)",
                }

    return scratchpads


def read_convergence_data(exp_dir):
    """Read summary.json from each generation to track convergence."""
    convergence = []
    gen_dir = exp_dir / "generations"
    if not gen_dir.exists():
        return convergence

    cumulative_best = 0.0

    for gen_folder in sorted(gen_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]) if p.name.startswith("gen_") else -1):
        if gen_folder.is_dir() and gen_folder.name.startswith("gen_"):
            gen_num = int(gen_folder.name.split("_")[1])
            summary_file = gen_folder / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)

                gen_best = data.get("best_score", data.get("best_trial", {}).get("score", 0))
                if isinstance(gen_best, dict):
                    gen_best = gen_best.get("score", 0)

                # Handle different summary formats
                if gen_best == 0:
                    # Try to extract from trials
                    trials = data.get("trials", [])
                    if trials:
                        scores = [t.get("score", 0) for t in trials if t.get("success", False)]
                        gen_best = max(scores) if scores else 0

                cumulative_best = max(cumulative_best, gen_best)

                num_trials = data.get("num_trials", data.get("total_trials", 0))
                num_successful = data.get("num_successful", data.get("successful_trials", 0))

                convergence.append({
                    "generation": gen_num,
                    "gen_best_score": gen_best,
                    "cumulative_best": cumulative_best,
                    "num_trials": num_trials,
                    "num_successful": num_successful,
                })

    return convergence


def main():
    all_results = {}

    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        result = {"name": name}

        # 1. Query LLM calls
        if QUERY_LLM_ENABLED.get(name, False):
            log_path = exp_dir / "root_llm_log.jsonl"
            calls = parse_log_for_query_llm(log_path)
            result["query_llm_calls"] = calls
            result["query_llm_total"] = len(calls)

            # Category breakdown
            categories = defaultdict(int)
            for c in calls:
                categories[c["category"]] += 1
            result["query_llm_categories"] = dict(categories)

            print(f"\n  query_llm calls: {len(calls)}")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")
        else:
            result["query_llm_calls"] = []
            result["query_llm_total"] = 0
            result["query_llm_categories"] = {}
            print(f"\n  query_llm: DISABLED")

        # 2. Scratchpad content
        if SCRATCHPAD_ENABLED.get(name, False):
            scratchpads = read_scratchpads(exp_dir)
            result["scratchpads"] = scratchpads

            total_words = sum(s["word_count"] for s in scratchpads.values())
            non_empty = sum(1 for s in scratchpads.values() if not s["is_empty"])
            all_themes = set()
            for s in scratchpads.values():
                all_themes.update(s["themes"])

            print(f"\n  Scratchpad: {non_empty}/{len(scratchpads)} generations with content")
            print(f"    Total words: {total_words}")
            print(f"    Themes: {', '.join(sorted(all_themes)) if all_themes else 'none'}")
        else:
            result["scratchpads"] = {}
            print(f"\n  Scratchpad: DISABLED")

        # 3. REPL functions
        log_path = exp_dir / "root_llm_log.jsonl"
        repl_funcs = parse_log_for_repl_functions(log_path)
        result["repl_functions"] = repl_funcs

        # Deduplicate by name
        unique_funcs = {}
        for f in repl_funcs:
            if f["name"] not in unique_funcs:
                unique_funcs[f["name"]] = f
        result["unique_repl_functions"] = list(unique_funcs.values())

        print(f"\n  REPL functions: {len(repl_funcs)} total, {len(unique_funcs)} unique")
        for f in unique_funcs.values():
            print(f"    - {f['name']}({f['params']}) [gen {f['generation']}]")

        # 4. Convergence data
        convergence = read_convergence_data(exp_dir)
        result["convergence"] = convergence

        print(f"\n  Convergence ({len(convergence)} generations):")
        for c in convergence:
            marker = " <-- BEST" if c["gen_best_score"] == c["cumulative_best"] and c["gen_best_score"] > 2.6359 else ""
            print(
                f"    Gen {c['generation']:>2}: best={c['gen_best_score']:.10f}  "
                f"cumul={c['cumulative_best']:.10f}  "
                f"({c['num_successful']}/{c['num_trials']} success){marker}"
            )

        all_results[name] = result

    # Print comprehensive query_llm analysis
    print("\n\n")
    print("=" * 80)
    print("COMPREHENSIVE QUERY_LLM ANALYSIS")
    print("=" * 80)

    for name in ["baseline", "no_scratchpad", "no_trial_reasoning", "shinka_evolve", "openevolve"]:
        if name not in all_results:
            continue
        r = all_results[name]
        if not r["query_llm_calls"]:
            continue

        print(f"\n--- {name} ({r['query_llm_total']} calls) ---")
        print(f"  Category breakdown:")
        for cat, count in sorted(r["query_llm_categories"].items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
        print(f"\n  Calls by generation:")
        gen_counts = defaultdict(int)
        for c in r["query_llm_calls"]:
            gen_counts[c["generation"]] += 1
        for gen in sorted(gen_counts.keys()):
            print(f"    Gen {gen}: {gen_counts[gen]} calls")
        print(f"\n  Sample prompts:")
        seen_categories = set()
        for c in r["query_llm_calls"]:
            if c["category"] not in seen_categories:
                seen_categories.add(c["category"])
                print(f"    [{c['category']}] Gen {c['generation']}: {c['prompt_excerpt'][:120]}...")

    # Print scratchpad comparison
    print("\n\n")
    print("=" * 80)
    print("SCRATCHPAD COMPARISON")
    print("=" * 80)

    for name in ["baseline", "no_query_llm", "no_trial_reasoning", "shinka_evolve", "openevolve"]:
        if name not in all_results:
            continue
        r = all_results[name]
        if not r["scratchpads"]:
            continue

        total_words = sum(s["word_count"] for s in r["scratchpads"].values())
        non_empty = sum(1 for s in r["scratchpads"].values() if not s["is_empty"])
        all_themes = set()
        for s in r["scratchpads"].values():
            all_themes.update(s["themes"])

        print(f"\n--- {name} ---")
        print(f"  Non-empty generations: {non_empty}/{len(r['scratchpads'])}")
        print(f"  Total word count: {total_words}")
        print(f"  Themes present: {', '.join(sorted(all_themes))}")
        for gen, sp in sorted(r["scratchpads"].items()):
            if not sp["is_empty"]:
                print(f"  Gen {gen}: {sp['word_count']} words, themes: {sp['themes']}")

    # Print convergence comparison
    print("\n\n")
    print("=" * 80)
    print("CONVERGENCE COMPARISON")
    print("=" * 80)

    TARGET = 2.6359
    print(f"\nGeneration first reaching cumulative best >= {TARGET}:")
    for name, r in sorted(all_results.items()):
        conv = r["convergence"]
        reached = None
        for c in conv:
            if c["cumulative_best"] >= TARGET:
                reached = c["generation"]
                break
        if reached is not None:
            total_evals = sum(c["num_trials"] for c in conv[:reached + 1])
            print(f"  {name:<25} Gen {reached:>2} ({total_evals} total evaluations)")
        else:
            print(f"  {name:<25} Never reached")

    # Save results
    output_path = Path(__file__).parent / "log_analysis_results.json"

    # Make serializable
    serializable = {}
    for name, r in all_results.items():
        sr = dict(r)
        # Scratchpads are already serializable
        serializable[name] = sr

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
