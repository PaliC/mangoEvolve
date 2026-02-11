#!/usr/bin/env python3
"""
Compute both relaxed (tolerance=1e-6) and strict (tolerance=0) scores
for the best trial from each MangoEvolve experiment.

For strict scores, we uniformly shrink all radii by the minimum factor needed
to eliminate any constraint violations (overlaps or boundary breaches).
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
ABLATIONS_DIR = REPO_ROOT / "saved_experiments" / "ablations"
OPENEVOLVE_DIR = (
    REPO_ROOT
    / "saved_experiments"
    / "openevolve_config_gemini_flash_20260114_120026"
)

N_CIRCLES = 26
TIMEOUT_SECONDS = 300  # Match experiment timeout


def find_experiments():
    """Find all experiments and their best trial info."""
    experiments = {}

    # Ablation experiments
    for entry in sorted(ABLATIONS_DIR.iterdir()):
        if entry.is_dir():
            exp_json = entry / "experiment.json"
            if exp_json.exists():
                with open(exp_json) as f:
                    data = json.load(f)
                best = data.get("best_trial", {})
                if best:
                    experiments[entry.name] = {
                        "dir": entry,
                        "best_trial_id": best["trial_id"],
                        "best_score": best["score"],
                        "best_generation": best["generation"],
                    }

    # OpenEvolve experiment
    oe_json = OPENEVOLVE_DIR / "experiment.json"
    if oe_json.exists():
        with open(oe_json) as f:
            data = json.load(f)
        best = data.get("best_trial", {})
        if best:
            experiments["openevolve_config_gemini_flash"] = {
                "dir": OPENEVOLVE_DIR,
                "best_trial_id": best["trial_id"],
                "best_score": best["score"],
                "best_generation": best["generation"],
            }

    return experiments


def load_trial_code(exp_dir, trial_id, generation):
    """Load the code from a trial JSON file."""
    trial_path = exp_dir / "generations" / f"gen_{generation}" / f"{trial_id}.json"
    if not trial_path.exists():
        # Search all generations
        for gen_dir in sorted((exp_dir / "generations").iterdir()):
            candidate = gen_dir / f"{trial_id}.json"
            if candidate.exists():
                trial_path = candidate
                break
    if not trial_path.exists():
        return None
    with open(trial_path) as f:
        data = json.load(f)
    return data.get("code")


def run_code_with_timeout(code, timeout_seconds=TIMEOUT_SECONDS):
    """Run circle packing code in a subprocess and return (centers, radii, sum_radii, error)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as code_file:
        code_file.write(code)
        code_path = code_file.name

    results_path = f"{code_path}.results"

    runner_script = f'''
import sys
import numpy as np
import pickle
import traceback

try:
    with open("{code_path}", "r") as f:
        code = f.read()

    namespace = {{"np": np, "numpy": np}}
    exec(code, namespace)

    if "run_packing" in namespace:
        centers, radii, sum_radii = namespace["run_packing"]()
    elif "construct_packing" in namespace:
        centers, radii, sum_radii = namespace["construct_packing"]()
    else:
        raise ValueError("Code must define run_packing() or construct_packing()")

    centers = np.array(centers)
    radii = np.array(radii)
    sum_radii = float(sum_radii)

    results = {{
        "centers": centers,
        "radii": radii,
        "sum_radii": sum_radii,
        "error": None
    }}
except Exception as e:
    results = {{
        "centers": None,
        "radii": None,
        "sum_radii": None,
        "error": f"{{type(e).__name__}}: {{str(e)}}"
    }}

with open("{results_path}", "wb") as f:
    pickle.dump(results, f)
'''

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as runner_file:
        runner_file.write(runner_script)
        runner_path = runner_file.name

    try:
        process = subprocess.Popen(
            [sys.executable, runner_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            if process.returncode != 0:
                error_msg = (
                    stderr.decode()
                    if stderr
                    else f"Process exited with code {process.returncode}"
                )
                return None, None, None, error_msg
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if results["error"]:
                    return None, None, None, results["error"]
                return (
                    results["centers"],
                    results["radii"],
                    results["sum_radii"],
                    None,
                )
            else:
                return None, None, None, "Results file not created"
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return None, None, None, f"Timeout after {timeout_seconds}s"
    finally:
        for path in [code_path, runner_path, results_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


def validate_relaxed(centers, radii, tolerance=1e-6):
    """Validate packing with tolerance (relaxed). Returns (valid, error)."""
    n = len(radii)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tolerance or x + r > 1 + tolerance:
            return False, f"Circle {i} outside x-bounds"
        if y - r < -tolerance or y + r > 1 + tolerance:
            return False, f"Circle {i} outside y-bounds"
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = radii[i] + radii[j] - tolerance
            if dist < min_dist:
                return False, f"Circles {i},{j} overlap"
    return True, None


def compute_strict_score(centers, radii):
    """
    Compute the strict score by finding the uniform scaling factor alpha
    such that alpha * radii satisfies all constraints with tolerance=0.

    Returns (alpha, strict_score, violation_type, violation_details).
    """
    n = len(radii)
    alpha = 1.0  # Start with no scaling
    worst_violation = None
    worst_detail = None

    # Check boundary constraints: for each circle, r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        max_r_boundary = min(x, 1 - x, y, 1 - y)
        if r > 0 and max_r_boundary < r:
            ratio = max_r_boundary / r
            if ratio < alpha:
                alpha = ratio
                worst_violation = "boundary"
                worst_detail = f"circle {i}: r={r:.10f}, max_boundary={max_r_boundary:.10f}"

    # Check overlap constraints: for each pair, dist(i,j) >= r_i + r_j
    # With uniform scaling alpha: alpha*(r_i + r_j) <= dist(i,j)
    # So alpha <= dist(i,j) / (r_i + r_j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            sum_r = radii[i] + radii[j]
            if sum_r > 0 and dist < sum_r:
                ratio = dist / sum_r
                if ratio < alpha:
                    alpha = ratio
                    worst_violation = "overlap"
                    worst_detail = f"circles {i},{j}: dist={dist:.10f}, sum_r={sum_r:.10f}"

    strict_score = alpha * np.sum(radii)
    return alpha, strict_score, worst_violation, worst_detail


def main():
    print("=" * 80)
    print("MangoEvolve: Relaxed vs Strict Circle Packing Scores")
    print("=" * 80)
    print()

    experiments = find_experiments()

    results = []

    for name, info in sorted(experiments.items()):
        short_name = name.replace("_20260210_225140_20260210_225140", "").replace(
            "_20260114_120026", ""
        )
        print(f"\n--- {short_name} ---")
        print(f"  Best trial: {info['best_trial_id']} (gen {info['best_generation']})")
        print(f"  Recorded score: {info['best_score']:.16f}")

        code = load_trial_code(
            info["dir"], info["best_trial_id"], info["best_generation"]
        )
        if code is None:
            print("  ERROR: Could not load trial code")
            results.append(
                {
                    "experiment": short_name,
                    "recorded_score": info["best_score"],
                    "relaxed_score": None,
                    "strict_score": None,
                    "alpha": None,
                    "error": "code not found",
                }
            )
            continue

        print(f"  Executing code (timeout={TIMEOUT_SECONDS}s)...")
        start = time.time()
        centers, radii, sum_radii, error = run_code_with_timeout(code)
        elapsed = time.time() - start

        if error:
            print(f"  ERROR: {error}")
            results.append(
                {
                    "experiment": short_name,
                    "recorded_score": info["best_score"],
                    "relaxed_score": None,
                    "strict_score": None,
                    "alpha": None,
                    "error": error,
                }
            )
            continue

        print(f"  Execution time: {elapsed:.1f}s")
        relaxed_score = float(np.sum(radii))
        print(f"  Relaxed score (re-executed): {relaxed_score:.16f}")

        # Validate relaxed
        valid, val_err = validate_relaxed(centers, radii)
        if not valid:
            print(f"  WARNING: Relaxed validation failed: {val_err}")

        # Compute strict score
        alpha, strict_score, violation_type, violation_detail = compute_strict_score(
            centers, radii
        )
        print(f"  Scaling factor (alpha): {alpha:.16f}")
        print(f"  Strict score: {strict_score:.16f}")
        print(f"  Score difference (relaxed - strict): {relaxed_score - strict_score:.2e}")
        if violation_type:
            print(f"  Worst violation: {violation_type} - {violation_detail}")
        else:
            print("  No strict violations (alpha=1.0, packing already strict-valid)")

        results.append(
            {
                "experiment": short_name,
                "recorded_score": info["best_score"],
                "relaxed_score": relaxed_score,
                "strict_score": strict_score,
                "alpha": alpha,
                "worst_violation": violation_type,
                "error": None,
            }
        )

    # Summary table
    print("\n")
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Experiment':<40} {'Recorded':>16} {'Relaxed':>16} {'Strict':>16} {'Delta':>12} {'Alpha':>12}"
    )
    print("-" * 100)

    for r in results:
        if r["error"]:
            print(f"{r['experiment']:<40} {r['recorded_score']:>16.10f} {'ERROR':>16} {'ERROR':>16}")
        else:
            delta = r["relaxed_score"] - r["strict_score"]
            print(
                f"{r['experiment']:<40} "
                f"{r['recorded_score']:>16.10f} "
                f"{r['relaxed_score']:>16.10f} "
                f"{r['strict_score']:>16.10f} "
                f"{delta:>12.2e} "
                f"{r['alpha']:>12.10f}"
            )

    # State of the art comparison
    print("\n")
    print("=" * 100)
    print("STATE-OF-THE-ART COMPARISON")
    print("=" * 100)
    print(f"{'System':<40} {'Relaxed':>16} {'Strict':>16}")
    print("-" * 72)
    print(f"{'Previous best known':.<40} {'2.634':>16} {'2.634':>16}")
    print(f"{'AlphaEvolve':.<40} {'2.6358627500':>16} {'N/A':>16}")
    print(f"{'ShinkaEvolve':.<40} {'2.6359831000':>16} {'2.6359777093':>16}")
    print(f"{'OpenEvolve (community)':.<40} {'2.6359773948':>16} {'N/A':>16}")
    print("-" * 72)
    for r in results:
        if not r["error"]:
            print(
                f"{'MangoEvolve ' + r['experiment']:<40} "
                f"{r['relaxed_score']:>16.10f} "
                f"{r['strict_score']:>16.10f}"
            )

    # Save results as JSON
    output_path = Path(__file__).parent / "strict_scores_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
