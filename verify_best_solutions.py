"""
Verify MangoEvolve best circle packing solutions.

Runs each best trial's code, validates the packing with ZERO tolerance,
and reports constraint margins (how far each solution is from violating constraints).
"""

import json
import subprocess
import sys
import tempfile
import pickle
import os
import numpy as np

EXPERIMENTS = [
    {
        "name": "Gemini 3 Mixed",
        "path": "saved_experiments/circle_packing_gemini_3_mixed_20260108_100912/generations/gen_18/trial_18_3.json",
        "reported_score": 2.635983084917486,
    },
    {
        "name": "Opus Thinking #1",
        "path": "saved_experiments/circle_packing_opus_thinking_mixed_20251231_135159/generations/gen_14/trial_14_15.json",
        "reported_score": 2.6359831208890547,
    },
    {
        "name": "Opus Thinking #2 (best)",
        "path": "saved_experiments/circle_packing_opus_thinking_mixed_20251231_163512/generations/gen_14/trial_14_1.json",
        "reported_score": 2.6359850561146603,
    },
    {
        "name": "Opus Thinking #3",
        "path": "saved_experiments/circle_packing_opus_thinking_mixed_20251231_193658/generations/gen_16/trial_16_7.json",
        "reported_score": 2.635983089920464,
    },
    {
        "name": "OE Config #1",
        "path": "saved_experiments/openevolve_config_gemini_flash_20260111_191309/generations/gen_12/trial_12_0.json",
        "reported_score": 2.6359830853750683,
    },
    {
        "name": "OE Config #2",
        "path": "saved_experiments/openevolve_config_gemini_flash_20260114_120026/generations/gen_14/trial_14_4.json",
        "reported_score": 2.6359830849177386,
    },
]

N_CIRCLES = 26


def validate_strict(centers, radii):
    """Validate packing with ZERO tolerance. Returns detailed constraint analysis."""
    n = len(radii)
    results = {
        "valid_strict": True,
        "valid_with_1e6": True,
        "errors_strict": [],
        "min_wall_gap": float("inf"),
        "min_pair_gap": float("inf"),
        "worst_wall_circle": None,
        "worst_pair": None,
        "wall_violations_strict": 0,
        "pair_violations_strict": 0,
    }

    # Check boundary constraints (circle must be inside [0,1]x[0,1])
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        gaps = [x - r, y - r, 1.0 - (x + r), 1.0 - (y + r)]
        min_gap = min(gaps)
        if min_gap < results["min_wall_gap"]:
            results["min_wall_gap"] = min_gap
            results["worst_wall_circle"] = i

        if min_gap < 0:
            results["wall_violations_strict"] += 1
            results["errors_strict"].append(
                f"Circle {i}: wall gap = {min_gap:.2e} (x={x:.10f}, y={y:.10f}, r={r:.10f})"
            )
            if min_gap < -1e-6:
                results["valid_with_1e6"] = False
            results["valid_strict"] = False

    # Check pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            required = radii[i] + radii[j]
            gap = dist - required  # positive = no overlap
            if gap < results["min_pair_gap"]:
                results["min_pair_gap"] = gap
                results["worst_pair"] = (i, j)

            if gap < 0:
                results["pair_violations_strict"] += 1
                results["errors_strict"].append(
                    f"Circles {i},{j}: overlap gap = {gap:.2e} (dist={dist:.10f}, required={required:.10f})"
                )
                if gap < -1e-6:
                    results["valid_with_1e6"] = False
                results["valid_strict"] = False

    return results


def compute_shrink_factor(centers, radii):
    """
    Compute how much we could uniformly shrink all radii and still have a
    strictly valid packing (all gaps >= 0).

    Returns the factor alpha such that radii * alpha gives a strictly valid packing.
    alpha < 1 means we need to shrink, alpha > 1 means there's room to grow.
    """
    n = len(radii)
    # For wall constraints: x_i - alpha*r_i >= 0 => alpha <= x_i/r_i (and similar)
    min_alpha_wall = float("inf")
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r > 0:
            alphas = [x / r, y / r, (1.0 - x) / r, (1.0 - y) / r]
            min_alpha_wall = min(min_alpha_wall, min(alphas))

    # For pairwise constraints: dist(i,j) >= alpha*(r_i + r_j)
    # => alpha <= dist(i,j) / (r_i + r_j)
    min_alpha_pair = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            sum_r = radii[i] + radii[j]
            if sum_r > 0:
                min_alpha_pair = min(min_alpha_pair, dist / sum_r)

    return min(min_alpha_wall, min_alpha_pair)


def run_trial_code(code, timeout=300):
    """Run trial code in a subprocess, return (centers, radii, sum_radii, error)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        code_path = f.name

    results_path = f"{code_path}.results"
    runner = f'''
import sys
import numpy as np
import pickle

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
        raise ValueError("No run_packing() or construct_packing() found")
    results = {{"centers": np.array(centers), "radii": np.array(radii), "sum_radii": float(sum_radii), "error": None}}
except Exception as e:
    results = {{"centers": None, "radii": None, "sum_radii": None, "error": f"{{type(e).__name__}}: {{e}}"}}

with open("{results_path}", "wb") as f:
    pickle.dump(results, f)
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(runner)
        runner_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return None, None, None, proc.stderr.decode()[:500]

        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                res = pickle.load(f)
            return res["centers"], res["radii"], res["sum_radii"], res["error"]
        return None, None, None, "No results file"
    except subprocess.TimeoutExpired:
        return None, None, None, f"Timeout after {timeout}s"
    finally:
        for p in [code_path, runner_path, results_path]:
            if os.path.exists(p):
                os.unlink(p)


def main():
    print("=" * 80)
    print("MangoEvolve Circle Packing Solution Verification")
    print("=" * 80)
    print(f"\nValidation with ZERO tolerance (evaluator uses 1e-6)")
    print(f"n_circles = {N_CIRCLES}\n")

    for exp in EXPERIMENTS:
        print("-" * 80)
        print(f"Experiment: {exp['name']}")
        print(f"File: {exp['path']}")

        # Load code from JSON
        with open(exp["path"]) as f:
            trial_data = json.load(f)
        code = trial_data["code"]

        # Run the code
        print(f"Running code (timeout=300s)...", end=" ", flush=True)
        centers, radii, sum_radii, error = run_trial_code(code)

        if error:
            print(f"ERROR: {error}")
            continue

        print(f"done.")

        # Basic info
        actual_sum = float(np.sum(radii))
        print(f"  Reported score:  {exp['reported_score']:.16f}")
        print(f"  Code sum_radii:  {sum_radii:.16f}")
        print(f"  np.sum(radii):   {actual_sum:.16f}")
        print(f"  Score match:     {abs(actual_sum - exp['reported_score']) < 1e-10}")

        # Strict validation
        v = validate_strict(centers, radii)
        print(f"\n  Strict validation (tol=0):")
        print(f"    Valid:                {v['valid_strict']}")
        print(f"    Wall violations:      {v['wall_violations_strict']}")
        print(f"    Pair violations:      {v['pair_violations_strict']}")
        print(f"    Min wall gap:         {v['min_wall_gap']:.2e}  (circle {v['worst_wall_circle']})")
        print(f"    Min pair gap:         {v['min_pair_gap']:.2e}  (circles {v['worst_pair']})")
        if v["errors_strict"]:
            print(f"    Violations:")
            for e in v["errors_strict"][:5]:
                print(f"      {e}")

        # Valid with 1e-6 tolerance (matches evaluator behavior)
        print(f"\n  Evaluator validation (tol=1e-6):")
        print(f"    Valid:                {v['valid_with_1e6']}")

        # Shrink factor analysis
        alpha = compute_shrink_factor(centers, radii)
        print(f"\n  Shrink factor analysis:")
        print(f"    alpha (uniform scale): {alpha:.15f}")
        if alpha >= 1.0:
            print(f"    Interpretation:       Strictly valid! Room to GROW by {(alpha-1)*100:.6f}%")
            shrunk_score = actual_sum * alpha
            print(f"    Score if grown:       {shrunk_score:.16f}")
        else:
            print(f"    Interpretation:       Need to SHRINK by {(1-alpha)*100:.6f}%")
            shrunk_score = actual_sum * alpha
            print(f"    Score if shrunk:      {shrunk_score:.16f}")
            print(f"    Score loss:           {actual_sum - shrunk_score:.2e}")

        print()

    print("=" * 80)
    print("Summary: Shrunk scores (trivially valid without any tolerance)")
    print("=" * 80)


if __name__ == "__main__":
    main()
