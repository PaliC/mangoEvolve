"""Minimizing Max-Min Distance Evaluator for MangoEvolve.

Adapted from OpenEvolve's AlphaEvolve math problems:
https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems/minimizing_max_min_dist
"""

from __future__ import annotations

import contextlib
import os
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist

from mango_evolve.problem import BaseProblemEvaluator, ProblemSpec

BENCHMARKS = {
    (2, 16): 1 / 12.889266112,
    (3, 14): 1 / 4.165849767,
}


def _run_code_with_timeout(
    code: str,
    entry_function: str,
    timeout_seconds: int,
    python_executable: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Execute candidate code in a separate process with a timeout."""
    python_cmd = python_executable if python_executable else sys.executable

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as code_file:
        code_file.write(code)
        code_path = code_file.name

    results_path = f"{code_path}.results"

    runner_script = f'''
import numpy as np
import pickle

try:
    with open("{code_path}", "r") as f:
        code = f.read()

    namespace = {{"np": np, "numpy": np}}
    exec(code, namespace)

    if "{entry_function}" not in namespace:
        raise ValueError("Code must define `{entry_function}()`")

    points = namespace["{entry_function}"]()
    points = np.array(points)

    results = {{"points": points, "error": None}}
except Exception as e:
    results = {{"points": None, "error": f"{{type(e).__name__}}: {{e}}"}}

with open("{results_path}", "wb") as f:
    pickle.dump(results, f)
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as runner_file:
        runner_file.write(runner_script)
        runner_path = runner_file.name

    try:
        process = subprocess.Popen(
            [python_cmd, runner_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = process.communicate(timeout=timeout_seconds)
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else f"Exit code {process.returncode}"
                return None, error_msg
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if results.get("error"):
                    return None, results["error"]
                return results.get("points"), None
            return None, "Results file not created"
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return None, f"Timeout after {timeout_seconds}s"
    finally:
        for path in (code_path, runner_path, results_path):
            if os.path.exists(path):
                with contextlib.suppress(Exception):
                    os.unlink(path)


def evaluate_code(
    code: str,
    num_points: int,
    dimension: int,
    benchmark: float,
    timeout_seconds: int = 30,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Evaluate a min/max distance candidate program."""
    start_time = time.time()
    entry_function = f"min_max_dist_dim{dimension}_{num_points}"

    points, error = _run_code_with_timeout(
        code,
        entry_function=entry_function,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable,
    )

    if error or points is None:
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": error or "Invalid result from code execution",
        }

    points = np.asarray(points, dtype=float)
    if points.shape != (num_points, dimension):
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": f"Invalid points shape: {points.shape}, expected ({num_points}, {dimension})",
        }

    if not np.all(np.isfinite(points)):
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": "Points contain NaN or infinite values",
        }

    if num_points < 2:
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": "At least two points are required",
        }

    pairwise_distances = pdist(points)
    min_distance = float(np.min(pairwise_distances))
    max_distance = float(np.max(pairwise_distances))

    if max_distance <= 0:
        ratio_squared = 0.0
    else:
        ratio_squared = (min_distance / max_distance) ** 2

    combined_score = ratio_squared / benchmark if benchmark > 0 else 0.0

    return {
        "valid": True,
        "score": float(combined_score),
        "min_max_ratio": float(ratio_squared),
        "min_distance": min_distance,
        "max_distance": max_distance,
        "eval_time": time.time() - start_time,
        "error": None,
    }


class MinimizingMaxMinDistanceEvaluator(BaseProblemEvaluator):
    """Evaluator for minimizing the max/min distance ratio."""

    def __init__(
        self,
        num_points: int,
        dimension: int,
        timeout_seconds: int = 30,
        benchmark: float | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.num_points = num_points
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable
        if benchmark is None:
            benchmark = BENCHMARKS.get((dimension, num_points))
        if benchmark is None:
            raise ValueError(
                "Benchmark not provided and not known for "
                f"(dimension={dimension}, num_points={num_points})."
            )
        self.benchmark = float(benchmark)

    def get_problem_spec(self) -> ProblemSpec:
        entry_function = f"min_max_dist_dim{self.dimension}_{self.num_points}"
        return ProblemSpec(
            name="Minimizing Max-Min Distance",
            description=(
                f"Place {self.num_points} points in {self.dimension}D space to maximize the "
                "squared ratio (min pairwise distance / max pairwise distance)^2. "
                "The score is normalized by a benchmark so 1.0 matches the reference value."
            ),
            objective="maximize",
            metric_name="combined_score",
            best_known_solution=1.0,
            entry_function=entry_function,
            return_description=(
                f"A numpy array of shape ({self.num_points}, {self.dimension}) with point coordinates"
            ),
            allowed_modules=["numpy", "math", "random", "scipy", "standard library"],
            constraints=[
                f"Return exactly {self.num_points} points in {self.dimension} dimensions",
                f"Code must complete within {self.timeout_seconds} seconds",
            ],
            example_code=f'''import numpy as np


def {entry_function}():
    points = np.zeros(({self.num_points}, {self.dimension}))
    return points
''',
            secondary_metrics=["min_max_ratio", "min_distance", "max_distance"],
        )

    def evaluate(self, code: str) -> dict[str, Any]:
        return evaluate_code(
            code,
            num_points=self.num_points,
            dimension=self.dimension,
            benchmark=self.benchmark,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
        )
