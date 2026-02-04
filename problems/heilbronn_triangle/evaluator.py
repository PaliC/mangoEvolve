"""Heilbronn Triangle Evaluator for MangoEvolve.

Adapted from OpenEvolve's AlphaEvolve math problems:
https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems/heilbronn_triangle
"""

from __future__ import annotations

import contextlib
import itertools
import os
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from mango_evolve.problem import BaseProblemEvaluator, ProblemSpec

BENCHMARK = 0.036529889880030156
DEFAULT_NUM_POINTS = 11
DEFAULT_TOL = 1e-6


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(
        abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
    )


def _check_inside_triangle(points: np.ndarray, tol: float) -> None:
    """Ensure points lie inside the unit-area equilateral triangle.

    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    for x, y in points:
        cond1 = y >= -tol
        cond2 = np.sqrt(3) * x <= np.sqrt(3) - y + tol
        cond3 = y <= np.sqrt(3) * x + tol
        if not (cond1 and cond2 and cond3):
            raise ValueError(
                f"Point ({x}, {y}) is outside the triangle (tolerance: {tol})."
            )


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
    num_points: int = DEFAULT_NUM_POINTS,
    timeout_seconds: int = 30,
    tol: float = DEFAULT_TOL,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Evaluate a Heilbronn triangle candidate program."""
    start_time = time.time()
    entry_function = f"heilbronn_triangle{num_points}"

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
    if points.shape != (num_points, 2):
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": f"Invalid points shape: {points.shape}, expected ({num_points}, 2)",
        }

    if not np.all(np.isfinite(points)):
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": "Points contain NaN or infinite values",
        }

    try:
        _check_inside_triangle(points, tol)
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "score": 0.0,
            "eval_time": time.time() - start_time,
            "error": str(exc),
        }

    # Compute minimum triangle area among all triples
    min_triangle_area = min(
        _triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)
    )

    triangle_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    ref_area = _triangle_area(*triangle_vertices)
    min_area_normalized = min_triangle_area / ref_area if ref_area > 0 else 0.0
    combined_score = min_area_normalized / BENCHMARK if BENCHMARK > 0 else 0.0

    return {
        "valid": True,
        "score": float(combined_score),
        "min_area_normalized": float(min_area_normalized),
        "min_triangle_area": float(min_triangle_area),
        "eval_time": time.time() - start_time,
        "error": None,
    }


class HeilbronnTriangleEvaluator(BaseProblemEvaluator):
    """Evaluator for the Heilbronn triangle problem (n=11)."""

    def __init__(
        self,
        num_points: int = DEFAULT_NUM_POINTS,
        timeout_seconds: int = 30,
        tol: float = DEFAULT_TOL,
        python_executable: str | None = None,
    ) -> None:
        self.num_points = num_points
        self.timeout_seconds = timeout_seconds
        self.tol = tol
        self.python_executable = python_executable

    def get_problem_spec(self) -> ProblemSpec:
        entry_function = f"heilbronn_triangle{self.num_points}"
        return ProblemSpec(
            name="Heilbronn Triangle",
            description=(
                "Place points on or inside an equilateral triangle of unit area "
                "(vertices at (0,0), (1,0), (0.5, sqrt(3)/2)) to maximize the "
                "minimum area among all triangles formed by any three points. "
                f"Use exactly {self.num_points} points."
            ),
            objective="maximize",
            metric_name="combined_score",
            best_known_solution=1.0,
            entry_function=entry_function,
            return_description=(
                f"A numpy array of shape ({self.num_points}, 2) with point coordinates"
            ),
            allowed_modules=["numpy", "math", "random", "itertools", "scipy", "standard library"],
            constraints=[
                f"Points must lie inside or on the triangle boundary (tolerance {self.tol})",
                f"Code must complete within {self.timeout_seconds} seconds",
                "Return exactly the required number of points",
            ],
            example_code=f'''import numpy as np


def {entry_function}():
    points = np.zeros(({self.num_points}, 2))
    return points
''',
            secondary_metrics=["min_area_normalized", "min_triangle_area"],
        )

    def evaluate(self, code: str) -> dict[str, Any]:
        return evaluate_code(
            code,
            num_points=self.num_points,
            timeout_seconds=self.timeout_seconds,
            tol=self.tol,
            python_executable=self.python_executable,
        )
