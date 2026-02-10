"""
Comprehensive performance profiler for MangoEvolve.

Mocks all LLM API calls with realistic responses extracted from the codebase's
test fixtures. Instruments every critical path to identify bottlenecks that
cause experiments to run over an hour.

Usage:
    uv run python profiling/profile_bottlenecks.py
"""

import cProfile
import io
import json
import multiprocessing
import os
import pstats
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mango_evolve.config import Config, config_from_dict
from mango_evolve.cost_tracker import CostTracker
from mango_evolve.evolution_api import EvolutionAPI, TrialResult
from mango_evolve.llm.client import MockLLMClient
from mango_evolve.llm.providers.base import LLMResponse
from mango_evolve.logger import ExperimentLogger
from mango_evolve.repl import REPLEnvironment
from mango_evolve.root_llm import RootLLMOrchestrator

# ─────────────────────────────────────────────────────────────────────
# Realistic mock data (from test fixtures & actual experiment patterns)
# ─────────────────────────────────────────────────────────────────────

VALID_PACKING_CODE = '''
import numpy as np

def construct_packing():
    """Simple grid-based packing."""
    n = 26
    centers = []
    grid_size = 5
    spacing = 1.0 / (grid_size + 1)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx < n:
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                centers.append([x, y])
                idx += 1
    centers.append([0.5, 0.5])
    centers = np.array(centers[:n])
    radii = np.ones(n) * (spacing / 2 - 0.01)
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1-x, 1-y)
        radii[i] = min(radii[i], max_r)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.99
                radii[i] *= scale
                radii[j] *= scale
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

def run_packing():
    return construct_packing()
'''

IMPROVED_PACKING_CODE = '''
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Optimized packing with scipy minimize."""
    n = 26
    rng = np.random.RandomState(42)
    centers = rng.uniform(0.1, 0.9, (n, 2))
    radii = np.ones(n) * 0.05
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1-x, 1-y)
        radii[i] = min(radii[i], max_r)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.99
                radii[i] *= scale
                radii[j] *= scale
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

def run_packing():
    return construct_packing()
'''


def _make_spawn_response(code: str, approach: str = "grid-based") -> str:
    """Build a realistic root LLM response that spawns children."""
    return f'''I'll spawn children exploring different approaches for circle packing.

```python
results = spawn_children([
    {{
        "prompt": "Generate a circle packing solution using a {approach} approach. Pack 26 circles into a unit square [0,1]x[0,1] to maximize sum of radii. Return (centers, radii, sum_radii) from run_packing().",
        "temperature": 0.7
    }},
    {{
        "prompt": "Generate a circle packing solution using optimization. Pack 26 circles into unit square. Define run_packing() returning (centers, radii, sum_radii).",
        "temperature": 0.8
    }},
    {{
        "prompt": "Generate a circle packing solution using hexagonal packing. 26 circles in unit square. Define run_packing() returning (centers, radii, sum_radii).",
        "temperature": 0.6
    }},
])
print(f"Spawned {{len(results)}} children")
for r in results:
    print(f"  {{r.trial_id}}: score={{r.score:.6f}} success={{r.success}}")
```
'''


def _make_selection_response(gen: int) -> str:
    """Build a realistic selection response."""
    return f'''Based on the trial results, I'll select the best trials to carry forward.

```selection
{{
    "selections": [
        {{"trial_id": "trial_{gen}_0", "reasoning": "Best score in generation", "category": "performance"}},
        {{"trial_id": "trial_{gen}_1", "reasoning": "Different approach with potential", "category": "diversity"}}
    ],
    "summary": "Selected top performer and a diverse approach for generation {gen}"
}}
```
'''


def _make_analysis_spawn_response(gen: int) -> str:
    """Build a response that includes analysis followed by spawning."""
    return f'''Let me analyze previous results and spawn improved children.

```python
# Analyze previous results
top = trials.filter(success=True, sort_by="-score", limit=3)
for t in top:
    print(f"  {{t.trial_id}}: {{t.score:.6f}}")

scratchpad.append(f"\\n## Gen {gen}: Top scores: " + ", ".join(f"{{t.score:.4f}}" for t in top))
```

Now spawning children based on the analysis:

```python
best = trials.filter(success=True, sort_by="-score", limit=1)
parent_code = best[0].code if best else ""
parent_id = best[0].trial_id if best else None

results = spawn_children([
    {{
        "prompt": f"Improve this circle packing (score={{best[0].score if best else 0}}):\\n{{parent_code}}\\nTry to increase the score.",
        "parent_id": parent_id,
        "temperature": 0.7
    }},
    {{
        "prompt": "Generate a novel circle packing using basin-hopping optimization. 26 circles in [0,1]x[0,1]. Define run_packing() returning (centers, radii, sum_radii).",
        "temperature": 0.9
    }},
    {{
        "prompt": f"Refine this approach: {{parent_code}}\\nFocus on boundary handling.",
        "parent_id": parent_id,
        "temperature": 0.5
    }},
])
```
'''


# ─────────────────────────────────────────────────────────────────────
# Timing infrastructure
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TimingRecord:
    name: str
    duration: float
    count: int = 1
    details: str = ""


class TimingCollector:
    """Collects timing data for various operations."""

    def __init__(self):
        self.records: list[TimingRecord] = []
        self._active_timers: dict[str, float] = {}

    @contextmanager
    def time(self, name: str, details: str = ""):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append(TimingRecord(name, elapsed, details=details))

    def summary(self) -> dict[str, dict]:
        """Aggregate timing data by name."""
        aggregated: dict[str, dict] = {}
        for record in self.records:
            if record.name not in aggregated:
                aggregated[record.name] = {
                    "total_time": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": 0.0,
                    "details": [],
                }
            agg = aggregated[record.name]
            agg["total_time"] += record.duration
            agg["count"] += 1
            agg["min"] = min(agg["min"], record.duration)
            agg["max"] = max(agg["max"], record.duration)
            if record.details:
                agg["details"].append(record.details)
        # Compute averages
        for name, agg in aggregated.items():
            agg["avg"] = agg["total_time"] / agg["count"] if agg["count"] else 0
        return aggregated


timing = TimingCollector()


# ─────────────────────────────────────────────────────────────────────
# Mock spawn_child that simulates realistic worker behavior
# ─────────────────────────────────────────────────────────────────────

def mock_spawn_child(args: tuple) -> dict[str, Any]:
    """Mock spawn_child that does real evaluation but skips LLM API calls."""
    import uuid
    from mango_evolve.parallel_worker import _parse_worker_args, _write_trial_file
    from mango_evolve.config import load_evaluator_from_string
    from mango_evolve.utils.code_extraction import extract_python_code, extract_reasoning

    (
        prompt, parent_id, model, evaluator_fn, evaluator_kwargs,
        max_tokens, temperature, trial_id, generation, experiment_dir,
        system_prompt, provider, model_alias,
    ) = _parse_worker_args(args)

    call_id = str(uuid.uuid4())

    # Simulate LLM response (no API call) - use valid packing code
    response_text = f"Here's a circle packing solution:\n\n```python\n{VALID_PACKING_CODE}\n```"
    code = VALID_PACKING_CODE
    reasoning = "Grid-based approach with neighbor-aware radii."

    model_config = {"model": model, "temperature": temperature}

    # Do REAL evaluation (this is where real time is spent)
    eval_start = time.perf_counter()
    try:
        evaluator = load_evaluator_from_string(evaluator_fn, evaluator_kwargs)
        metrics = evaluator.evaluate(code)
    except Exception as e:
        metrics = {"valid": False, "error": str(e)}
    eval_duration = time.perf_counter() - eval_start

    success = bool(metrics.get("valid", False))
    error = metrics.get("error") if not success else None

    # Write trial file (real I/O)
    io_start = time.perf_counter()
    _write_trial_file(
        trial_id=trial_id, generation=generation,
        experiment_dir=experiment_dir, code=code,
        metrics=metrics, prompt=prompt, response=response_text,
        reasoning=reasoning, parent_id=parent_id,
        model_config=model_config,
    )
    io_duration = time.perf_counter() - io_start

    return {
        "trial_id": trial_id,
        "prompt": prompt,
        "parent_id": parent_id,
        "response_text": response_text,
        "code": code,
        "reasoning": reasoning,
        "metrics": metrics,
        "success": success,
        "error": str(error) if error else None,
        "input_tokens": 1500,
        "output_tokens": 800,
        "call_id": call_id,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 100,
        "model_alias": model_alias,
        "model_config": model_config,
        "_profile_eval_time": eval_duration,
        "_profile_io_time": io_duration,
    }


def mock_query_llm(args: tuple) -> dict[str, Any]:
    """Mock query_llm that returns a realistic response without API calls."""
    import uuid
    from mango_evolve.parallel_worker import _parse_worker_args

    (
        prompt, parent_id, model, _evaluator_fn, _evaluator_kwargs,
        max_tokens, temperature, trial_id, _generation, _experiment_dir,
        system_prompt, provider, model_alias,
    ) = _parse_worker_args(args)

    return {
        "trial_id": trial_id,
        "prompt": prompt,
        "parent_id": parent_id,
        "response_text": "The grid-based approach works well because it ensures even spacing.",
        "code": "",
        "reasoning": "Analysis of approaches.",
        "metrics": {},
        "success": True,
        "error": None,
        "input_tokens": 500,
        "output_tokens": 200,
        "call_id": str(uuid.uuid4()),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "model_alias": model_alias,
        "model_config": {"model": model, "temperature": temperature},
    }


# ─────────────────────────────────────────────────────────────────────
# Individual component profilers
# ─────────────────────────────────────────────────────────────────────

def profile_evaluator_subprocess():
    """Profile the evaluation subprocess (circle packing)."""
    print("\n" + "=" * 70)
    print("PROFILING: Evaluator (subprocess execution)")
    print("=" * 70)

    from mango_evolve.evaluation.circle_packing import (
        CirclePackingEvaluator,
        evaluate_code,
        run_code_with_timeout,
        validate_packing,
    )
    import numpy as np

    evaluator = CirclePackingEvaluator(n_circles=26, timeout_seconds=30)

    # Profile full evaluate() call (subprocess + validation)
    times_full = []
    for i in range(5):
        with timing.time("evaluator.evaluate()", f"run {i}"):
            start = time.perf_counter()
            result = evaluator.evaluate(VALID_PACKING_CODE)
            elapsed = time.perf_counter() - start
            times_full.append(elapsed)
            print(f"  Run {i}: {elapsed:.3f}s (valid={result['valid']}, score={result.get('score', 0):.4f})")

    print(f"\n  evaluate() summary:")
    print(f"    Mean: {sum(times_full)/len(times_full):.3f}s")
    print(f"    Min:  {min(times_full):.3f}s")
    print(f"    Max:  {max(times_full):.3f}s")

    # Profile subprocess overhead specifically
    print("\n  Subprocess overhead breakdown:")
    times_subprocess = []
    for i in range(5):
        start = time.perf_counter()
        centers, radii, sum_r, error = run_code_with_timeout(
            VALID_PACKING_CODE, timeout_seconds=30
        )
        elapsed = time.perf_counter() - start
        times_subprocess.append(elapsed)
    print(f"    run_code_with_timeout mean: {sum(times_subprocess)/len(times_subprocess):.3f}s")

    # Profile validation only (no subprocess)
    centers, radii, sum_r, _ = run_code_with_timeout(VALID_PACKING_CODE, timeout_seconds=30)
    if centers is not None:
        times_validate = []
        for i in range(100):
            start = time.perf_counter()
            validate_packing(centers, radii, 26)
            elapsed = time.perf_counter() - start
            times_validate.append(elapsed)
        print(f"    validate_packing mean: {sum(times_validate)/len(times_validate)*1000:.3f}ms (x100)")

    # Profile with broken code
    broken_code = "def run_packing():\n    return undefined_var"
    start = time.perf_counter()
    result = evaluator.evaluate(broken_code)
    elapsed = time.perf_counter() - start
    print(f"    Broken code evaluation: {elapsed:.3f}s")

    return times_full


def _dummy_worker(x):
    """Module-level dummy worker for pool.map (must be picklable)."""
    time.sleep(0.001)
    return x * 2


def profile_multiprocessing_pool():
    """Profile multiprocessing pool creation and worker dispatch."""
    print("\n" + "=" * 70)
    print("PROFILING: Multiprocessing Pool overhead")
    print("=" * 70)

    # Profile pool creation alone
    pool_create_times = []
    for i in range(5):
        start = time.perf_counter()
        pool = multiprocessing.Pool(processes=4)
        elapsed = time.perf_counter() - start
        pool_close_start = time.perf_counter()
        pool.close()
        pool.join()
        close_elapsed = time.perf_counter() - pool_close_start
        pool_create_times.append(elapsed)
        print(f"  Pool creation {i}: {elapsed*1000:.1f}ms, close: {close_elapsed*1000:.1f}ms")

    print(f"\n  Pool creation mean: {sum(pool_create_times)/len(pool_create_times)*1000:.1f}ms")

    # Profile pool.map with mock function (no real work)
    map_times = []
    for n_workers in [1, 2, 4, 8]:
        start = time.perf_counter()
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_dummy_worker, range(4))
        elapsed = time.perf_counter() - start
        map_times.append((n_workers, elapsed))
        print(f"  pool.map(4 items, {n_workers} workers): {elapsed*1000:.1f}ms")

    # Profile pool.map with evaluator (real work)
    print("\n  Pool.map with real evaluator work (3 children):")
    with tempfile.TemporaryDirectory() as tmpdir:
        worker_args = []
        for i in range(3):
            worker_args.append((
                "Generate circle packing",  # prompt
                None,  # parent_id
                "mock-model",  # model
                "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator",
                {"n_circles": 26, "timeout_seconds": 30},
                8192,  # max_tokens
                0.7,  # temperature
                f"trial_0_{i}",
                0,  # generation
                tmpdir,
                None,  # system_prompt
                "anthropic",
                "default",
            ))

        for n_workers in [1, 2, 3]:
            start = time.perf_counter()
            with multiprocessing.Pool(processes=n_workers) as pool:
                results = pool.map(mock_spawn_child, worker_args)
            elapsed = time.perf_counter() - start
            with timing.time("pool.map(spawn_child)", f"{n_workers} workers, 3 children"):
                pass  # Already timed
            eval_times = [r.get("_profile_eval_time", 0) for r in results]
            io_times = [r.get("_profile_io_time", 0) for r in results]
            print(f"  {n_workers} workers: total={elapsed:.3f}s, "
                  f"eval_avg={sum(eval_times)/len(eval_times):.3f}s, "
                  f"io_avg={sum(io_times)/len(io_times)*1000:.1f}ms")


def profile_repl_execution():
    """Profile REPL code execution overhead."""
    print("\n" + "=" * 70)
    print("PROFILING: REPL execution overhead")
    print("=" * 70)

    # Simple REPL without API functions
    repl = REPLEnvironment()

    # Profile basic code execution
    simple_codes = [
        ("assignment", "x = 42"),
        ("computation", "import math\nresult = sum(math.sqrt(i) for i in range(1000))"),
        ("list comprehension", "data = [i**2 for i in range(10000)]"),
        ("function definition", "def helper(n):\n    return sum(range(n))\nhelper(1000)"),
        ("numpy operations", "import numpy as np\narr = np.random.rand(1000)\nnp.sort(arr)"),
    ]

    for name, code in simple_codes:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = repl.execute(code)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg = sum(times) / len(times)
        with timing.time("repl.execute()", name):
            pass
        print(f"  {name}: {avg*1000:.2f}ms avg (success={result.success})")

    # Profile REPL with injected API functions (simulating evolution context)
    mock_api = {
        "spawn_children": lambda children, **kw: [],
        "evaluate_program": lambda code: {"valid": True, "score": 2.0},
        "terminate_evolution": lambda reason, **kw: {},
        "update_scratchpad": lambda content: {},
        "query_llm": lambda queries, **kw: [],
        "scratchpad": MagicMock(),
        "trials": MagicMock(),
    }
    repl_with_api = REPLEnvironment(namespace=mock_api)

    # Profile the overhead of having API functions injected
    for name, code in simple_codes[:3]:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            repl_with_api.execute(code)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg = sum(times) / len(times)
        print(f"  {name} (with API): {avg*1000:.2f}ms avg")

    # Profile code extraction (regex)
    from mango_evolve.utils.code_extraction import extract_python_blocks, extract_selection_block

    long_response = "Here's my analysis.\n\n```python\nresults = spawn_children([{'prompt': 'test'}])\n```\n\nNow let me explain...\n\n```python\nprint('hello world')\n```\n\nAnd a selection:\n```selection\n{\"selections\": [{\"trial_id\": \"trial_0_0\"}]}\n```"

    times_extract = []
    for _ in range(1000):
        start = time.perf_counter()
        blocks = extract_python_blocks(long_response)
        elapsed = time.perf_counter() - start
        times_extract.append(elapsed)
    print(f"\n  extract_python_blocks (1000x): {sum(times_extract)/len(times_extract)*1000:.3f}ms avg")

    times_selection = []
    for _ in range(1000):
        start = time.perf_counter()
        extract_selection_block(long_response)
        elapsed = time.perf_counter() - start
        times_selection.append(elapsed)
    print(f"  extract_selection_block (1000x): {sum(times_selection)/len(times_selection)*1000:.3f}ms avg")


def profile_cost_tracker():
    """Profile cost tracker operations."""
    print("\n" + "=" * 70)
    print("PROFILING: CostTracker operations")
    print("=" * 70)

    config = _make_test_config()
    tracker = CostTracker(config)

    # Profile record_usage (called after every LLM call)
    import uuid
    times_record = []
    for i in range(200):
        start = time.perf_counter()
        tracker.record_usage(
            input_tokens=5000,
            output_tokens=2000,
            llm_type="root" if i % 3 == 0 else "child:default",
            call_id=str(uuid.uuid4()),
            cache_creation_input_tokens=500,
            cache_read_input_tokens=300,
        )
        elapsed = time.perf_counter() - start
        times_record.append(elapsed)

    avg = sum(times_record) / len(times_record)
    print(f"  record_usage (200 calls): {avg*1000:.3f}ms avg")

    # Profile get_summary (called in update_pbar_postfix, EVERY code block execution)
    times_summary = []
    for _ in range(100):
        start = time.perf_counter()
        summary = tracker.get_summary()
        elapsed = time.perf_counter() - start
        times_summary.append(elapsed)

    avg = sum(times_summary) / len(times_summary)
    print(f"  get_summary (100 calls, {len(tracker.usage_log)} log entries): {avg*1000:.3f}ms avg")

    # Profile check_budget / raise_if_over_budget
    times_budget = []
    for _ in range(1000):
        start = time.perf_counter()
        tracker.raise_if_over_budget()
        elapsed = time.perf_counter() - start
        times_budget.append(elapsed)
    avg = sum(times_budget) / len(times_budget)
    print(f"  raise_if_over_budget (1000 calls): {avg*1000:.4f}ms avg")

    # Profile to_dict (serialization, called at end)
    start = time.perf_counter()
    d = tracker.to_dict()
    elapsed = time.perf_counter() - start
    print(f"  to_dict ({len(tracker.usage_log)} entries): {elapsed*1000:.1f}ms")


def profile_logger_io():
    """Profile file I/O operations from the logger."""
    print("\n" + "=" * 70)
    print("PROFILING: Logger file I/O")
    print("=" * 70)

    config = _make_test_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = _make_test_config_dict()
        config_dict["experiment"]["output_dir"] = tmpdir
        config = config_from_dict(config_dict)
        logger = ExperimentLogger(config)
        logger.create_experiment_directory()

        # Profile log_trial (JSON write per trial)
        times_trial = []
        for i in range(20):
            start = time.perf_counter()
            logger.log_trial(
                trial_id=f"trial_0_{i}",
                generation=0,
                code=VALID_PACKING_CODE,
                metrics={"valid": True, "score": 2.4 + i * 0.01},
                prompt="Generate circle packing" * 10,  # Realistic prompt length
                response="Here is the solution:\n```python\n" + VALID_PACKING_CODE + "\n```",
                reasoning="Grid-based approach with optimization" * 5,
                parent_id=f"trial_0_{i-1}" if i > 0 else None,
                model_config={"model": "claude-sonnet", "temperature": 0.7},
            )
            elapsed = time.perf_counter() - start
            times_trial.append(elapsed)

        avg = sum(times_trial) / len(times_trial)
        print(f"  log_trial (20 calls): {avg*1000:.2f}ms avg")

        # Profile log_root_turn (JSONL append per turn)
        times_turn = []
        for i in range(50):
            start = time.perf_counter()
            logger.log_root_turn(
                turn_number=i,
                role="assistant" if i % 2 == 0 else "user",
                content="This is a realistic message content " * 20,
                code_executed="x = 42" if i % 3 == 0 else None,
                execution_result="42" if i % 3 == 0 else None,
            )
            elapsed = time.perf_counter() - start
            times_turn.append(elapsed)

        avg = sum(times_turn) / len(times_turn)
        print(f"  log_root_turn (50 calls): {avg*1000:.2f}ms avg")

        # Profile log_generation (summary JSON)
        trials_data = [
            {
                "trial_id": f"trial_0_{i}",
                "metrics": {"valid": True, "score": 2.4 + i * 0.01},
                "code": VALID_PACKING_CODE,
            }
            for i in range(10)
        ]
        start = time.perf_counter()
        logger.log_generation(
            generation=0,
            trials=trials_data,
            selected_trial_ids=["trial_0_0", "trial_0_1"],
            selection_reasoning="Top performers",
            best_trial_id="trial_0_9",
            best_score=2.49,
        )
        elapsed = time.perf_counter() - start
        print(f"  log_generation: {elapsed*1000:.2f}ms")

        # Profile save_scratchpad
        start = time.perf_counter()
        logger.save_scratchpad(
            generation=0,
            scratchpad="## Key Insights\n- Grid approaches work\n- Score ~2.4\n" * 20,
            lineage_map="trial_0_0 (2.40) -> trial_1_0 (2.42)\n" * 20,
        )
        elapsed = time.perf_counter() - start
        print(f"  save_scratchpad: {elapsed*1000:.2f}ms")

        # Profile save_experiment
        start = time.perf_counter()
        logger.save_experiment(
            termination_reason="max_generations_reached",
            scratchpad="Final scratchpad content",
        )
        elapsed = time.perf_counter() - start
        print(f"  save_experiment: {elapsed*1000:.2f}ms")


def profile_prompt_construction():
    """Profile system prompt and message construction."""
    print("\n" + "=" * 70)
    print("PROFILING: Prompt construction")
    print("=" * 70)

    from mango_evolve.llm.prompts import (
        build_root_system_prompt_static,
        get_root_system_prompt_parts_with_models,
        build_child_system_prompt,
    )
    from mango_evolve.problem import ProblemSpec

    spec = ProblemSpec(
        name="Circle Packing",
        description="Pack 26 circles into [0,1]x[0,1] to maximize sum of radii.",
        objective="maximize",
        metric_name="sum of radii",
        entry_function="run_packing",
        return_description="Tuple of (centers, radii, sum_radii)",
        best_known_solution=2.6359850561146603,
        helper_functions=["construct_packing"],
        allowed_modules=["numpy", "scipy"],
        constraints=["Code must complete within 30 seconds"],
    )

    from mango_evolve.config import ChildLLMConfig
    child_configs = {
        "default": ChildLLMConfig(
            model="claude-sonnet-4-20250514",
            cost_per_million_input_tokens=3.0,
            cost_per_million_output_tokens=15.0,
            alias="default",
        ),
    }

    # Profile static prompt build
    times = []
    for _ in range(100):
        start = time.perf_counter()
        static = build_root_system_prompt_static(spec)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    print(f"  build_root_system_prompt_static (100x): {sum(times)/len(times)*1000:.3f}ms avg")
    print(f"    Prompt size: {len(static)} chars (~{len(static)//4} tokens)")

    # Profile structured prompt build (called every generation)
    times = []
    for gen in range(10):
        start = time.perf_counter()
        parts = get_root_system_prompt_parts_with_models(
            spec=spec,
            child_llm_configs=child_configs,
            default_child_llm_alias="default",
            max_children_per_generation=4,
            max_generations=10,
            current_generation=gen,
            timeout_seconds=300,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    print(f"  get_root_system_prompt_parts_with_models (10x): {sum(times)/len(times)*1000:.3f}ms avg")
    total_size = sum(len(p["text"]) for p in parts)
    print(f"    System prompt total size: {total_size} chars (~{total_size//4} tokens)")

    # Profile child system prompt
    times = []
    for _ in range(100):
        start = time.perf_counter()
        child_prompt = build_child_system_prompt(spec)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    print(f"  build_child_system_prompt (100x): {sum(times)/len(times)*1000:.3f}ms avg")
    print(f"    Child prompt size: {len(child_prompt)} chars (~{len(child_prompt)//4} tokens)")


def profile_lineage_map_building():
    """Profile lineage map construction with many trials."""
    print("\n" + "=" * 70)
    print("PROFILING: Lineage map building (scales with trial count)")
    print("=" * 70)

    config = _make_test_config()
    cost_tracker = CostTracker(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = _make_test_config_dict()
        config_dict["experiment"]["output_dir"] = tmpdir
        config = config_from_dict(config_dict)
        logger = ExperimentLogger(config)
        logger.create_experiment_directory()

        from mango_evolve.problem import ProblemSpec
        spec = ProblemSpec(
            name="Circle Packing",
            description="Pack circles",
            objective="maximize",
            metric_name="sum of radii",
            entry_function="run_packing",
            return_description="(centers, radii, sum_radii)",
        )

        from mango_evolve.config import ChildLLMConfig
        child_configs = {
            "default": ChildLLMConfig(
                model="test", cost_per_million_input_tokens=3.0,
                cost_per_million_output_tokens=15.0, alias="default",
                calibration_calls=0,
            ),
        }

        api = EvolutionAPI(
            evaluator=MagicMock(),
            problem_spec=spec,
            child_llm_configs=child_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=20,
            max_children_per_generation=10,
            default_child_llm_alias="default",
        )
        api.end_calibration_phase()

        # Simulate growing trial populations
        for trial_count in [10, 50, 100, 200]:
            api.all_trials.clear()
            for i in range(trial_count):
                gen = i // 10
                trial = TrialResult(
                    trial_id=f"trial_{gen}_{i % 10}",
                    code="x = 1",
                    metrics={"score": 2.0 + i * 0.001, "valid": True},
                    prompt="test",
                    response="test",
                    reasoning=f"Approach {i}" * 10,
                    success=True,
                    parent_id=f"trial_{gen-1}_{i % 10}" if gen > 0 else None,
                    generation=gen,
                )
                api.all_trials[trial.trial_id] = trial

            start = time.perf_counter()
            lineage = api._build_lineage_map()
            elapsed = time.perf_counter() - start
            print(f"  {trial_count} trials: {elapsed*1000:.1f}ms ({len(lineage)} chars)")
            with timing.time("_build_lineage_map", f"{trial_count} trials"):
                api._build_lineage_map()


def profile_cost_summary_scaling():
    """Profile CostTracker.get_summary() as usage log grows."""
    print("\n" + "=" * 70)
    print("PROFILING: CostTracker.get_summary() scaling")
    print("=" * 70)

    import uuid

    config = _make_test_config()

    for n_entries in [10, 50, 100, 500, 1000]:
        tracker = CostTracker(config)
        for i in range(n_entries):
            tracker.record_usage(
                input_tokens=5000, output_tokens=2000,
                llm_type="root" if i % 3 == 0 else "child:default",
                call_id=str(uuid.uuid4()),
                cache_creation_input_tokens=500,
                cache_read_input_tokens=300,
            )

        times = []
        for _ in range(100):
            start = time.perf_counter()
            tracker.get_summary()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = sum(times) / len(times)
        print(f"  {n_entries:>5} entries: {avg*1000:.3f}ms avg (100 calls)")


def profile_update_pbar_postfix():
    """
    Profile the update_pbar_postfix callback which is called after every
    code block execution. This is a key hot path.
    """
    print("\n" + "=" * 70)
    print("PROFILING: update_pbar_postfix (hot path, called per code block)")
    print("=" * 70)

    config = _make_test_config()
    cost_tracker = CostTracker(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = _make_test_config_dict()
        config_dict["experiment"]["output_dir"] = tmpdir
        config = config_from_dict(config_dict)
        logger = ExperimentLogger(config)
        logger.create_experiment_directory()

        from mango_evolve.problem import ProblemSpec
        spec = ProblemSpec(
            name="Circle Packing",
            description="Pack circles",
            objective="maximize",
            metric_name="sum of radii",
            entry_function="run_packing",
            return_description="(centers, radii, sum_radii)",
        )

        from mango_evolve.config import ChildLLMConfig
        child_configs = {
            "default": ChildLLMConfig(
                model="test", cost_per_million_input_tokens=3.0,
                cost_per_million_output_tokens=15.0, alias="default",
                calibration_calls=0,
            ),
        }

        api = EvolutionAPI(
            evaluator=MagicMock(),
            problem_spec=spec,
            child_llm_configs=child_configs,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=20,
            max_children_per_generation=10,
            default_child_llm_alias="default",
        )
        api.end_calibration_phase()

        # Simulate accumulated trial data
        import uuid
        for i in range(100):
            trial = TrialResult(
                trial_id=f"trial_{i//10}_{i%10}",
                code="x=1", metrics={"score": 2.0+i*0.001, "valid": True},
                prompt="t", response="r", reasoning="reason",
                success=True, generation=i//10,
            )
            api.all_trials[trial.trial_id] = trial
            cost_tracker.record_usage(
                input_tokens=5000, output_tokens=2000,
                llm_type="child:default",
                call_id=str(uuid.uuid4()),
            )

        # This simulates what update_pbar_postfix does
        def simulate_pbar_update():
            cost_summary = cost_tracker.get_summary()
            trials_count = len(api.all_trials)
            successes = sum(1 for t in api.all_trials.values() if t.success)
            best_score = max(
                (t.metrics.get("score", 0) for t in api.all_trials.values() if t.success),
                default=0,
            )
            current_gen_trials = len(api.generations[api.current_generation].trials)

        times = []
        for _ in range(500):
            start = time.perf_counter()
            simulate_pbar_update()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = sum(times) / len(times)
        print(f"  update_pbar_postfix (500 calls, 100 trials): {avg*1000:.3f}ms avg")
        print(f"  Total time for 500 calls: {sum(times)*1000:.1f}ms")

        # Estimate impact over a full experiment
        # ~3 calls per generation (spawn, selection, advance) x 5 gens = 15 calls
        # But with analysis code blocks, could be 5-10 per generation = 25-50 total
        estimated_calls = 50
        print(f"  Estimated impact over experiment ({estimated_calls} calls): {avg * estimated_calls * 1000:.1f}ms")


def profile_evaluator_loading():
    """Profile dynamic evaluator loading (done in each worker process)."""
    print("\n" + "=" * 70)
    print("PROFILING: Dynamic evaluator loading (per worker process)")
    print("=" * 70)

    from mango_evolve.config import load_evaluator_from_string

    times = []
    for _ in range(20):
        start = time.perf_counter()
        evaluator = load_evaluator_from_string(
            "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator",
            {"n_circles": 26, "timeout_seconds": 30},
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"  load_evaluator_from_string (20 calls): {avg*1000:.2f}ms avg")
    print(f"  NOTE: Each worker process re-imports and re-instantiates the evaluator")


def profile_end_to_end_generation():
    """
    Profile a complete single generation with mocked LLM calls.
    This is the most representative test of real-world bottlenecks.
    """
    print("\n" + "=" * 70)
    print("PROFILING: End-to-end single generation (mocked LLM)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = _make_test_config_dict()
        config_dict["experiment"]["output_dir"] = tmpdir
        config_dict["evolution"]["max_generations"] = 2
        config_dict["evolution"]["max_children_per_generation"] = 3
        config = config_from_dict(config_dict)

        cost_tracker = CostTracker(config)
        logger = ExperimentLogger(config)
        logger.create_experiment_directory()

        # Create mock root LLM with realistic responses
        mock_root = MockLLMClient(
            model="claude-sonnet",
            cost_tracker=cost_tracker,
            llm_type="root",
            responses=[
                # Gen 0: spawn children
                _make_spawn_response(VALID_PACKING_CODE, "grid-based"),
                # Gen 0: selection
                _make_selection_response(0),
                # Gen 1: spawn children
                _make_analysis_spawn_response(1),
                # Gen 1: selection
                _make_selection_response(1),
            ],
        )

        # Patch spawn_child to use our mock (skip LLM API, do real eval)
        with patch("mango_evolve.evolution_api.spawn_child", mock_spawn_child), \
             patch("mango_evolve.evolution_api.query_llm_worker", mock_query_llm):
            orchestrator = RootLLMOrchestrator(
                config=config,
                root_llm=mock_root,
                logger=logger,
            )
            # Skip calibration
            orchestrator.evolution_api.end_calibration_phase()

            total_start = time.perf_counter()

            # Time individual phases manually
            phases = {}

            # Phase 1: Build generation start message
            phase_start = time.perf_counter()
            msg = orchestrator._build_generation_start_message()
            orchestrator.messages = [{"role": "user", "content": msg}]
            phases["build_generation_start_message"] = time.perf_counter() - phase_start

            # Phase 2: Get system prompt
            phase_start = time.perf_counter()
            system_prompt = orchestrator._get_system_prompt()
            phases["get_system_prompt"] = time.perf_counter() - phase_start

            # Phase 3: Root LLM call (mocked)
            phase_start = time.perf_counter()
            response = orchestrator.root_llm.generate(
                messages=orchestrator.messages,
                system=system_prompt,
                max_tokens=8192,
                temperature=0.7,
            )
            phases["root_llm_generate (mock)"] = time.perf_counter() - phase_start

            # Phase 4: Extract code blocks
            phase_start = time.perf_counter()
            code_blocks = orchestrator.extract_code_blocks(response.content)
            phases["extract_code_blocks"] = time.perf_counter() - phase_start

            # Phase 5: Execute code blocks (this includes spawn_children)
            phase_start = time.perf_counter()
            for code in code_blocks:
                result = orchestrator.execute_code_in_repl(code)
            phases["execute_code_blocks (incl spawn)"] = time.perf_counter() - phase_start

            total_elapsed = time.perf_counter() - total_start

            print(f"\n  Phase breakdown for one generation:")
            for phase_name, duration in phases.items():
                pct = (duration / total_elapsed * 100) if total_elapsed > 0 else 0
                print(f"    {phase_name}: {duration*1000:.1f}ms ({pct:.1f}%)")
            print(f"    TOTAL: {total_elapsed*1000:.1f}ms")


def profile_full_orchestrator_run():
    """
    Profile a complete orchestrator run (multiple generations) with mocked LLM.
    Most realistic simulation of an actual experiment.
    """
    print("\n" + "=" * 70)
    print("PROFILING: Full orchestrator run (3 generations, mocked LLM)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = _make_test_config_dict()
        config_dict["experiment"]["output_dir"] = tmpdir
        config_dict["evolution"]["max_generations"] = 3
        config_dict["evolution"]["max_children_per_generation"] = 3
        config = config_from_dict(config_dict)

        cost_tracker = CostTracker(config)
        logger = ExperimentLogger(config)
        logger.create_experiment_directory()

        # Build enough responses for 3 generations (spawn + selection each)
        responses = []
        for gen in range(3):
            responses.append(_make_spawn_response(VALID_PACKING_CODE, f"approach-gen{gen}"))
            responses.append(_make_selection_response(gen))

        mock_root = MockLLMClient(
            model="claude-sonnet",
            cost_tracker=cost_tracker,
            llm_type="root",
            responses=responses,
        )

        with patch("mango_evolve.evolution_api.spawn_child", mock_spawn_child), \
             patch("mango_evolve.evolution_api.query_llm_worker", mock_query_llm):
            orchestrator = RootLLMOrchestrator(
                config=config,
                root_llm=mock_root,
                logger=logger,
            )
            orchestrator.evolution_api.end_calibration_phase()

            start = time.perf_counter()
            result = orchestrator.run()
            total_elapsed = time.perf_counter() - start

            print(f"\n  Full run result:")
            print(f"    Terminated: {result.terminated}, reason: {result.reason}")
            print(f"    Generations: {result.num_generations}")
            print(f"    Total trials: {result.total_trials}, successful: {result.successful_trials}")
            print(f"    Best score: {result.best_score:.6f}")
            print(f"    Total time: {total_elapsed:.3f}s")
            print(f"    Time per generation: {total_elapsed/max(result.num_generations, 1):.3f}s")


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def _make_test_config_dict() -> dict:
    return {
        "experiment": {
            "name": "profile_test",
            "output_dir": "./profile_experiments",
        },
        "root_llm": {
            "model": "claude-sonnet-4-20250514",
            "cost_per_million_input_tokens": 3.0,
            "cost_per_million_output_tokens": 15.0,
            "max_iterations": 30,
        },
        "child_llms": [
            {
                "alias": "default",
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "cost_per_million_input_tokens": 3.0,
                "cost_per_million_output_tokens": 15.0,
                "calibration_calls": 0,
            }
        ],
        "default_child_llm_alias": "default",
        "evaluation": {
            "evaluator_fn": "problems.circle_packing.evaluator:CirclePackingEvaluator",
            "evaluator_kwargs": {
                "n_circles": 26,
                "timeout_seconds": 30,
            },
        },
        "evolution": {
            "max_generations": 3,
            "max_children_per_generation": 3,
        },
        "budget": {
            "max_total_cost": 100.0,
        },
    }


def _make_test_config() -> Config:
    return config_from_dict(_make_test_config_dict())


# ─────────────────────────────────────────────────────────────────────
# Main profiling runner
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MangoEvolve Performance Profiling Suite")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print()

    # Run all profilers
    profile_evaluator_subprocess()
    profile_multiprocessing_pool()
    profile_repl_execution()
    profile_cost_tracker()
    profile_cost_summary_scaling()
    profile_update_pbar_postfix()
    profile_logger_io()
    profile_prompt_construction()
    profile_lineage_map_building()
    profile_evaluator_loading()
    profile_end_to_end_generation()
    profile_full_orchestrator_run()

    # Final summary
    print("\n" + "=" * 70)
    print("BOTTLENECK SUMMARY")
    print("=" * 70)

    summary = timing.summary()
    sorted_ops = sorted(summary.items(), key=lambda x: x[1]["total_time"], reverse=True)

    print(f"\n{'Operation':<45} {'Total':>8} {'Count':>6} {'Avg':>10} {'Max':>10}")
    print("-" * 85)
    for name, stats in sorted_ops:
        total = f"{stats['total_time']:.3f}s"
        count = str(stats["count"])
        avg = f"{stats['avg']*1000:.1f}ms"
        mx = f"{stats['max']*1000:.1f}ms"
        print(f"  {name:<43} {total:>8} {count:>6} {avg:>10} {mx:>10}")

    # Run cProfile on the full orchestrator for detailed function-level data
    print("\n" + "=" * 70)
    print("cProfile: Full orchestrator run (top 30 functions by cumulative time)")
    print("=" * 70)

    profiler = cProfile.Profile()
    profiler.enable()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = _make_test_config_dict()
        config_dict["experiment"]["output_dir"] = tmpdir
        config = config_from_dict(config_dict)
        cost_tracker = CostTracker(config)
        logger = ExperimentLogger(config)
        logger.create_experiment_directory()

        responses = []
        for gen in range(3):
            responses.append(_make_spawn_response(VALID_PACKING_CODE, f"gen{gen}"))
            responses.append(_make_selection_response(gen))

        mock_root = MockLLMClient(
            model="mock", cost_tracker=cost_tracker, llm_type="root",
            responses=responses,
        )

        with patch("mango_evolve.evolution_api.spawn_child", mock_spawn_child), \
             patch("mango_evolve.evolution_api.query_llm_worker", mock_query_llm):
            orch = RootLLMOrchestrator(
                config=config, root_llm=mock_root, logger=logger,
            )
            orch.evolution_api.end_calibration_phase()
            orch.run()

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())

    # Save cProfile data for external analysis
    profile_path = Path(__file__).parent / "profile_results.prof"
    profiler.dump_stats(str(profile_path))
    print(f"\ncProfile data saved to: {profile_path}")
    print("Analyze with: python -m pstats profiling/profile_results.prof")
    print("Or visualize: pip install snakeviz && snakeviz profiling/profile_results.prof")


if __name__ == "__main__":
    main()
