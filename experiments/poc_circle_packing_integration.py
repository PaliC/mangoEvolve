"""
Proof of Concept: Circle Packing Full Integration

This validates the complete evolution flow for circle packing:
1. Root LLM REPL spawns child LLMs
2. Child LLMs generate circle packing code
3. Code is evaluated for geometric validity and sum of radii
4. Results feed back to Root LLM for selection
5. Generations advance based on Root LLM decisions

Uses mock LLMs with realistic code generation.
"""

import io
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Import the real circle packing evaluator
sys.path.insert(0, "/Users/sahan/repos/tetris_evolve/src")
from tetris_evolve.evaluation.circle_packing import CirclePackingEvaluator


# ============================================================================
# Cost Tracker
# ============================================================================


class BudgetExceededError(Exception):
    pass


@dataclass
class ModelPricing:
    input_cost_per_token: float
    output_cost_per_token: float

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_token
            + output_tokens * self.output_cost_per_token
        )


class CostTracker:
    def __init__(
        self, max_budget: float, root_pricing: ModelPricing, child_pricing: ModelPricing
    ):
        self.max_budget = max_budget
        self.root_pricing = root_pricing
        self.child_pricing = child_pricing
        self.total_cost = 0.0
        self.usage_log = []

    def record_usage(self, input_tokens: int, output_tokens: int, llm_type: str):
        pricing = self.root_pricing if llm_type == "root" else self.child_pricing
        cost = pricing.compute_cost(input_tokens, output_tokens)
        self.total_cost += cost
        self.usage_log.append(
            {"type": llm_type, "input": input_tokens, "output": output_tokens, "cost": cost}
        )

    def raise_if_over_budget(self):
        if self.total_cost >= self.max_budget:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.4f} >= ${self.max_budget:.2f}"
            )

    def get_remaining_budget(self) -> float:
        return max(0, self.max_budget - self.total_cost)


# ============================================================================
# Mock LLM Client with Realistic Circle Packing Code Generation
# ============================================================================


class MockCirclePackingLLM:
    """
    Mock LLM that generates different circle packing strategies based on prompts.
    """

    def __init__(self, cost_tracker: CostTracker, llm_type: str):
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self.call_count = 0

    def generate(self, prompt: str) -> tuple:
        """Generate mock response with circle packing code."""
        self.call_count += 1
        self.cost_tracker.raise_if_over_budget()

        input_tokens = len(prompt) // 4
        output_tokens = 800 + (self.call_count * 50)
        self.cost_tracker.record_usage(input_tokens, output_tokens, self.llm_type)

        if self.llm_type == "child":
            response = self._generate_packing_code(prompt)
        else:
            response = "Analyzing results..."

        return response, input_tokens, output_tokens

    def _generate_packing_code(self, prompt: str) -> str:
        """Generate different packing strategies based on prompt."""
        prompt_lower = prompt.lower()

        if "grid" in prompt_lower:
            return self._grid_strategy()
        elif "hexagonal" in prompt_lower or "hex" in prompt_lower:
            return self._hexagonal_strategy()
        elif "greedy" in prompt_lower:
            return self._greedy_strategy()
        elif "corner" in prompt_lower:
            return self._corner_strategy()
        elif "concentric" in prompt_lower or "ring" in prompt_lower:
            return self._concentric_strategy()
        elif "optimize" in prompt_lower or "improve" in prompt_lower:
            return self._optimized_strategy()
        else:
            # Default: random variation
            return self._random_variation_strategy()

    def _grid_strategy(self) -> str:
        return '''I'll implement a grid-based packing strategy.

```python
import numpy as np

def construct_packing():
    """Grid-based circle packing with optimized spacing."""
    n = 26
    centers = []

    # Use a 6x5 grid area = 30, take 26
    rows, cols = 6, 5
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                centers.append([x, y])
                idx += 1

    # Take first 26
    centers = np.array(centers[:n])

    # Compute radii
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y, spacing_x/2.1, spacing_y/2.1)

    # Avoid overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.98
                radii[i] *= scale
                radii[j] *= scale

    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

This uses a regular grid pattern for predictable, non-overlapping placement.'''

    def _hexagonal_strategy(self) -> str:
        return '''I'll implement a hexagonal close-packing strategy.

```python
import numpy as np

def construct_packing():
    """Hexagonal close-packing inspired layout."""
    n = 26
    centers = []

    # Hexagonal grid parameters
    r = 0.08  # Base radius estimate
    dx = 2 * r * 1.1
    dy = r * np.sqrt(3) * 1.1

    rows = 6
    for row in range(rows):
        y = 0.1 + row * dy
        if y > 0.9:
            break
        offset = (dx / 2) if row % 2 == 1 else 0
        cols = int((0.9 - 0.1) / dx) + 1
        for col in range(cols):
            x = 0.1 + offset + col * dx
            if x > 0.9:
                break
            if len(centers) < n:
                centers.append([x, y])

    # Pad if needed
    while len(centers) < n:
        centers.append([0.5 + np.random.uniform(-0.1, 0.1),
                       0.5 + np.random.uniform(-0.1, 0.1)])

    centers = np.array(centers[:n])

    # Compute radii
    radii = np.ones(n) * r
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1-x, 1-y)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.98
                radii[i] *= scale
                radii[j] *= scale

    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

Hexagonal packing achieves higher density than square grids.'''

    def _greedy_strategy(self) -> str:
        return '''I'll implement a greedy placement strategy.

```python
import numpy as np

def construct_packing():
    """Greedy placement - add circles one by one to maximize radius."""
    n = 26
    centers = []
    radii = []

    # Start with largest possible circle in center
    centers.append([0.5, 0.5])
    radii.append(0.5)

    # Greedily add more circles
    for _ in range(n - 1):
        best_pos = None
        best_r = 0

        # Try many random positions
        for _ in range(500):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)

            # Max radius at this position
            r = min(x, y, 1-x, 1-y)
            for cx, cy, cr in zip([c[0] for c in centers],
                                  [c[1] for c in centers], radii):
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                r = min(r, dist - cr)

            if r > best_r and r > 0.01:
                best_r = r
                best_pos = [x, y]

        if best_pos:
            centers.append(best_pos)
            radii.append(best_r * 0.98)
        else:
            # Fallback
            centers.append([np.random.uniform(0.1, 0.9),
                           np.random.uniform(0.1, 0.9)])
            radii.append(0.01)

    centers = np.array(centers)
    radii = np.array(radii)
    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

Greedy approach finds locally optimal positions for each circle.'''

    def _corner_strategy(self) -> str:
        return '''I'll pack from corners inward.

```python
import numpy as np

def construct_packing():
    """Corner-first strategy with varying sizes."""
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Four corners with large circles
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for i, (cx, cy) in enumerate(corners):
        centers[i] = [cx, cy]
        radii[i] = 0.14

    # Edges
    edges = [[0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
    for i, (cx, cy) in enumerate(edges):
        centers[i+4] = [cx, cy]
        radii[i+4] = 0.09

    # Fill remaining with smaller circles
    remaining = n - 8
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        r_pos = 0.3
        cx = 0.5 + r_pos * np.cos(angle)
        cy = 0.5 + r_pos * np.sin(angle)
        centers[i+8] = [cx, cy]
        radii[i+8] = 0.05

    # Adjust for overlaps
    for _ in range(3):
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j]) * 0.95
                    radii[i] *= scale
                    radii[j] *= scale

    # Boundary check
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1-x, 1-y)

    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

Uses corners and edges for stable large circles.'''

    def _concentric_strategy(self) -> str:
        return '''I'll use concentric rings.

```python
import numpy as np

def construct_packing():
    """Concentric rings around center."""
    n = 26
    centers = np.zeros((n, 2))

    # Center
    centers[0] = [0.5, 0.5]

    # Ring 1: 6 circles
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i+1] = [0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)]

    # Ring 2: 10 circles
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        centers[i+7] = [0.5 + 0.38 * np.cos(angle), 0.5 + 0.38 * np.sin(angle)]

    # Ring 3: 9 circles
    for i in range(9):
        angle = 2 * np.pi * i / 9
        r = 0.42
        cx = 0.5 + r * np.cos(angle)
        cy = 0.5 + r * np.sin(angle)
        # Clamp to square
        cx = np.clip(cx, 0.08, 0.92)
        cy = np.clip(cy, 0.08, 0.92)
        centers[i+17] = [cx, cy]

    # Compute radii
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.98
                radii[i] *= scale
                radii[j] *= scale

    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

Rings provide good coverage of the square.'''

    def _optimized_strategy(self) -> str:
        return '''I'll use scipy optimization to improve placement.

```python
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Optimization-based packing using scipy."""
    n = 26

    # Initial guess: grid pattern
    centers_init = []
    for i in range(6):
        for j in range(5):
            if len(centers_init) < n:
                centers_init.append([0.1 + j * 0.2, 0.1 + i * 0.16])
    centers_init = np.array(centers_init[:n])

    def objective(flat_centers):
        centers = flat_centers.reshape(n, 2)
        radii = np.ones(n) * 0.5

        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            radii[i] = min(radii[i], max(0.001, x), max(0.001, y),
                          max(0.001, 1-x), max(0.001, 1-y))

        # Pairwise constraints
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 0:
                    max_sum = dist
                    if radii[i] + radii[j] > max_sum:
                        scale = max_sum / (radii[i] + radii[j]) * 0.98
                        radii[i] *= scale
                        radii[j] *= scale

        return -np.sum(radii)  # Negative for minimization

    # Optimize
    bounds = [(0.02, 0.98)] * (n * 2)
    result = minimize(objective, centers_init.flatten(), method='L-BFGS-B',
                     bounds=bounds, options={'maxiter': 200})

    centers = result.x.reshape(n, 2)

    # Final radii computation
    radii = np.ones(n) * 0.5
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1-x, 1-y)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 0 and radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.98
                radii[i] *= scale
                radii[j] *= scale

    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

Uses numerical optimization to find better arrangements.'''

    def _random_variation_strategy(self) -> str:
        """Generate a random but valid packing."""
        return '''I'll try an adaptive approach.

```python
import numpy as np

def construct_packing():
    """Adaptive placement with local optimization."""
    n = 26
    np.random.seed(42)

    # Start with perturbed grid
    centers = []
    base_spacing = 1.0 / 6
    for i in range(6):
        for j in range(6):
            if len(centers) < n:
                x = base_spacing * (j + 0.5) + np.random.uniform(-0.03, 0.03)
                y = base_spacing * (i + 0.5) + np.random.uniform(-0.03, 0.03)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                centers.append([x, y])

    centers = np.array(centers[:n])

    # Local optimization: push circles apart
    for iteration in range(50):
        for i in range(n):
            force = np.zeros(2)
            for j in range(n):
                if i != j:
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < 0.2 and dist > 0:
                        force += diff / dist * 0.01

            centers[i] += force
            centers[i] = np.clip(centers[i], 0.05, 0.95)

    # Compute radii
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.98
                radii[i] *= scale
                radii[j] *= scale

    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
```

Combines grid initialization with force-based optimization.'''


# ============================================================================
# Evolution API
# ============================================================================


@dataclass
class TrialResult:
    trial_id: str
    code: str
    metrics: Dict[str, float]
    reasoning: str
    success: bool
    error: Optional[str] = None


class EvolutionAPI:
    def __init__(
        self,
        cost_tracker: CostTracker,
        child_llm: MockCirclePackingLLM,
        evaluator: CirclePackingEvaluator,
    ):
        self.cost_tracker = cost_tracker
        self.child_llm = child_llm
        self.evaluator = evaluator

        self.current_generation = 0
        self.generations: List[List[TrialResult]] = [[]]
        self.termination_result = None

    def spawn_child_llm(self, prompt: str, parent_id: Optional[str] = None) -> Dict:
        """Spawn a child LLM to generate circle packing code."""
        self.cost_tracker.raise_if_over_budget()

        response, _, _ = self.child_llm.generate(prompt)
        code = self._extract_code(response)
        reasoning = self._extract_reasoning(response)

        if code:
            metrics = self.evaluator.evaluate(code)
            success = metrics.get("valid", False)
            error = metrics.get("error")
        else:
            metrics = {"valid": False, "sum_radii": 0.0, "combined_score": 0.0}
            success = False
            error = "No code extracted"

        trial_id = f"gen{self.current_generation}_trial{len(self.generations[-1])}"
        trial = TrialResult(
            trial_id=trial_id,
            code=code or "",
            metrics=metrics,
            reasoning=reasoning,
            success=success,
            error=error,
        )

        self.generations[-1].append(trial)
        return asdict(trial)

    def evaluate_program(self, code: str) -> Dict[str, Any]:
        return self.evaluator.evaluate(code)

    def advance_generation(self, selected_trial_ids: List[str], reasoning: str) -> int:
        self.current_generation += 1
        self.generations.append([])
        return self.current_generation

    def terminate_evolution(self, reason: str) -> Dict:
        all_trials = [t for gen in self.generations for t in gen]
        successful = [t for t in all_trials if t.success]

        if successful:
            best = max(successful, key=lambda t: t.metrics.get("sum_radii", 0))
        else:
            best = None

        self.termination_result = {
            "reason": reason,
            "total_generations": self.current_generation + 1,
            "total_trials": len(all_trials),
            "successful_trials": len(successful),
            "best_trial": asdict(best) if best else None,
            "total_cost": self.cost_tracker.total_cost,
        }
        return self.termination_result

    def get_generation_history(self) -> List[Dict]:
        return [
            {"generation": i, "num_trials": len(gen), "trials": [asdict(t) for t in gen]}
            for i, gen in enumerate(self.generations)
        ]

    def get_best_trials(self, n: int = 5) -> List[Dict]:
        all_trials = [t for gen in self.generations for t in gen if t.success]
        sorted_trials = sorted(
            all_trials, key=lambda t: t.metrics.get("sum_radii", 0), reverse=True
        )
        return [asdict(t) for t in sorted_trials[:n]]

    def get_cost_remaining(self) -> float:
        return self.cost_tracker.get_remaining_budget()

    def _extract_code(self, response: str) -> Optional[str]:
        import re
        pattern = r"```python\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_reasoning(self, response: str) -> str:
        import re
        pattern = r"```python.*?```"
        parts = re.split(pattern, response, flags=re.DOTALL)
        return " ".join(p.strip() for p in parts if p.strip())[:500]


# ============================================================================
# REPL Environment
# ============================================================================


class REPLEnvironment:
    def __init__(self, evolution_api: EvolutionAPI):
        self.evolution_api = evolution_api
        self.globals = self._create_globals()
        self.locals: Dict[str, Any] = {}

    def _create_globals(self) -> Dict:
        safe_builtins = {
            "print": print, "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "range": range, "enumerate": enumerate,
            "max": max, "min": min, "sorted": sorted, "sum": sum,
            "__import__": __import__, "Exception": Exception,
        }
        return {
            "__builtins__": safe_builtins,
            "spawn_child_llm": self.evolution_api.spawn_child_llm,
            "evaluate_program": self.evolution_api.evaluate_program,
            "advance_generation": self.evolution_api.advance_generation,
            "terminate_evolution": self.evolution_api.terminate_evolution,
            "get_generation_history": self.evolution_api.get_generation_history,
            "get_best_trials": self.evolution_api.get_best_trials,
            "get_cost_remaining": self.evolution_api.get_cost_remaining,
        }

    def execute(self, code: str) -> Dict[str, Any]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            sys.stdout, sys.stderr = stdout, stderr
            combined = {**self.globals, **self.locals}
            exec(code, combined, combined)
            for k, v in combined.items():
                if k not in self.globals and not k.startswith("_"):
                    self.locals[k] = v
            return {"success": True, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        except Exception as e:
            return {"success": False, "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(), "error": str(e)}
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


# ============================================================================
# Tests
# ============================================================================


def test_circle_packing_evolution():
    """Test the complete evolution pipeline for circle packing."""
    print("Testing Circle Packing Evolution Pipeline...")
    print("=" * 60)

    # Setup
    cost_tracker = CostTracker(
        max_budget=5.0,
        root_pricing=ModelPricing(3e-6, 15e-6),
        child_pricing=ModelPricing(0.25e-6, 1.25e-6),
    )

    child_llm = MockCirclePackingLLM(cost_tracker, "child")
    evaluator = CirclePackingEvaluator(timeout_seconds=10)
    evolution_api = EvolutionAPI(cost_tracker, child_llm, evaluator)
    repl = REPLEnvironment(evolution_api)

    # Generation 0: Explore different strategies
    print("\nGeneration 0: Exploring diverse strategies...")
    result = repl.execute('''
strategies = [
    "grid-based packing",
    "hexagonal packing",
    "greedy placement",
    "corner-first strategy",
    "concentric rings",
]

results = []
for strategy in strategies:
    trial = spawn_child_llm(f"Implement a {strategy} for 26 circles")
    results.append(trial)
    status = "VALID" if trial["success"] else "INVALID"
    score = trial["metrics"].get("sum_radii", 0)
    print(f"  {trial['trial_id']}: {status}, sum_radii={score:.4f}")
''')
    assert result["success"], f"Gen 0 failed: {result.get('error')}"
    print(result["stdout"])

    # Analyze and select
    print("\nAnalyzing Generation 0...")
    result = repl.execute('''
best = get_best_trials(3)
print(f"Top 3 strategies:")
for t in best:
    print(f"  {t['trial_id']}: sum_radii={t['metrics']['sum_radii']:.4f}")

selected = [t['trial_id'] for t in best]
new_gen = advance_generation(selected, "Selected top 3 for optimization")
print(f"\\nAdvanced to generation {new_gen}")
''')
    assert result["success"]
    print(result["stdout"])

    # Generation 1: Optimization
    print("\nGeneration 1: Optimization strategies...")
    result = repl.execute('''
# Try optimization-based approaches
opt1 = spawn_child_llm("Use scipy optimize to improve circle placement")
opt2 = spawn_child_llm("Implement adaptive placement with local optimization")

print(f"Optimization results:")
print(f"  {opt1['trial_id']}: sum_radii={opt1['metrics'].get('sum_radii', 0):.4f}")
print(f"  {opt2['trial_id']}: sum_radii={opt2['metrics'].get('sum_radii', 0):.4f}")
''')
    assert result["success"]
    print(result["stdout"])

    # Terminate
    print("\nTerminating evolution...")
    result = repl.execute('''
remaining = get_cost_remaining()
print(f"Budget remaining: ${remaining:.4f}")

final = terminate_evolution("Completed 2 generations of exploration")
print(f"\\nFinal Results:")
print(f"  Total trials: {final['total_trials']}")
print(f"  Successful: {final['successful_trials']}")
if final['best_trial']:
    best = final['best_trial']
    print(f"  Best sum_radii: {best['metrics']['sum_radii']:.4f}")
    print(f"  Best trial: {best['trial_id']}")
    target_ratio = best['metrics'].get('target_ratio', 0)
    print(f"  Target ratio (vs 2.635): {target_ratio:.4f}")
''')
    assert result["success"]
    print(result["stdout"])

    # Verify
    assert evolution_api.current_generation == 1
    assert len(evolution_api.generations[0]) == 5  # 5 initial strategies
    assert len(evolution_api.generations[1]) == 2  # 2 optimization trials
    assert evolution_api.termination_result is not None

    print("\n" + "=" * 60)
    print("Circle Packing Evolution Test PASSED!")


def test_real_evaluator_integration():
    """Test that the real evaluator works with generated code."""
    print("\n\nTesting Real Evaluator Integration...")
    print("=" * 60)

    evaluator = CirclePackingEvaluator(timeout_seconds=10)
    mock_llm = MockCirclePackingLLM(
        CostTracker(10.0, ModelPricing(1e-6, 1e-6), ModelPricing(1e-6, 1e-6)),
        "child"
    )

    strategies = ["grid", "hexagonal", "greedy", "concentric", "optimize"]

    print("\nEvaluating different strategies:")
    results = []
    for strategy in strategies:
        response, _, _ = mock_llm.generate(f"{strategy} packing")
        import re
        match = re.search(r"```python\s*\n(.*?)\n```", response, re.DOTALL)
        if match:
            code = match.group(1)
            metrics = evaluator.evaluate(code)
            results.append((strategy, metrics))
            status = "VALID" if metrics["valid"] else "INVALID"
            print(f"  {strategy:12s}: {status}, sum={metrics['sum_radii']:.4f}, ratio={metrics['target_ratio']:.4f}")

    # Find best
    valid_results = [(s, m) for s, m in results if m["valid"]]
    if valid_results:
        best_strategy, best_metrics = max(valid_results, key=lambda x: x[1]["sum_radii"])
        print(f"\nBest strategy: {best_strategy}")
        print(f"  Sum of radii: {best_metrics['sum_radii']:.4f}")
        print(f"  Target ratio: {best_metrics['target_ratio']:.4f} (target: 2.635)")

    print("\n" + "=" * 60)
    print("Real Evaluator Integration Test PASSED!")


if __name__ == "__main__":
    test_circle_packing_evolution()
    test_real_evaluator_integration()
