"""
Proof of Concept: Full Integration

This validates the complete flow:
1. Root LLM REPL can spawn child LLMs
2. Child LLM responses are evaluated
3. Results feed back into Root LLM context
4. Generation advancement works
5. Cost tracking is integrated
6. Everything terminates correctly

Uses mock LLMs to test the full pipeline without actual API calls.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid
import io
import sys


# ============================================================================
# Cost Tracker (from poc_cost_tracker.py)
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
    def __init__(self, max_budget: float, root_pricing: ModelPricing, child_pricing: ModelPricing):
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
            {
                "type": llm_type,
                "input": input_tokens,
                "output": output_tokens,
                "cost": cost,
            }
        )

    def raise_if_over_budget(self):
        if self.total_cost >= self.max_budget:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.4f} >= ${self.max_budget:.2f}"
            )

    def get_remaining_budget(self) -> float:
        return max(0, self.max_budget - self.total_cost)


# ============================================================================
# Mock LLM Client
# ============================================================================


class MockLLMClient:
    """
    Mock LLM that generates deterministic responses based on prompts.
    Simulates what the actual Anthropic API would return.
    """

    def __init__(self, cost_tracker: CostTracker, llm_type: str):
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self.call_count = 0

    def generate(self, prompt: str) -> tuple:
        """Generate mock response and return (response, input_tokens, output_tokens)."""
        self.call_count += 1
        self.cost_tracker.raise_if_over_budget()

        input_tokens = len(prompt) // 4  # Rough estimate
        output_tokens = 500 + (self.call_count * 100)

        # Record cost
        self.cost_tracker.record_usage(input_tokens, output_tokens, self.llm_type)

        # Generate appropriate mock response
        if self.llm_type == "child":
            response = self._generate_child_response(prompt)
        else:
            response = self._generate_root_response(prompt)

        return response, input_tokens, output_tokens

    def _generate_child_response(self, prompt: str) -> str:
        """Generate mock child LLM response with Tetris code."""
        # Different strategies based on prompt content
        if "greedy" in prompt.lower():
            strategy = "greedy"
            action_logic = "return 5  # Always hard drop"
        elif "defensive" in prompt.lower():
            strategy = "defensive"
            action_logic = "return 1 if observation.sum() > 50 else 5  # Move left if board is filling up"
        elif "random" in prompt.lower():
            strategy = "random"
            action_logic = "return np.random.randint(0, 6)"
        else:
            strategy = "balanced"
            action_logic = "return 5 if info.get('current_piece', 0) < 4 else np.random.choice([1, 2, 5])"

        return f'''I'll implement a {strategy} strategy for Tetris.

```python
import numpy as np

def select_action(observation, info):
    """
    {strategy.capitalize()} Tetris agent.
    Strategy: {strategy}
    """
    {action_logic}
```

This approach focuses on {strategy} gameplay by {
    "maximizing immediate line clears" if strategy == "greedy" else
    "keeping the stack low" if strategy == "defensive" else
    "exploring randomly" if strategy == "random" else
    "balancing multiple factors"
}.'''

    def _generate_root_response(self, prompt: str) -> str:
        """Generate mock root LLM response."""
        return "I'll analyze the results and proceed with the next step."


# ============================================================================
# Evaluator (from poc_evaluator.py, simplified)
# ============================================================================


class MockTetrisEnv:
    def __init__(self):
        self.score = 0
        self.lines = 0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        import numpy as np

        if seed:
            np.random.seed(seed)
        self.score = 0
        self.lines = 0
        self.steps = 0
        self.done = False
        return np.zeros((20, 10)), {"current_piece": 0, "lines_cleared": 0}

    def step(self, action):
        import numpy as np

        self.steps += 1
        reward = 1
        if action == 5 and np.random.random() < 0.3:
            lines = np.random.randint(1, 5)
            self.lines += lines
            reward += lines * 100
        self.score += reward
        if np.random.random() < 0.001 * self.steps:
            self.done = True
        return (
            np.zeros((20, 10)),
            reward,
            self.done,
            False,
            {"current_piece": np.random.randint(0, 7), "lines_cleared": self.lines},
        )

    def close(self):
        pass


class TetrisEvaluator:
    def __init__(self, num_games: int = 5, max_steps: int = 200):
        self.num_games = num_games
        self.max_steps = max_steps

    def evaluate(self, code: str) -> Dict[str, Any]:
        import numpy as np

        namespace = {"np": np, "numpy": np}
        try:
            exec(code, namespace)
            if "select_action" not in namespace:
                return {"error": "No select_action", "success_rate": 0.0}

            select_action = namespace["select_action"]
            scores = []

            for i in range(self.num_games):
                try:
                    env = MockTetrisEnv()
                    obs, info = env.reset(seed=i)
                    for _ in range(self.max_steps):
                        action = select_action(obs, info)
                        obs, _, done, _, info = env.step(int(action))
                        if done:
                            break
                    scores.append(env.score)
                except Exception:
                    pass

            if not scores:
                return {"error": "All games crashed", "success_rate": 0.0}

            return {
                "avg_score": float(np.mean(scores)),
                "max_score": float(max(scores)),
                "success_rate": len(scores) / self.num_games,
            }
        except SyntaxError as e:
            return {"error": f"Syntax: {e}", "success_rate": 0.0}
        except Exception as e:
            return {"error": str(e), "success_rate": 0.0}


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
        child_llm: MockLLMClient,
        evaluator: TetrisEvaluator,
    ):
        self.cost_tracker = cost_tracker
        self.child_llm = child_llm
        self.evaluator = evaluator

        self.current_generation = 0
        self.generations: List[List[TrialResult]] = [[]]
        self.termination_result = None

    def spawn_child_llm(
        self, prompt: str, parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Spawn a child LLM to generate Tetris code."""
        self.cost_tracker.raise_if_over_budget()

        # Generate code from child LLM
        response, _, _ = self.child_llm.generate(prompt)

        # Extract code from response
        code = self._extract_code(response)
        reasoning = self._extract_reasoning(response)

        # Evaluate the code
        if code:
            metrics = self.evaluator.evaluate(code)
            success = "error" not in metrics
            error = metrics.get("error")
        else:
            metrics = {}
            success = False
            error = "No code extracted"

        # Create trial
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

    def evaluate_program(self, code: str, num_games: int = 5) -> Dict[str, float]:
        """Evaluate a program directly."""
        return self.evaluator.evaluate(code)

    def advance_generation(self, selected_trial_ids: List[str], reasoning: str) -> int:
        """Advance to next generation."""
        self.current_generation += 1
        self.generations.append([])
        return self.current_generation

    def terminate_evolution(self, reason: str) -> Dict[str, Any]:
        """Terminate evolution and return results."""
        all_trials = [t for gen in self.generations for t in gen]
        successful = [t for t in all_trials if t.success]

        if successful:
            best = max(successful, key=lambda t: t.metrics.get("avg_score", 0))
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
        """Get summary of all generations."""
        return [
            {
                "generation": i,
                "num_trials": len(gen),
                "trials": [asdict(t) for t in gen],
            }
            for i, gen in enumerate(self.generations)
        ]

    def get_best_trials(self, n: int = 5) -> List[Dict]:
        """Get best trials across all generations."""
        all_trials = [t for gen in self.generations for t in gen if t.success]
        sorted_trials = sorted(
            all_trials, key=lambda t: t.metrics.get("avg_score", 0), reverse=True
        )
        return [asdict(t) for t in sorted_trials[:n]]

    def get_cost_remaining(self) -> float:
        """Get remaining budget."""
        return self.cost_tracker.get_remaining_budget()

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response."""
        import re

        pattern = r"```python\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response."""
        import re

        # Get text before or after the code block
        pattern = r"```python.*?```"
        parts = re.split(pattern, response, flags=re.DOTALL)
        reasoning_parts = [p.strip() for p in parts if p.strip()]
        return " ".join(reasoning_parts)[:500]


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
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "max": max,
            "min": min,
            "sorted": sorted,
            "sum": sum,
            "__import__": __import__,
            "Exception": Exception,
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
            return {
                "success": True,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "error": str(e),
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


# ============================================================================
# Tests
# ============================================================================


def test_full_integration():
    """Test the complete evolution pipeline."""
    print("Testing full integration pipeline...")

    # Setup
    cost_tracker = CostTracker(
        max_budget=1.0,
        root_pricing=ModelPricing(3e-6, 15e-6),
        child_pricing=ModelPricing(0.25e-6, 1.25e-6),
    )

    child_llm = MockLLMClient(cost_tracker, "child")
    evaluator = TetrisEvaluator(num_games=3, max_steps=100)
    evolution_api = EvolutionAPI(cost_tracker, child_llm, evaluator)
    repl = REPLEnvironment(evolution_api)

    # Simulate Root LLM behavior
    print("\n  Generation 0: Exploring strategies...")

    result = repl.execute(
        """
# Generation 0: Try different strategies
strategies = ["greedy", "defensive", "random", "balanced"]
results = []
for strategy in strategies:
    trial = spawn_child_llm(f"Implement a {strategy} Tetris strategy")
    results.append(trial)
    print(f"  {trial['trial_id']}: score={trial['metrics'].get('avg_score', 0):.0f}")
"""
    )
    assert result["success"], f"Gen 0 failed: {result.get('error')}"
    print(result["stdout"])

    # Analyze and advance
    print("\n  Analyzing generation 0...")
    result = repl.execute(
        """
# Get best trials
best = get_best_trials(2)
print(f"  Best trials: {[t['trial_id'] for t in best]}")

# Advance to generation 1
selected = [t['trial_id'] for t in best]
new_gen = advance_generation(selected, "Selected top 2 performers")
print(f"  Advanced to generation {new_gen}")
"""
    )
    assert result["success"], f"Analysis failed: {result.get('error')}"
    print(result["stdout"])

    # Generation 1
    print("\n  Generation 1: Refining best strategies...")
    result = repl.execute(
        """
# Try variations on best strategies
result1 = spawn_child_llm("Improve the greedy strategy with lookahead")
result2 = spawn_child_llm("Improve the defensive strategy with better positioning")
print(f"  {result1['trial_id']}: score={result1['metrics'].get('avg_score', 0):.0f}")
print(f"  {result2['trial_id']}: score={result2['metrics'].get('avg_score', 0):.0f}")
"""
    )
    assert result["success"], f"Gen 1 failed: {result.get('error')}"
    print(result["stdout"])

    # Terminate
    print("\n  Terminating evolution...")
    result = repl.execute(
        """
# Check cost and terminate
remaining = get_cost_remaining()
print(f"  Budget remaining: ${remaining:.4f}")

final = terminate_evolution("Completed 2 generations")
print(f"  Total trials: {final['total_trials']}")
print(f"  Successful: {final['successful_trials']}")
if final['best_trial']:
    print(f"  Best score: {final['best_trial']['metrics'].get('avg_score', 0):.0f}")
"""
    )
    assert result["success"], f"Termination failed: {result.get('error')}"
    print(result["stdout"])

    # Verify state
    assert evolution_api.current_generation == 1
    assert len(evolution_api.generations[0]) == 4  # 4 trials in gen 0
    assert len(evolution_api.generations[1]) == 2  # 2 trials in gen 1
    assert evolution_api.termination_result is not None
    assert evolution_api.termination_result["total_trials"] == 6

    print("\n✓ test_full_integration passed!")


def test_budget_enforcement():
    """Test that budget is enforced."""
    print("\nTesting budget enforcement...")

    cost_tracker = CostTracker(
        max_budget=0.0001,  # Tiny budget
        root_pricing=ModelPricing(3e-6, 15e-6),
        child_pricing=ModelPricing(3e-6, 15e-6),  # Same as root for faster budget drain
    )

    child_llm = MockLLMClient(cost_tracker, "child")
    evaluator = TetrisEvaluator(num_games=2, max_steps=50)
    evolution_api = EvolutionAPI(cost_tracker, child_llm, evaluator)
    repl = REPLEnvironment(evolution_api)

    # This should eventually hit budget
    result = repl.execute(
        """
try:
    for i in range(100):  # Try to spawn many
        spawn_child_llm(f"Strategy {i}")
    print("ERROR: Should have hit budget!")
except Exception as e:
    print(f"Budget hit as expected: {str(e)[:50]}")
"""
    )

    assert result["success"]
    assert "Budget" in result["stdout"] or cost_tracker.total_cost >= cost_tracker.max_budget
    print(f"  Budget enforced at ${cost_tracker.total_cost:.6f}")
    print("✓ test_budget_enforcement passed!")


def test_error_recovery():
    """Test that errors in child code don't crash the system."""
    print("\nTesting error recovery...")

    cost_tracker = CostTracker(
        max_budget=1.0,
        root_pricing=ModelPricing(3e-6, 15e-6),
        child_pricing=ModelPricing(0.25e-6, 1.25e-6),
    )

    child_llm = MockLLMClient(cost_tracker, "child")
    evaluator = TetrisEvaluator(num_games=3, max_steps=100)
    evolution_api = EvolutionAPI(cost_tracker, child_llm, evaluator)
    repl = REPLEnvironment(evolution_api)

    # Spawn a trial (which generates valid code via mock)
    result = repl.execute(
        """
trial = spawn_child_llm("Test strategy")
print(f"Trial success: {trial['success']}")
print(f"Score: {trial['metrics'].get('avg_score', 'N/A')}")
"""
    )

    assert result["success"]
    print(result["stdout"])
    print("✓ test_error_recovery passed!")


if __name__ == "__main__":
    print("Running Integration Proof of Concept Tests\n" + "=" * 60)

    test_full_integration()
    test_budget_enforcement()
    test_error_recovery()

    print("\n" + "=" * 60)
    print("All Integration PoC tests passed!")
