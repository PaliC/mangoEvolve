"""
Proof of Concept: REPL Environment

This validates that we can:
1. Execute code in an isolated namespace
2. Inject custom functions into the namespace
3. Persist state across executions
4. Capture stdout/stderr properly
"""

import sys
import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class MockEvolutionAPI:
    """Mock evolution API to test injection into REPL."""

    def __init__(self):
        self.spawn_count = 0
        self.trials = []

    def spawn_child_llm(self, prompt: str, parent_id: Optional[str] = None) -> Dict:
        """Mock implementation of spawn_child_llm."""
        self.spawn_count += 1
        trial_id = f"trial_{self.spawn_count}"
        result = {
            "trial_id": trial_id,
            "code": f"# Mock code for: {prompt[:50]}...",
            "metrics": {"avg_score": 100.0 * self.spawn_count, "avg_lines": 5.0},
            "reasoning": f"Mock reasoning for trial {trial_id}",
            "success": True,
            "error": None,
        }
        self.trials.append(result)
        return result

    def evaluate_program(self, code: str, num_games: int = 10) -> Dict[str, float]:
        """Mock implementation of evaluate_program."""
        return {
            "avg_score": 150.0,
            "avg_lines_cleared": 10.0,
            "success_rate": 1.0,
        }

    def get_best_trials(self, n: int = 5) -> list:
        """Get best trials."""
        return sorted(self.trials, key=lambda t: t["metrics"]["avg_score"], reverse=True)[:n]


class REPLEnvironment:
    """REPL environment for Root LLM code execution."""

    def __init__(self, evolution_api: Optional[MockEvolutionAPI] = None):
        self.evolution_api = evolution_api or MockEvolutionAPI()
        self.globals = self._create_globals()
        self.locals: Dict[str, Any] = {}

    def _create_globals(self) -> Dict[str, Any]:
        """Create safe globals with injected API functions."""
        safe_builtins = {
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
            "type": type,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "__import__": __import__,
        }

        return {
            "__builtins__": safe_builtins,
            # Inject evolution API
            "spawn_child_llm": self.evolution_api.spawn_child_llm,
            "evaluate_program": self.evolution_api.evaluate_program,
            "get_best_trials": self.evolution_api.get_best_trials,
        }

    def execute(self, code: str) -> REPLResult:
        """Execute Python code in the REPL environment."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Combine globals and locals for execution
            combined = {**self.globals, **self.locals}

            # Execute the code
            exec(code, combined, combined)

            # Update locals with any new variables
            for key, value in combined.items():
                if key not in self.globals and not key.startswith("_"):
                    self.locals[key] = value

            return REPLResult(
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                locals=self.locals.copy(),
                success=True,
            )

        except Exception as e:
            return REPLResult(
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                locals=self.locals.copy(),
                success=False,
                error=str(e),
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def test_basic_execution():
    """Test basic code execution."""
    repl = REPLEnvironment()

    result = repl.execute("x = 5\nprint(f'x = {x}')")

    assert result.success, f"Execution failed: {result.error}"
    assert "x = 5" in result.stdout
    assert "x" in result.locals
    assert result.locals["x"] == 5
    print("✓ test_basic_execution passed")


def test_state_persistence():
    """Test that state persists across executions."""
    repl = REPLEnvironment()

    repl.execute("counter = 0")
    repl.execute("counter += 1")
    result = repl.execute("print(f'counter = {counter}')")

    assert result.success
    assert result.locals["counter"] == 1
    assert "counter = 1" in result.stdout
    print("✓ test_state_persistence passed")


def test_api_injection():
    """Test that evolution API functions are available."""
    api = MockEvolutionAPI()
    repl = REPLEnvironment(evolution_api=api)

    result = repl.execute("""
result = spawn_child_llm("Try a greedy approach")
print(f"Trial ID: {result['trial_id']}")
print(f"Score: {result['metrics']['avg_score']}")
""")

    assert result.success, f"Execution failed: {result.error}"
    assert api.spawn_count == 1
    assert "Trial ID: trial_1" in result.stdout
    print("✓ test_api_injection passed")


def test_custom_functions():
    """Test that Root LLM can define custom helper functions."""
    api = MockEvolutionAPI()
    repl = REPLEnvironment(evolution_api=api)

    # Simulate Root LLM defining a helper function
    repl.execute("""
def run_experiments(strategies):
    results = []
    for strategy in strategies:
        trial = spawn_child_llm(strategy)
        results.append(trial)
    return results
""")

    # Now use the helper function
    result = repl.execute("""
strategies = ["greedy", "defensive", "balanced"]
experiments = run_experiments(strategies)
print(f"Ran {len(experiments)} experiments")
best = max(experiments, key=lambda x: x['metrics']['avg_score'])
print(f"Best trial: {best['trial_id']}")
""")

    assert result.success, f"Execution failed: {result.error}"
    assert api.spawn_count == 3
    assert "Ran 3 experiments" in result.stdout
    print("✓ test_custom_functions passed")


def test_error_handling():
    """Test that errors are captured properly."""
    repl = REPLEnvironment()

    result = repl.execute("x = 1 / 0")

    assert not result.success
    assert "ZeroDivisionError" in result.error or "division by zero" in result.error
    print("✓ test_error_handling passed")


def test_import_capability():
    """Test that imports work."""
    repl = REPLEnvironment()

    result = repl.execute("""
import json
data = {'key': 'value'}
print(json.dumps(data))
""")

    assert result.success, f"Execution failed: {result.error}"
    assert '{"key": "value"}' in result.stdout
    print("✓ test_import_capability passed")


def test_complex_workflow():
    """Test a more complex workflow similar to actual usage."""
    api = MockEvolutionAPI()
    repl = REPLEnvironment(evolution_api=api)

    # Simulate a generation of exploration
    code = """
# Generation 0: Explore diverse strategies
strategies = [
    "Maximize immediate line clears using a greedy heuristic",
    "Minimize stack height to avoid game over",
    "Balance between line clears and stack safety",
    "Look ahead to the next piece for better placement",
]

results = []
for strategy in strategies:
    trial = spawn_child_llm(strategy)
    results.append(trial)
    print(f"Trial {trial['trial_id']}: score={trial['metrics']['avg_score']}")

# Analyze results
successful = [r for r in results if r['success']]
best = max(successful, key=lambda r: r['metrics']['avg_score'])
print(f"\\nBest strategy: {best['trial_id']} with score {best['metrics']['avg_score']}")
"""

    result = repl.execute(code)

    assert result.success, f"Execution failed: {result.error}"
    assert api.spawn_count == 4
    assert "results" in result.locals
    assert len(result.locals["results"]) == 4
    print("✓ test_complex_workflow passed")


if __name__ == "__main__":
    print("Running REPL Proof of Concept Tests\n" + "=" * 40)

    test_basic_execution()
    test_state_persistence()
    test_api_injection()
    test_custom_functions()
    test_error_handling()
    test_import_capability()
    test_complex_workflow()

    print("\n" + "=" * 40)
    print("All REPL PoC tests passed!")
