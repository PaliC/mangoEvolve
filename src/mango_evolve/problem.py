"""Problem specification and base evaluator for MangoEvolve.

This module defines the contract between problems and the evolution engine.
Each problem provides a ProblemSpec that describes it to the LLMs, and an
evaluator that scores candidate solutions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemSpec:
    """Problem specification for prompt generation.

    This dataclass contains all the information needed to generate prompts
    for both the Root LLM and Child LLMs. The evaluator provides this spec
    so that problem definition stays co-located with evaluation logic.
    """

    # Required fields (no defaults)
    name: str
    description: str
    objective: str  # "maximize" or "minimize"
    metric_name: str
    entry_function: str
    return_description: str

    # Optional fields (with defaults)
    best_known_solution: float | None = None
    helper_functions: list[str] | None = None
    allowed_modules: list[str] | None = None
    constraints: list[str] | None = None
    example_code: str | None = None

    # For optimization problems with reference implementations (e.g., KernelBench)
    reference_code: str | None = None  # Code to optimize/improve
    reference_context: str | None = None  # Additional context (hardware, inputs, etc.)

    # Additional metrics shown to LLM (evaluator computes these)
    secondary_metrics: list[str] | None = None  # e.g., ["memory_usage", "correctness"]


class BaseProblemEvaluator(ABC):
    """Base class for problem evaluators with metadata.

    All problem evaluators should extend this class and implement:
    - get_problem_spec(): Return the problem specification for prompt generation
    - evaluate(): Score a candidate solution

    Example:
        class MyEvaluator(BaseProblemEvaluator):
            def get_problem_spec(self) -> ProblemSpec:
                return ProblemSpec(
                    name="My Problem",
                    description="Solve this interesting problem...",
                    objective="maximize",
                    metric_name="score",
                    entry_function="solve",
                    return_description="Return the solution value",
                )

            def evaluate(self, code: str) -> dict[str, Any]:
                # Execute code, compute score
                return {"valid": True, "score": 42.0, "eval_time": 1.5, "error": None}
    """

    @abstractmethod
    def get_problem_spec(self) -> ProblemSpec:
        """Return problem specification for prompt generation.

        This method should return a ProblemSpec that fully describes the problem
        to the LLMs. The spec is used to generate system prompts and mutation
        prompts throughout the evolution process.

        Returns:
            ProblemSpec with all problem details
        """
        pass

    @abstractmethod
    def evaluate(self, code: str) -> dict[str, Any]:
        """Evaluate code and return metrics.

        This method executes the candidate code and computes metrics. The returned
        dict must include the required keys, but may include additional metrics
        that will be shown in trial results.

        Args:
            code: Python source code to evaluate

        Returns:
            dict with required keys:
                - valid (bool): Whether the solution is valid
                - score (float): Primary metric value (0 if invalid)
                - eval_time (float): Execution time in seconds
                - error (str | None): Error message if any

            Optional additional keys for secondary metrics:
                - Any additional metrics specified in ProblemSpec.secondary_metrics
                - These will be included in trial results for the Root LLM to see
        """
        pass
