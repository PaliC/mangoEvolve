"""
Experiment tracking system for mango_evolve.

Tracks token usage, cost, and wall-clock timing for performance analysis.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import Config
from .exceptions import BudgetExceededError


@dataclass
class TokenUsage:
    """Record of a single LLM API call's token usage."""

    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
    llm_type: str  # "root" or "child"
    call_id: str
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class TimingRecord:
    """Record of a wall-clock timing measurement."""

    operation: str  # "root_llm_call", "child_llm_call", "evaluation", "spawn_children", "selection", "generation"
    duration_s: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Summary of experiment tracking data (cost + timing)."""

    # Cost data
    total_cost: float
    remaining_budget: float
    total_input_tokens: int
    total_output_tokens: int
    root_cost: float
    root_calls: int
    child_costs: dict[str, float]
    child_calls: dict[str, int]
    total_child_cost: float
    total_child_calls: int
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    cache_savings: float = 0.0

    # Timing data
    total_wall_time_s: float = 0.0
    total_root_llm_time_s: float = 0.0
    total_child_llm_time_s: float = 0.0
    total_eval_time_s: float = 0.0
    total_spawn_time_s: float = 0.0
    num_generations_timed: int = 0
    avg_generation_time_s: float = 0.0


# Backward-compat alias
CostSummary = ExperimentSummary


class ExperimentTracker:
    """
    Tracks token usage, cost, and wall-clock timing.

    Supports different pricing for root and child LLMs (per-model).
    """

    def __init__(self, config: Config):
        """
        Initialize the experiment tracker.

        Args:
            config: Configuration containing LLM pricing and budget info
        """
        self.config = config
        self.usage_log: list[TokenUsage] = []
        self.timing_log: list[TimingRecord] = []
        self.total_cost: float = 0.0
        self._experiment_start_time: float = time.monotonic()

        # Cache pricing info (convert from per-million to per-token)
        # Root LLM pricing
        self._pricing: dict[str, dict[str, float]] = {
            "root": {
                "input": config.root_llm.cost_per_million_input_tokens / 1_000_000,
                "output": config.root_llm.cost_per_million_output_tokens / 1_000_000,
            },
        }

        # Add pricing for each child LLM by alias (format: "child:<alias>")
        self._child_aliases: list[str] = []
        for child_config in config.child_llms:
            alias = child_config.effective_alias
            self._child_aliases.append(alias)
            self._pricing[f"child:{alias}"] = {
                "input": child_config.cost_per_million_input_tokens / 1_000_000,
                "output": child_config.cost_per_million_output_tokens / 1_000_000,
            }

        self._max_budget = config.budget.max_total_cost

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        llm_type: str,
        call_id: str | None = None,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> TokenUsage:
        """
        Record token usage and compute cost.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            llm_type: "root" or "child:<alias>" (e.g., "child:fast_model")
            call_id: Optional unique identifier for this call
            cache_creation_input_tokens: Tokens written to cache (25% markup)
            cache_read_input_tokens: Tokens read from cache (90% discount)

        Returns:
            TokenUsage record with computed cost

        Note:
            Anthropic's cache pricing:
            - Cache creation: 25% more than base input price
            - Cache read: 90% discount (10% of base price)
            - Regular input tokens: counted in input_tokens but excludes cached
        """
        if llm_type not in self._pricing:
            valid_types = ["root"] + [f"child:{alias}" for alias in self._child_aliases]
            raise ValueError(
                f"Invalid llm_type: {llm_type}. Must be one of: {', '.join(valid_types)}"
            )

        pricing = self._pricing[llm_type]

        # Calculate cost with cache pricing
        non_cached_tokens = input_tokens - cache_creation_input_tokens - cache_read_input_tokens
        non_cached_tokens = max(0, non_cached_tokens)  # Safety check

        cost = (
            (non_cached_tokens * pricing["input"])
            + (cache_creation_input_tokens * pricing["input"] * 1.25)
            + (cache_read_input_tokens * pricing["input"] * 0.10)
            + (output_tokens * pricing["output"])
        )

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=datetime.now(),
            llm_type=llm_type,
            call_id=call_id or str(uuid.uuid4()),
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

        self.usage_log.append(usage)
        self.total_cost += cost

        return usage

    def record_timing(
        self,
        operation: str,
        duration_s: float,
        **metadata: Any,
    ) -> TimingRecord:
        """
        Record a wall-clock timing measurement.

        Args:
            operation: Operation name (e.g., "root_llm_call", "child_llm_call",
                      "evaluation", "spawn_children", "selection", "generation")
            duration_s: Duration in seconds
            **metadata: Additional context (generation, trial_id, model, etc.)

        Returns:
            TimingRecord with the recorded data
        """
        record = TimingRecord(
            operation=operation,
            duration_s=duration_s,
            timestamp=datetime.now(),
            metadata=dict(metadata),
        )
        self.timing_log.append(record)
        return record

    def get_generation_timing(self, generation: int) -> dict[str, Any]:
        """
        Get timing breakdown for a specific generation.

        Args:
            generation: Generation number

        Returns:
            Dictionary with timing breakdown for the generation
        """
        gen_records = [
            r for r in self.timing_log if r.metadata.get("generation") == generation
        ]
        result: dict[str, Any] = {}
        for r in gen_records:
            if r.operation not in result:
                result[r.operation] = {"total_s": 0.0, "count": 0}
            result[r.operation]["total_s"] += r.duration_s
            result[r.operation]["count"] += 1
        return result

    def check_budget(self) -> bool:
        """
        Check if we're still within budget.

        Returns:
            True if within budget, False if exceeded
        """
        return self.total_cost <= self._max_budget

    def get_remaining_budget(self) -> float:
        """
        Get remaining budget in USD.

        Returns:
            Remaining budget (can be negative if exceeded)
        """
        return self._max_budget - self.total_cost

    def raise_if_over_budget(self) -> None:
        """
        Raise BudgetExceededError if over budget.

        Raises:
            BudgetExceededError: If budget is exceeded
        """
        if not self.check_budget():
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self.total_cost:.4f} of ${self._max_budget:.2f}"
            )

    def get_summary(self) -> ExperimentSummary:
        """
        Get a summary of all tracking data (cost + timing).

        Returns:
            ExperimentSummary with aggregated statistics
        """
        root_usage = [u for u in self.usage_log if u.llm_type == "root"]

        # Per-model child costs and calls
        child_costs: dict[str, float] = {}
        child_calls: dict[str, int] = {}
        for alias in self._child_aliases:
            llm_type = f"child:{alias}"
            usage = [u for u in self.usage_log if u.llm_type == llm_type]
            child_costs[alias] = sum(u.cost for u in usage)
            child_calls[alias] = len(usage)

        total_child_cost = sum(child_costs.values())
        total_child_calls = sum(child_calls.values())

        # Calculate cache statistics
        total_cache_creation = sum(u.cache_creation_input_tokens for u in self.usage_log)
        total_cache_read = sum(u.cache_read_input_tokens for u in self.usage_log)

        # Calculate cache savings
        cache_savings = 0.0
        for u in self.usage_log:
            pricing = self._pricing.get(u.llm_type, self._pricing["root"])
            cache_savings += u.cache_read_input_tokens * pricing["input"] * 0.90

        # Aggregate timing data
        total_wall_time_s = time.monotonic() - self._experiment_start_time
        total_root_llm_time_s = sum(
            r.duration_s for r in self.timing_log if r.operation == "root_llm_call"
        )
        total_child_llm_time_s = sum(
            r.duration_s for r in self.timing_log if r.operation == "child_llm_call"
        )
        total_eval_time_s = sum(
            r.duration_s for r in self.timing_log if r.operation == "evaluation"
        )
        total_spawn_time_s = sum(
            r.duration_s for r in self.timing_log if r.operation == "spawn_children"
        )
        gen_records = [r for r in self.timing_log if r.operation == "generation"]
        num_generations_timed = len(gen_records)
        avg_generation_time_s = (
            sum(r.duration_s for r in gen_records) / num_generations_timed
            if num_generations_timed > 0
            else 0.0
        )

        return ExperimentSummary(
            total_cost=self.total_cost,
            remaining_budget=self.get_remaining_budget(),
            total_input_tokens=sum(u.input_tokens for u in self.usage_log),
            total_output_tokens=sum(u.output_tokens for u in self.usage_log),
            root_cost=sum(u.cost for u in root_usage),
            root_calls=len(root_usage),
            child_costs=child_costs,
            child_calls=child_calls,
            total_child_cost=total_child_cost,
            total_child_calls=total_child_calls,
            total_cache_creation_tokens=total_cache_creation,
            total_cache_read_tokens=total_cache_read,
            cache_savings=cache_savings,
            total_wall_time_s=total_wall_time_s,
            total_root_llm_time_s=total_root_llm_time_s,
            total_child_llm_time_s=total_child_llm_time_s,
            total_eval_time_s=total_eval_time_s,
            total_spawn_time_s=total_spawn_time_s,
            num_generations_timed=num_generations_timed,
            avg_generation_time_s=avg_generation_time_s,
        )

    def to_dict(self) -> dict:
        """
        Serialize tracker state to dictionary.

        Returns:
            Dictionary representation of the tracker
        """
        summary = self.get_summary()
        return {
            "total_cost": self.total_cost,
            "max_budget": self._max_budget,
            "root_cost": summary.root_cost,
            "root_calls": summary.root_calls,
            "child_costs": summary.child_costs,
            "child_calls": summary.child_calls,
            "total_child_cost": summary.total_child_cost,
            "total_child_calls": summary.total_child_calls,
            "cache_savings": summary.cache_savings,
            "total_cache_creation_tokens": summary.total_cache_creation_tokens,
            "total_cache_read_tokens": summary.total_cache_read_tokens,
            "usage_log": [
                {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost": u.cost,
                    "timestamp": u.timestamp.isoformat(),
                    "llm_type": u.llm_type,
                    "call_id": u.call_id,
                    "cache_creation_input_tokens": u.cache_creation_input_tokens,
                    "cache_read_input_tokens": u.cache_read_input_tokens,
                }
                for u in self.usage_log
            ],
            "timing": {
                "total_wall_time_s": summary.total_wall_time_s,
                "total_root_llm_time_s": summary.total_root_llm_time_s,
                "total_child_llm_time_s": summary.total_child_llm_time_s,
                "total_eval_time_s": summary.total_eval_time_s,
                "total_spawn_time_s": summary.total_spawn_time_s,
                "num_generations_timed": summary.num_generations_timed,
                "avg_generation_time_s": summary.avg_generation_time_s,
            },
            "timing_log": [
                {
                    "operation": r.operation,
                    "duration_s": r.duration_s,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata,
                }
                for r in self.timing_log
            ],
        }

    def format_timing_summary(self) -> str:
        """
        Format a human-readable timing summary for terminal output.

        Returns:
            Formatted timing summary string
        """
        summary = self.get_summary()
        total = summary.total_wall_time_s

        if total <= 0:
            return "  (no timing data)"

        def fmt_time(s: float) -> str:
            if s >= 60:
                return f"{s:.1f}s ({int(s // 60)}m {int(s % 60)}s)"
            return f"{s:.1f}s"

        def pct(s: float) -> str:
            return f"{s / total * 100:.1f}%" if total > 0 else "0.0%"

        root_llm_calls = sum(1 for r in self.timing_log if r.operation == "root_llm_call")
        child_llm_calls = sum(1 for r in self.timing_log if r.operation == "child_llm_call")
        eval_calls = sum(1 for r in self.timing_log if r.operation == "evaluation")
        overhead = total - summary.total_root_llm_time_s - summary.total_spawn_time_s

        lines = [
            f"  Total wall time:       {fmt_time(total)}",
            f"  Root LLM calls:        {fmt_time(summary.total_root_llm_time_s)} ({pct(summary.total_root_llm_time_s)})  [{root_llm_calls} calls]",
            f"  Spawn children:        {fmt_time(summary.total_spawn_time_s)} ({pct(summary.total_spawn_time_s)})",
            f"    Child LLM calls:     {fmt_time(summary.total_child_llm_time_s)}  [{child_llm_calls} calls]",
            f"    Evaluation:          {fmt_time(summary.total_eval_time_s)}  [{eval_calls} evals]",
            f"  Overhead:              {fmt_time(max(0, overhead))} ({pct(max(0, overhead))})",
        ]

        if summary.num_generations_timed > 0:
            lines.append(f"  Avg generation time:   {fmt_time(summary.avg_generation_time_s)}")

        return "\n".join(lines)


# Backward-compat alias
CostTracker = ExperimentTracker
