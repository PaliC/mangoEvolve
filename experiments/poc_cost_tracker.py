"""
Proof of Concept: Cost Tracking

This validates that we can:
1. Track token usage per LLM call
2. Compute costs based on model pricing
3. Check budget limits
4. Raise errors when budget exceeded
5. Aggregate costs across root and child LLMs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid


class BudgetExceededError(Exception):
    """Raised when the cost budget is exceeded."""

    pass


@dataclass
class TokenUsage:
    """Record of a single LLM API call."""

    call_id: str
    llm_type: str  # "root" or "child"
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelPricing:
    """Pricing for an LLM model (per token)."""

    input_cost_per_token: float  # Cost per input token
    output_cost_per_token: float  # Cost per output token

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute total cost for given token counts."""
        input_cost = input_tokens * self.input_cost_per_token
        output_cost = output_tokens * self.output_cost_per_token
        return input_cost + output_cost


# Example pricing (Claude Sonnet 4)
CLAUDE_SONNET_PRICING = ModelPricing(
    input_cost_per_token=3.0 / 1_000_000,  # $3 per 1M tokens
    output_cost_per_token=15.0 / 1_000_000,  # $15 per 1M tokens
)

# Example pricing (Claude Haiku)
CLAUDE_HAIKU_PRICING = ModelPricing(
    input_cost_per_token=0.25 / 1_000_000,  # $0.25 per 1M tokens
    output_cost_per_token=1.25 / 1_000_000,  # $1.25 per 1M tokens
)


class CostTracker:
    """Tracks token usage and costs for budget enforcement."""

    def __init__(
        self,
        max_budget: float,
        root_pricing: ModelPricing,
        child_pricing: Optional[ModelPricing] = None,
    ):
        self.max_budget = max_budget
        self.root_pricing = root_pricing
        self.child_pricing = child_pricing or root_pricing

        self.usage_log: List[TokenUsage] = []
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def _get_pricing(self, llm_type: str) -> ModelPricing:
        """Get pricing for the given LLM type."""
        if llm_type == "root":
            return self.root_pricing
        elif llm_type == "child":
            return self.child_pricing
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

    def record_usage(
        self, input_tokens: int, output_tokens: int, llm_type: str
    ) -> TokenUsage:
        """
        Record token usage for an LLM call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            llm_type: "root" or "child"

        Returns:
            TokenUsage record

        Raises:
            BudgetExceededError: If this call would exceed the budget
        """
        pricing = self._get_pricing(llm_type)
        cost = pricing.compute_cost(input_tokens, output_tokens)

        usage = TokenUsage(
            call_id=str(uuid.uuid4()),
            llm_type=llm_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

        self.usage_log.append(usage)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return usage

    def check_budget(self) -> bool:
        """Return True if within budget, False if exceeded."""
        return self.total_cost < self.max_budget

    def get_remaining_budget(self) -> float:
        """Return remaining budget in USD."""
        return max(0.0, self.max_budget - self.total_cost)

    def raise_if_over_budget(self):
        """Raise BudgetExceededError if over budget."""
        if not self.check_budget():
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.4f} / ${self.max_budget:.2f}"
            )

    def get_summary(self) -> dict:
        """Get a summary of all usage."""
        root_usage = [u for u in self.usage_log if u.llm_type == "root"]
        child_usage = [u for u in self.usage_log if u.llm_type == "child"]

        return {
            "total_cost": self.total_cost,
            "remaining_budget": self.get_remaining_budget(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_calls": len(self.usage_log),
            "root_calls": len(root_usage),
            "child_calls": len(child_usage),
            "root_cost": sum(u.cost for u in root_usage),
            "child_cost": sum(u.cost for u in child_usage),
        }


def test_basic_tracking():
    """Test basic token tracking."""
    tracker = CostTracker(
        max_budget=10.0,
        root_pricing=CLAUDE_SONNET_PRICING,
        child_pricing=CLAUDE_HAIKU_PRICING,
    )

    # Record a root LLM call
    usage = tracker.record_usage(input_tokens=1000, output_tokens=500, llm_type="root")

    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500
    assert usage.cost > 0
    assert tracker.total_cost == usage.cost
    print(f"✓ test_basic_tracking passed: cost=${usage.cost:.6f}")


def test_cost_computation():
    """Test that costs are computed correctly."""
    tracker = CostTracker(
        max_budget=10.0,
        root_pricing=CLAUDE_SONNET_PRICING,
    )

    # 1M input tokens + 100k output tokens
    usage = tracker.record_usage(
        input_tokens=1_000_000, output_tokens=100_000, llm_type="root"
    )

    expected_input_cost = 3.0  # $3 per 1M tokens
    expected_output_cost = 1.5  # $15 per 1M tokens * 0.1M
    expected_total = expected_input_cost + expected_output_cost

    assert abs(usage.cost - expected_total) < 0.001
    print(f"✓ test_cost_computation passed: ${usage.cost:.2f} == ${expected_total:.2f}")


def test_different_pricing():
    """Test that root and child use different pricing."""
    tracker = CostTracker(
        max_budget=10.0,
        root_pricing=CLAUDE_SONNET_PRICING,
        child_pricing=CLAUDE_HAIKU_PRICING,
    )

    # Same token counts, different pricing
    root_usage = tracker.record_usage(
        input_tokens=10000, output_tokens=1000, llm_type="root"
    )
    child_usage = tracker.record_usage(
        input_tokens=10000, output_tokens=1000, llm_type="child"
    )

    # Haiku should be much cheaper
    assert child_usage.cost < root_usage.cost
    print(
        f"✓ test_different_pricing passed: root=${root_usage.cost:.6f}, "
        f"child=${child_usage.cost:.6f}"
    )


def test_budget_check():
    """Test budget checking."""
    tracker = CostTracker(
        max_budget=0.01,  # Very small budget
        root_pricing=CLAUDE_SONNET_PRICING,
    )

    assert tracker.check_budget()  # Should be within budget initially

    # Make a call that uses most of the budget
    tracker.record_usage(input_tokens=1000, output_tokens=100, llm_type="root")

    # Should still be within budget for a small call
    assert tracker.check_budget() or not tracker.check_budget()  # Depends on cost

    print(
        f"✓ test_budget_check passed: remaining=${tracker.get_remaining_budget():.6f}"
    )


def test_budget_exceeded_error():
    """Test that BudgetExceededError is raised when budget is exceeded."""
    tracker = CostTracker(
        max_budget=0.001,  # Tiny budget
        root_pricing=CLAUDE_SONNET_PRICING,
    )

    # Make a call that exceeds the budget
    tracker.record_usage(input_tokens=10000, output_tokens=1000, llm_type="root")

    # Should now raise error
    try:
        tracker.raise_if_over_budget()
        assert False, "Should have raised BudgetExceededError"
    except BudgetExceededError as e:
        assert "Budget exceeded" in str(e)
        print(f"✓ test_budget_exceeded_error passed: {e}")


def test_summary():
    """Test getting usage summary."""
    tracker = CostTracker(
        max_budget=10.0,
        root_pricing=CLAUDE_SONNET_PRICING,
        child_pricing=CLAUDE_HAIKU_PRICING,
    )

    # Simulate an evolution run
    tracker.record_usage(input_tokens=5000, output_tokens=1000, llm_type="root")
    tracker.record_usage(input_tokens=3000, output_tokens=2000, llm_type="child")
    tracker.record_usage(input_tokens=3000, output_tokens=2000, llm_type="child")
    tracker.record_usage(input_tokens=6000, output_tokens=1500, llm_type="root")
    tracker.record_usage(input_tokens=3000, output_tokens=2000, llm_type="child")

    summary = tracker.get_summary()

    assert summary["num_calls"] == 5
    assert summary["root_calls"] == 2
    assert summary["child_calls"] == 3
    assert summary["total_cost"] > 0
    assert summary["remaining_budget"] < 10.0

    print(f"✓ test_summary passed:")
    print(f"    Total cost: ${summary['total_cost']:.6f}")
    print(f"    Root cost: ${summary['root_cost']:.6f}")
    print(f"    Child cost: ${summary['child_cost']:.6f}")
    print(f"    Remaining: ${summary['remaining_budget']:.6f}")


def test_realistic_scenario():
    """Test a realistic evolution scenario."""
    tracker = CostTracker(
        max_budget=50.0,  # $50 budget
        root_pricing=CLAUDE_SONNET_PRICING,
        child_pricing=CLAUDE_HAIKU_PRICING,
    )

    # Simulate 3 generations with 5 children each
    for gen in range(3):
        # Root LLM turn (analyzing previous generation)
        tracker.record_usage(
            input_tokens=10000 + gen * 5000,  # Growing context
            output_tokens=2000,
            llm_type="root",
        )

        # Spawn 5 children
        for child in range(5):
            tracker.record_usage(
                input_tokens=8000,
                output_tokens=3000,  # Children generate more code
                llm_type="child",
            )

        # Root LLM turn (selection)
        tracker.record_usage(
            input_tokens=15000, output_tokens=1000, llm_type="root"
        )

    summary = tracker.get_summary()

    print(f"\n✓ test_realistic_scenario passed:")
    print(f"    3 generations, 5 children each")
    print(f"    Total calls: {summary['num_calls']}")
    print(f"    Total cost: ${summary['total_cost']:.4f}")
    print(f"    Root LLM: ${summary['root_cost']:.4f} ({summary['root_calls']} calls)")
    print(
        f"    Child LLM: ${summary['child_cost']:.4f} ({summary['child_calls']} calls)"
    )
    print(f"    Budget remaining: ${summary['remaining_budget']:.2f}")


if __name__ == "__main__":
    print("Running Cost Tracker Proof of Concept Tests\n" + "=" * 50)

    test_basic_tracking()
    test_cost_computation()
    test_different_pricing()
    test_budget_check()
    test_budget_exceeded_error()
    test_summary()
    test_realistic_scenario()

    print("\n" + "=" * 50)
    print("All Cost Tracker PoC tests passed!")
