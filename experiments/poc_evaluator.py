"""
Proof of Concept: Program Evaluator

This validates that we can:
1. Execute dynamically generated code safely
2. Extract functions from code strings
3. Run games with an agent function
4. Compute metrics over multiple runs
5. Handle errors gracefully

NOTE: This uses a MockTetrisEnv since tetris-gymnasium has Python version
compatibility issues with 3.14. The actual integration will work the same way.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


class MockTetrisEnv:
    """
    Mock Tetris environment that simulates game dynamics.
    This mimics the gymnasium interface.
    """

    def __init__(self):
        self.board = np.zeros((20, 10), dtype=np.int8)
        self.current_piece = np.random.randint(0, 7)
        self.next_piece = np.random.randint(0, 7)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False

    @property
    def observation_space(self):
        return {"shape": (20, 10)}

    @property
    def action_space(self):
        return {"n": 6}  # 0-5 actions

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((20, 10), dtype=np.int8)
        self.current_piece = np.random.randint(0, 7)
        self.next_piece = np.random.randint(0, 7)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.steps += 1

        # Simulate game dynamics (simplified)
        reward = 0

        if action == 5:  # Hard drop
            # Random chance to clear lines based on action
            if np.random.random() < 0.3:
                lines = np.random.randint(1, 5)
                self.lines_cleared += lines
                reward = lines * 100
                self.score += reward

        # Random chance of game over (increases with height)
        if np.random.random() < 0.001 * self.steps:
            self.game_over = True

        # Small reward for surviving
        reward += 1
        self.score += 1

        # Update pieces
        self.current_piece = self.next_piece
        self.next_piece = np.random.randint(0, 7)

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, self.game_over, False, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self.board.copy()

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dict."""
        return {
            "current_piece": self.current_piece,
            "next_piece": self.next_piece,
            "lines_cleared": self.lines_cleared,
            "score": self.score,
        }

    def close(self):
        pass


@dataclass
class EvaluationResult:
    avg_score: float
    avg_lines_cleared: float
    avg_steps: float
    max_score: float
    min_score: float
    std_score: float
    success_rate: float
    error: Optional[str] = None


class TetrisEvaluator:
    """Evaluates generated Tetris agent code."""

    def __init__(
        self,
        num_games: int = 10,
        max_steps_per_game: int = 1000,
        env_class=MockTetrisEnv,
    ):
        self.num_games = num_games
        self.max_steps = max_steps_per_game
        self.env_class = env_class

    def evaluate(
        self, code: str, num_games: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate agent code by running multiple games.

        Args:
            code: Python code that must define select_action(observation, info) -> int
            num_games: Number of games to run (defaults to self.num_games)

        Returns:
            Dictionary with evaluation metrics
        """
        num_games = num_games or self.num_games

        # Create isolated namespace for code execution
        namespace = {"np": np, "numpy": np}

        try:
            # Execute the code to define select_action
            exec(code, namespace)

            if "select_action" not in namespace:
                return {
                    "error": "No select_action function defined",
                    "success_rate": 0.0,
                }

            select_action = namespace["select_action"]

            # Validate it's callable
            if not callable(select_action):
                return {
                    "error": "select_action is not callable",
                    "success_rate": 0.0,
                }

            # Run games
            scores = []
            lines_cleared = []
            steps = []
            crashes = 0

            for game_idx in range(num_games):
                try:
                    score, lines, game_steps = self._run_game(select_action, game_idx)
                    scores.append(score)
                    lines_cleared.append(lines)
                    steps.append(game_steps)
                except Exception as e:
                    crashes += 1
                    # Log but continue
                    # print(f"Game {game_idx} crashed: {e}")

            if not scores:
                return {
                    "error": f"All {num_games} games crashed",
                    "success_rate": 0.0,
                }

            return {
                "avg_score": float(np.mean(scores)),
                "avg_lines_cleared": float(np.mean(lines_cleared)),
                "avg_steps": float(np.mean(steps)),
                "max_score": float(max(scores)),
                "min_score": float(min(scores)),
                "std_score": float(np.std(scores)),
                "success_rate": (num_games - crashes) / num_games,
            }

        except SyntaxError as e:
            return {"error": f"Syntax error: {e}", "success_rate": 0.0}
        except Exception as e:
            return {"error": f"Execution error: {e}", "success_rate": 0.0}

    def _run_game(
        self, select_action: Callable, seed: int
    ) -> Tuple[float, int, int]:
        """Run a single game and return (score, lines_cleared, steps)."""
        env = self.env_class()
        observation, info = env.reset(seed=seed)

        total_score = 0
        game_steps = 0

        while game_steps < self.max_steps:
            # Call the agent's select_action function
            action = select_action(observation, info)

            # Validate action
            if not isinstance(action, (int, np.integer)):
                action = int(action)
            action = max(0, min(5, action))  # Clamp to valid range

            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_score += reward
            game_steps += 1

            if terminated or truncated:
                break

        env.close()
        return total_score, info["lines_cleared"], game_steps


# Test agents (these are what the Child LLMs would generate)


RANDOM_AGENT = """
import numpy as np

def select_action(observation, info):
    '''Random agent - picks random actions.'''
    return np.random.randint(0, 6)
"""

GREEDY_AGENT = """
import numpy as np

def select_action(observation, info):
    '''Greedy agent - always tries to hard drop for line clears.'''
    # Simple strategy: mostly hard drop
    return 5  # Hard drop
"""

SMART_AGENT = """
import numpy as np

def select_action(observation, info):
    '''Smarter agent - varies actions based on piece.'''
    piece = info.get('current_piece', 0)

    # Vary behavior based on piece type
    if piece in [0, 1]:  # Long pieces
        return 5  # Hard drop
    elif piece in [2, 3]:  # Square/T pieces
        return np.random.choice([1, 2, 5])  # Move or drop
    else:
        return np.random.choice([3, 4, 5])  # Rotate or drop
"""

BROKEN_AGENT = """
def select_action(observation, info):
    '''Broken agent - will crash.'''
    return undefined_variable  # This will raise NameError
"""

NO_FUNCTION_AGENT = """
# Forgot to define select_action
x = 5
"""

SYNTAX_ERROR_AGENT = """
def select_action(observation, info)  # Missing colon
    return 0
"""


def test_random_agent():
    """Test that random agent can be evaluated."""
    evaluator = TetrisEvaluator(num_games=5, max_steps_per_game=100)
    result = evaluator.evaluate(RANDOM_AGENT)

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result["success_rate"] == 1.0
    assert result["avg_score"] > 0
    print(f"✓ test_random_agent passed: avg_score={result['avg_score']:.1f}")


def test_greedy_agent():
    """Test greedy agent."""
    evaluator = TetrisEvaluator(num_games=5, max_steps_per_game=100)
    result = evaluator.evaluate(GREEDY_AGENT)

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result["success_rate"] == 1.0
    print(f"✓ test_greedy_agent passed: avg_score={result['avg_score']:.1f}")


def test_smart_agent():
    """Test smart agent."""
    evaluator = TetrisEvaluator(num_games=5, max_steps_per_game=100)
    result = evaluator.evaluate(SMART_AGENT)

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result["success_rate"] == 1.0
    print(f"✓ test_smart_agent passed: avg_score={result['avg_score']:.1f}")


def test_broken_agent():
    """Test that broken agent errors are captured."""
    evaluator = TetrisEvaluator(num_games=5, max_steps_per_game=100)
    result = evaluator.evaluate(BROKEN_AGENT)

    assert result["success_rate"] == 0.0
    print(f"✓ test_broken_agent passed: success_rate=0.0 (as expected)")


def test_no_function():
    """Test that missing function is detected."""
    evaluator = TetrisEvaluator(num_games=5, max_steps_per_game=100)
    result = evaluator.evaluate(NO_FUNCTION_AGENT)

    assert "error" in result
    assert "select_action" in result["error"]
    print(f"✓ test_no_function passed: error detected correctly")


def test_syntax_error():
    """Test that syntax errors are captured."""
    evaluator = TetrisEvaluator(num_games=5, max_steps_per_game=100)
    result = evaluator.evaluate(SYNTAX_ERROR_AGENT)

    assert "error" in result
    assert "Syntax error" in result["error"]
    print(f"✓ test_syntax_error passed: syntax error detected")


def test_compare_agents():
    """Compare different agents to show metrics differentiation."""
    evaluator = TetrisEvaluator(num_games=10, max_steps_per_game=200)

    agents = {
        "random": RANDOM_AGENT,
        "greedy": GREEDY_AGENT,
        "smart": SMART_AGENT,
    }

    results = {}
    for name, code in agents.items():
        result = evaluator.evaluate(code)
        results[name] = result

    print("\n✓ test_compare_agents passed:")
    print("  Agent Comparison:")
    for name, result in results.items():
        print(
            f"    {name}: avg_score={result['avg_score']:.1f}, "
            f"lines={result['avg_lines_cleared']:.1f}"
        )


if __name__ == "__main__":
    print("Running Evaluator Proof of Concept Tests\n" + "=" * 50)

    test_random_agent()
    test_greedy_agent()
    test_smart_agent()
    test_broken_agent()
    test_no_function()
    test_syntax_error()
    test_compare_agents()

    print("\n" + "=" * 50)
    print("All Evaluator PoC tests passed!")
