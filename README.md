# Tetris AI with AlphaEvolve

This project uses the AlphaEvolve approach to evolve code-based AI agents that play Tetris.

## Overview

Instead of training neural networks, we evolve the actual **decision-making code** that controls the Tetris agent. AlphaEvolve will modify the heuristics, evaluation functions, and decision logic to discover better strategies.

## Components

### 1. `tetris_env.py` - Tetris Environment
- Gymnasium-compatible Tetris implementation
- 20x10 board, standard Tetris pieces
- Scoring based on lines cleared

### 2. `tetris_agent.py` - Evolved Agent Code
- Contains the code that AlphaEvolve will evolve
- Key functions inside `EVOLVE-BLOCK-START/END`:
  - `compute_heuristics()` - Extracts features from board state
  - `decide_action()` - Chooses actions based on heuristics
  - `evaluate_position()` - Scores potential placements

### 3. `tetris_evaluator.py` - Performance Evaluator
- Runs multiple games and computes metrics
- Scores agents on: average score, lines cleared, pieces placed, survival time
- This is what AlphaEvolve uses to rank evolved programs

### 4. `tetris_pufferlib.py` - PufferLib Integration
- Wraps environment for efficient vectorization
- Allows parallel game execution
- Optional: only needed for faster evaluation

## Quick Start

### Test the Environment
```bash
python tetris_env.py
```

### Test the Agent
```bash
python tetris_agent.py
```

### Evaluate Agent Performance
```bash
python tetris_evaluator.py tetris_agent.py
```

### Test PufferLib Integration (optional)
```bash
pip install pufferlib
python tetris_pufferlib.py
```

## Using with AlphaEvolve

### Option 1: Using OpenEvolve (Open Source)

1. Install OpenEvolve:
```bash
pip install openevolve
```

2. Set up your API key (for LLM):
```bash
export OPENAI_API_KEY="your-gemini-api-key"  # Or OpenAI key
```

3. Run evolution:
```bash
python run_tetris_evolution.py
```

### Option 2: Manual Evolution Loop

See `run_tetris_evolution.py` for a simple example of:
- Loading initial program
- Having LLM propose modifications
- Evaluating new versions
- Keeping the best performers

## What Gets Evolved?

The code between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers:

1. **Heuristic Functions**
   - `compute_heuristics()` - What features to extract
   - `count_holes()` - How to count board defects
   - `compute_bumpiness()` - Surface variation calculation

2. **Decision Logic**
   - `decide_action()` - Action selection strategy
   - Weights and thresholds
   - Conditional logic

3. **Evaluation Function**
   - `evaluate_position()` - How to score placements
   - Feature weights
   - Lookahead strategies

## Evaluation Metrics

The evaluator runs 5 games per evolved program and computes:

- **Primary Score** (used by AlphaEvolve):
  ```
  score = avg_game_score * 1.0 + 
          avg_lines * 10.0 + 
          avg_pieces * 1.0 + 
          avg_steps * 0.01
  ```

- **Secondary Metrics**:
  - Average/max score per game
  - Average/max lines cleared
  - Average pieces placed
  - Average survival steps
  - Success rate (games with score > 0)

## Expected Evolution Patterns

AlphaEvolve might discover:

1. **Better Heuristics**
   - More nuanced hole detection
   - Well depth calculations
   - Column height variance
   - Row transitions

2. **Smarter Decision Logic**
   - Look-ahead strategies
   - Piece-specific tactics
   - Emergency recovery moves
   - Setup for T-spins or Tetrises

3. **Improved Weights**
   - Optimal balance between avoiding holes and clearing lines
   - Dynamic weights based on board state
   - Risk/reward trade-offs

## Performance Tips

### For Faster Evaluation
- Reduce `num_games` in evaluator (default: 5)
- Reduce `max_steps` (default: 3000)
- Use PufferLib for parallel execution

### For Better Evolution
- Increase `num_games` for more stable scores
- Add more diverse heuristics
- Include next-piece lookahead in evaluation
- Multi-objective optimization (score + stability)

## Example Evolution Run

```python
# run_tetris_evolution.py
from openevolve import run_evolution

result = run_evolution(
    initial_program='tetris_agent.py',
    evaluator=lambda path: evaluate_agent(path, num_games=5),
    iterations=100
)

print(f"Best score: {result.best_score}")
print(f"Best code saved to: {result.best_code_path}")
```

## Debugging

If agent crashes or gets low scores:
1. Check syntax errors in evolved code
2. Verify heuristics return valid numbers
3. Ensure actions are in range [0, 5]
4. Test with `render=True` to visualize behavior

## Customization

### Change Board Size
```python
env = TetrisEnv(width=12, height=24)
agent = EvolvedTetrisAgent(width=12, height=24)
```

### Add New Heuristics
Add new functions in the `EVOLVE-BLOCK`:
```python
def compute_advanced_features(board):
    # Your new heuristic
    return feature_value
```

### Modify Scoring
Edit the primary score calculation in `tetris_evaluator.py`:
```python
'score': (
    np.mean(scores) * 2.0 +      # Weight game score more
    np.mean(lines_cleared) * 50.0  # Really prioritize lines
)
```

## Next Steps

1. ✓ Basic Tetris environment
2. ✓ Evolved agent framework
3. ✓ Evaluation system
4. ✓ PufferLib integration
5. → Integrate with OpenEvolve/AlphaEvolve
6. → Run evolution experiments
7. → Analyze discovered strategies
8. → Visualize best evolved agents

## Resources

- [AlphaEvolve Paper](https://arxiv.org/abs/2506.13131)
- [OpenEvolve GitHub](https://github.com/codelion/openevolve)
- [PufferLib Docs](https://puffer.ai/docs.html)
- [Gymnasium Docs](https://gymnasium.farama.org/)

## License

MIT
