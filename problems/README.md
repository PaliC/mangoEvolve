# MangoEvolve Problems

This directory contains problem definitions for MangoEvolve. Each problem is self-contained with its own evaluator, configs, and data.

## Structure

```
problems/
  problem_name/
    __init__.py           # Exports the evaluator class
    evaluator.py          # Problem evaluator extending BaseProblemEvaluator
    configs/              # YAML configuration files
      sonnet.yaml
      gemini_flash.yaml
    scripts/              # Setup scripts (e.g., data download)
    data/                 # Downloaded datasets (gitignored)
    experiments/          # Run outputs (gitignored)
```

## Available Problems

### Circle Packing

Pack N circles into a unit square to maximize the sum of radii. Classic optimization benchmark.

```bash
uv run python -m mango_evolve --config problems/circle_packing/configs/sonnet.yaml
```

### Symbolic Regression

Evolve mathematical expressions to fit data. Supports two modes:

**1. Synthetic Data (default)**
Generate data from a target function:

```yaml
evaluation:
  evaluator_fn: "problems.symbolic_regression.evaluator:SymbolicRegressionEvaluator"
  evaluator_kwargs:
    target_function: "np.sin(x[:, 0]) + 0.5 * x[:, 0]**2"
    n_samples: 100
    noise_std: 0.1
```

**2. LLM-SRBench (scientific benchmark)**
Real scientific datasets from biology, chemistry, materials science, and physics.

First, download the dataset:
```bash
uv add datasets  # Hugging Face datasets library
python -m problems.symbolic_regression.scripts.setup_llm_srbench
```

Then run with:
```yaml
evaluation:
  evaluator_fn: "problems.symbolic_regression.evaluator:SymbolicRegressionEvaluator"
  evaluator_kwargs:
    benchmark: "llm_srbench"
    domain: "phys"        # bio, chem, matsci, phys, or transform
    problem_index: 0      # 0 to N-1 problems in domain
    n_params: 5
```

LLM-SRBench domains:
- `bio`: Biology - population growth (24 problems)
- `chem`: Chemistry - reactions (36 problems)
- `matsci`: Materials Science (25 problems)
- `phys`: Physics - oscillation (44 problems)
- `transform`: Transformed physical models (111 problems)

**Running the Full Benchmark**

Use the benchmark runner to run experiments across multiple domains:

```bash
# List available domains
./scripts/run_llm_srbench.sh --list

# Run first 3 problems in physics domain
./scripts/run_llm_srbench.sh phys --max-problems 3

# Run all domains with 2 problems each
./scripts/run_llm_srbench.sh all --max-problems 2

# Run specific problems in chemistry
./scripts/run_llm_srbench.sh chem --problems 0,5,10

# Preview what would run (dry-run)
./scripts/run_llm_srbench.sh phys --dry-run

# Use custom LLM config
./scripts/run_llm_srbench.sh phys --base-config my_config.yaml
```

The benchmark runner generates configs on the fly and runs MangoEvolve for each problem, saving results to `problems/symbolic_regression/experiments/benchmark/`.

### Heilbronn Triangle

Place 11 points inside a unit-area equilateral triangle to maximize the minimum
triangle area formed by any three points.

```bash
uv run python -m mango_evolve --config problems/heilbronn_triangle/configs/sonnet.yaml
```

### Minimizing Max-Min Distance

Place points in Euclidean space to maximize the squared ratio of minimum to
maximum pairwise distance (normalized to a benchmark score of 1.0).

```bash
# 16 points in 2D
uv run python -m mango_evolve --config problems/minimizing_max_min_dist/configs/dim2_sonnet.yaml

# 14 points in 3D
uv run python -m mango_evolve --config problems/minimizing_max_min_dist/configs/dim3_sonnet.yaml
```

## Adding a New Problem

1. Create a folder under `problems/`
2. Implement an evaluator extending `BaseProblemEvaluator`:

```python
from mango_evolve.problem import BaseProblemEvaluator, ProblemSpec

class MyProblemEvaluator(BaseProblemEvaluator):
    def __init__(self, param1: int = 10, ...):
        self.param1 = param1
        ...

    def get_problem_spec(self) -> ProblemSpec:
        return ProblemSpec(
            name="My Problem",
            description="Description shown to LLM...",
            objective="maximize",  # or "minimize"
            metric_name="score",
            entry_function="solve",
            return_description="Return format...",
            ...
        )

    def evaluate(self, code: str) -> dict[str, Any]:
        # Execute code and return metrics
        return {
            "valid": True,
            "score": 0.95,
            "eval_time": 1.23,
            "error": None,
            # Add custom metrics
        }
```

3. Create config files pointing to your evaluator:

```yaml
experiment:
  name: "my_problem_test"
  output_dir: "experiments"

evaluation:
  evaluator_fn: "problems.my_problem.evaluator:MyProblemEvaluator"
  evaluator_kwargs:
    param1: 10

# ... rest of config (root_llm, child_llms, etc.)
```

4. Run with:
```bash
uv run python -m mango_evolve --config problems/my_problem/configs/my_config.yaml
```

## Viewing Results

After running an experiment, use the results script:

```bash
./scripts/get_results.sh experiments/my_experiment_20260114_120000
```

This shows:
- Summary (total trials, successful)
- All trials sorted by score
- Best trial code and metrics
