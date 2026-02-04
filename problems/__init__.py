"""Problem definitions for MangoEvolve.

Each problem is a self-contained folder with:
- evaluator.py: Problem evaluator extending BaseProblemEvaluator
- configs/: YAML configuration files for running the problem
- experiments/: Output directory for experiment runs (gitignored)

Example structure:
    problems/
        circle_packing/
            __init__.py
            evaluator.py
            configs/
                sonnet.yaml
                opus_thinking.yaml
            experiments/

To add a new problem:
1. Create a folder under problems/
2. Implement an evaluator extending BaseProblemEvaluator
3. Create config files pointing to your evaluator
4. Run with: uv run python -m mango_evolve --config problems/my_problem/configs/my_config.yaml
"""
