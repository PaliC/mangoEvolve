# LLM-Evolve

LLM-driven evolutionary code generation for optimization problems.

## Overview

This project combines ideas from:
- **AlphaEvolve** (DeepMind): Evolutionary program generation using LLMs
- **Recursive LLMs (RLM)**: Hierarchical LLM spawning with REPL environments

A "Root LLM" orchestrates an evolutionary process, spawning "Child LLMs" to generate candidate programs, evaluating them, and iteratively improving the best candidates.

## Current Target: Circle Packing

Pack 26 circles into a unit square to maximize the sum of their radii.

- **Benchmark**: AlphaEvolve achieved 2.635
- **Best PoC result**: 2.08 (hexagonal packing)
- **Evaluation**: Deterministic, fast (~60ms per trial)

## Architecture

```
Root LLM (REPL) --> spawn_child_llm() --> Child LLM --> Code
        |                                               |
        |                                               v
        |                                        evaluate_program()
        |                                               |
        <----------- metrics, reasoning <---------------+
        |
        v
  advance_generation() / terminate_evolution()
```

## Key Features

- **REPL Environment**: Root LLM writes Python code to control evolution
- **Cost Tracking**: Token usage tracked with budget enforcement
- **Observability**: Full logging of experiments, generations, and trials
- **Pluggable Evaluation**: Easy to swap evaluation functions

## Documentation

- [Circle Packing Design](docs/DESIGN_CIRCLE_PACKING.md) - Current target problem
- [General Design](docs/DESIGN.md) - Detailed architecture
- [Implementation TODO](docs/IMPLEMENTATION_TODO.md) - Task list with dependencies

## Quick Start

```bash
# Install dependencies
uv sync

# Run circle packing evaluator
uv run python -m tetris_evolve.evaluation.circle_packing

# Run full integration PoC
uv run python experiments/poc_circle_packing_integration.py
```

## Proof of Concepts

All PoCs pass and demonstrate the system works:

```bash
# Circle packing full integration (recommended)
uv run python experiments/poc_circle_packing_integration.py

# Core components
uv run python experiments/poc_repl.py
uv run python experiments/poc_cost_tracker.py
uv run python experiments/poc_evaluator.py
uv run python experiments/poc_integration.py
```

## Example Results (from PoC)

```
Testing Circle Packing Evolution Pipeline...
============================================================

Generation 0: Exploring diverse strategies...
  gen0_trial0: VALID, sum_radii=1.7687  (grid)
  gen0_trial1: VALID, sum_radii=2.0800  (hexagonal)
  gen0_trial2: INVALID, sum_radii=0.0000 (greedy - failed)
  gen0_trial3: VALID, sum_radii=1.6512  (corner-first)
  gen0_trial4: VALID, sum_radii=1.0795  (concentric)

Best strategy: hexagonal
  Sum of radii: 2.0800
  Target ratio: 0.7894 (target: 2.635)
```

## Project Status

- [x] Core infrastructure (REPL, cost tracking, evaluation)
- [x] Circle packing evaluator
- [x] Integration PoC with mock LLMs
- [ ] Real LLM integration (Anthropic API)
- [ ] Production logging
- [ ] CLI entry point

## License

MIT
