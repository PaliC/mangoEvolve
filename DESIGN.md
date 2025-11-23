# LLM Tetris AlphaEvolve Design Document

## Executive Summary

This system combines **AlphaEvolve's evolutionary coding approach** with **Recursive LLM (RLM) hierarchical decision-making** to evolve code that plays Tetris optimally. The Root LLM acts as an evolutionary strategist, selecting promising candidates and guiding evolution, while Recursive Child LLMs generate code mutations and implement new variants.

## Core Concepts

### 1. AlphaEvolve Integration
- **Evolutionary Framework**: Maintain a program database with generations of Tetris-playing algorithms
- **Automated Evaluation**: Each candidate is scored by running Tetris simulations (lines cleared, survival time, score)
- **Iterative Improvement**: High-performing programs inform future generations
- **Dual Strategy**: Use both exploration (many diverse ideas) and exploitation (refine best candidates)

### 2. Recursive LLM Integration
- **Root LLM (Depth=0)**: Strategic decision-maker that:
  - Analyzes performance data across all candidates in current generation
  - **Dynamically decides how many programs advance to next generation** (not hardcoded)
  - **Assigns same solution to multiple RLLMs to explore different improvement directions**
  - **Decides specific focus areas for each Recursive LLM** (e.g., "improve hole management", "optimize piece placement scoring", "increase lookahead depth")
  - Provides rich context to Child LLMs about what to optimize
- **Child LLMs (Depth=1+)**: Code generators that:
  - Receive parent programs + strategic guidance from Root
  - Generate new code variants based on Root's directives
  - Apply specific mutations/improvements
  - Return generated code to Root for evaluation

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                      ROOT LLM                           │
│  (Strategic Decision Maker - Depth 0)                   │
│                                                          │
│  Responsibilities:                                       │
│  • Analyze generation performance metrics               │
│  • Dynamically decide N candidates for next generation  │
│  • Assign same solution to multiple RLLMs               │
│  • Craft specific focus areas for each RLLM             │
│  • Identify improvement strategies                      │
│  • Maintain evolutionary memory/insights                │
└─────────────┬───────────────────────────────────────────┘
              │
              │ Spawns Child LLMs with:
              │ - Parent program code
              │ - Mutation strategy
              │ - Performance context
              │ - Specific optimization goals
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              CHILD LLMs (Parallel)                      │
│         (Code Generators - Depth 1)                      │
│                                                          │
│  Child 1         Child 2         Child 3                │
│  Parent: Prog A  Parent: Prog A  Parent: Prog B        │
│  Focus: holes    Focus: speed    Focus: lookahead      │
│  (Same parent, different improvement directions)        │
└─────────────┬───────────────────────────────────────────┘
              │
              │ Return generated code variants
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              EVALUATION ENGINE                          │
│                                                          │
│  • Run Tetris simulations for each variant             │
│  • Collect metrics (score, lines, survival time)       │
│  • Store results in Program Database                   │
└─────────────┬───────────────────────────────────────────┘
              │
              │ Metrics feed back to Root LLM
              │
              ▼
           [Next Generation]
```

### Component Architecture

#### 1. Program Database
```python
{
  "generation": int,
  "program_id": str,
  "code": str,
  "parent_ids": [str],
  "mutation_strategy": str,
  "metrics": {
    "avg_score": float,
    "avg_lines_cleared": float,
    "avg_survival_time": float,
    "games_played": int,
    "std_dev": float
  },
  "metadata": {
    "created_at": timestamp,
    "root_llm_notes": str,
    "child_llm_id": str
  }
}
```

#### 2. Root LLM Environment (REPL-based)

The Root LLM operates in a Python REPL environment with access to:

```python
# Available in Root LLM context
current_generation = 5
program_database = [...] # All programs from all generations
current_gen_programs = [...] # Programs from current generation only
metrics_summary = {...} # Statistical analysis of current generation

# Functions available to Root LLM
def spawn_rllm(parent_program, focus_area, mutation_directive, context):
    """
    Spawn a Recursive Child LLM to generate new code variant.
    Can spawn multiple RLLMs for the same parent with different focus areas.

    Args:
        parent_program: The program to mutate
        focus_area: Specific aspect to improve (e.g., "hole_management", "lookahead", "speed")
        mutation_directive: Detailed guidance on how to improve
        context: Additional context about generation and top performers
    """
    pass

def evaluate_program(code, num_games=100):
    """Run Tetris simulations and return metrics"""
    pass

def get_performance_analysis(generation):
    """Get detailed analysis of a generation's performance"""
    pass

def advance_generation(selected_programs):
    """
    Move to next generation with dynamically selected programs.
    Number of programs is decided by Root LLM, not hardcoded.
    """
    pass
```

The Root LLM can:
- Inspect performance data
- Identify patterns in successful programs
- Decide selection criteria dynamically (not hardcoded)
- Craft contextual mutation directives
- Spawn multiple Child LLMs in parallel

#### 3. Child LLM Interface

Each Child LLM receives:

```python
{
  "task": "generate_variant",
  "parent_program": {
    "code": str,
    "metrics": {...},
    "generation": int
  },
  "mutation_directive": {
    "strategy": "improve_hole_management",  # or other strategy
    "guidance": "The parent program leaves too many holes. Focus on...",
    "constraints": ["maintain_performance", "keep_structure"],
    "examples": [...]  # Optional: similar successful mutations
  },
  "context": {
    "top_performers": [...],  # For reference
    "common_patterns": [...],
    "generation_insights": str
  }
}
```

Child LLM outputs:
```python
{
  "code": str,
  "explanation": str,
  "expected_improvements": [str]
}
```

## Evolutionary Strategy

### Generation Lifecycle

1. **Initialization (Generation 0)**
   - Root LLM generates initial diverse population of Tetris players
   - Strategies: random placement, greedy scoring, hole minimization, etc.
   - Population size: 20-50 programs

2. **Evaluation Phase**
   - Run each program through N Tetris games (e.g., 100 games)
   - Collect comprehensive metrics
   - Store in Program Database

3. **Selection Phase (Root LLM Decision)**
   - Root LLM analyzes all metrics
   - **Dynamically decides N programs to advance** (not hardcoded K)
   - Discovers patterns in successful programs
   - Decides selection criteria (may vary by generation)
   - **For promising programs, assigns multiple RLLMs with different focus areas**
   - Example Root reasoning:
     ```
     "Generation 5 shows that programs focusing on minimizing
     holes outperform greedy scorers by 40%. The top 3 programs
     all use lookahead depth of 2. I'll select the top 8
     performers. For the best program, I'll spawn 3 RLLMs:
     one to explore deeper lookahead, one to optimize hole
     management further, and one to improve speed. For programs
     ranked 2-5, I'll spawn 2 RLLMs each with different focuses."
     ```

4. **Mutation Phase (Recursive Child LLMs)**
   - Root spawns Recursive Child LLMs with specific focus areas:
     - **Multiple RLLMs per promising solution**: Same parent, different improvement directions
     - **Exploitation**: Mutate top performers with targeted improvements
     - **Exploration**: Create novel variants with different approaches
     - **Crossover**: Combine features from multiple top performers
   - Each Child RLLM generates 1-3 variants based on its specific focus
   - Parallel generation for efficiency
   - Example: Best program → RLLM1 (focus: holes), RLLM2 (focus: lookahead), RLLM3 (focus: speed)

5. **Population Assembly**
   - Root collects all Child outputs
   - Optionally filters obviously broken code
   - Combines: selected parents + new children
   - Population maintained at ~20-50 programs

6. **Iteration**
   - Increment generation counter
   - Return to Evaluation Phase
   - Continue until convergence or iteration limit

### Mutation Strategies

The Root LLM can direct various mutation types:

1. **Targeted Improvement**
   - "Reduce hole creation in the current top performer"
   - "Optimize the scoring function for long-term stability"

2. **Feature Addition**
   - "Add piece lookahead to this program"
   - "Implement hold piece optimization"

3. **Algorithmic Shift**
   - "Convert this greedy approach to use beam search"
   - "Add Monte Carlo tree search exploration"

4. **Crossover/Recombination**
   - "Combine the hole management from Program A with the scoring from Program B"

5. **Exploration**
   - "Create a completely new approach using genetic algorithms"
   - "Try a reinforcement learning-inspired value function"

## Tetris Environment

### PufferLib Integration

We use **PufferLib** for the Tetris environment, which provides a gymnasium-compatible interface optimized for RL environments. PufferLib handles vectorized environments and efficient parallelization.

```python
import pufferlib
import gymnasium as gym

# PufferLib provides Tetris environment (or we create a custom one)
# The environment follows gymnasium API: reset(), step(action), etc.

class TetrisEnvironmentWrapper:
    """Wrapper around PufferLib Tetris environment"""

    def __init__(self, width=10, height=20, num_envs=1):
        # Initialize PufferLib vectorized environment
        self.env = pufferlib.vector.make(
            "Tetris-v0",  # or custom Tetris environment
            num_envs=num_envs,
            # ... configuration
        )

    def reset(self):
        """Reset environment and return initial observation"""
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        return self.env.step(action)

    def get_state_dict(self, obs):
        """Convert observation to human-readable state dict"""
        return {
            "board": np.array,  # 2D grid
            "current_piece": Piece,
            "next_pieces": [Piece],  # lookahead
            "hold_piece": Piece | None,
            "score": int,
            "lines_cleared": int
        }
```

**Note**: If PufferLib doesn't have a built-in Tetris environment, we'll create a custom gymnasium-compatible Tetris environment and register it with PufferLib for vectorized execution.

### Player Interface (What Generated Code Implements)

```python
class TetrisPlayer:
    def select_action(self, game_state):
        """
        Given current game state, return best action.

        Args:
            game_state: dict with board, current_piece, next_pieces, etc.

        Returns:
            action: dict with rotation and column
        """
        # Generated code implements this logic
        pass
```

### Evaluation Metrics

```python
def evaluate_player(player_code, num_games=100):
    """
    Run player through multiple games and collect metrics.

    Returns:
        {
            "avg_score": float,
            "avg_lines_cleared": float,
            "avg_survival_time": float,
            "max_score": int,
            "max_lines": int,
            "std_dev_score": float,
            "std_dev_lines": float,
            "success_rate": float,  # games lasting > threshold
            "code_errors": int  # runtime errors
        }
    """
    pass
```

## Implementation Phases

**NOTE: Each phase follows Test-Driven Development (TDD). Write tests first, then implement to pass tests.**

### Phase 0: Project Setup
**Setup Tasks:**
- [ ] Create project directory structure (src, tests, configs, etc.)
- [ ] Set up `pyproject.toml` with dependencies
- [ ] Install PufferLib and verify installation
- [ ] Research PufferLib Tetris environment availability
- [ ] Set up pytest configuration and test structure
- [ ] Create initial configuration files (config.yaml)
- [ ] Set up logging configuration
- [ ] Initialize git repository structure
- [ ] Create README with setup instructions

**Directory Structure:**
```
tetris_evolve/
├── src/
│   ├── tetris_evolve/
│   │   ├── __init__.py
│   │   ├── environment/       # Tetris environment wrapper
│   │   ├── evaluation/        # Player evaluation framework
│   │   ├── database/          # Program database
│   │   ├── rlm/               # Recursive LLM framework
│   │   ├── root_llm/          # Root LLM logic
│   │   ├── child_llm/         # Child RLLM logic
│   │   └── evolution/         # Evolution loop
├── tests/
│   ├── test_environment/
│   ├── test_evaluation/
│   ├── test_database/
│   ├── test_rlm/
│   └── test_evolution/
├── configs/
│   └── config.yaml
├── pyproject.toml
├── DESIGN.md
└── README.md
```

### Phase 1: Core Infrastructure
**Tests First:**
- [ ] Write tests for PufferLib Tetris environment wrapper
- [ ] Write tests for player evaluation framework
- [ ] Write tests for program database CRUD operations
- [ ] Write tests for basic REPL environment execution

**Implementation:**
- [ ] Integrate PufferLib Tetris environment
- [ ] Implement evaluation framework (runs games, collects metrics)
- [ ] Create program database (SQLite schema + API)
- [ ] Build basic RLM framework (REPL environment)
- [ ] Verify all tests pass

**Deliverables:**
- Working Tetris environment wrapper with tests
- Evaluation framework that can run games and collect metrics
- Program database with CRUD operations
- Basic REPL execution environment
- All tests passing

### Phase 2: Root LLM Integration
**Tests First:**
- [ ] Write tests for Root LLM REPL function injection
- [ ] Write tests for performance analysis functions
- [ ] Write tests for dynamic selection (variable N programs)
- [ ] Write tests for multi-RLLM assignment to same parent

**Implementation:**
- [ ] Implement Root LLM REPL environment with injected functions
- [ ] Add functions for program analysis and metrics
- [ ] Build generation management system
- [ ] Create prompt templates emphasizing dynamic decisions
- [ ] Test Root's ability to select N programs and assign multiple RLLMs
- [ ] Verify all tests pass

**Deliverables:**
- Root LLM REPL environment with function injection
- Performance analysis and metrics functions
- Generation management system
- Dynamic selection working (variable N programs)
- Multi-RLLM assignment capability
- All tests passing

### Phase 3: Recursive Child LLM Integration
**Tests First:**
- [ ] Write tests for RLLM spawning with focus areas
- [ ] Write tests for mutation directive creation
- [ ] Write tests for code generation pipeline
- [ ] Write tests for code validation and safety
- [ ] Write tests for parallel RLLM execution

**Implementation:**
- [ ] Implement RLLM spawning with focus_area parameter
- [ ] Create mutation directive templates
- [ ] Build code generation pipeline
- [ ] Add code validation and safety checks
- [ ] Implement parallel RLLM generation
- [ ] Verify all tests pass

**Deliverables:**
- RLLM spawning system with focus_area parameter
- Mutation directive templates
- Code generation and validation pipeline
- Parallel RLLM execution
- All tests passing

### Phase 4: Evolution Loop
**Tests First:**
- [ ] Write tests for full generation lifecycle
- [ ] Write tests for Root's dynamic decision-making
- [ ] Write tests for checkpoint save/restore
- [ ] Write tests for edge cases (no improvements, code errors, etc.)

**Implementation:**
- [ ] Connect all components into evolution loop
- [ ] Implement full generation lifecycle
- [ ] Add comprehensive logging and visualization
- [ ] Create checkpoint/resume functionality
- [ ] Run initial evolution experiments
- [ ] Verify all tests pass

**Deliverables:**
- Complete end-to-end evolution system
- Full generation lifecycle working
- Logging and visualization
- Checkpoint/resume functionality
- Initial evolution experiments completed
- All tests passing

### Phase 5: Optimization & Scaling
**Tests First:**
- [ ] Write tests for parallel evaluation
- [ ] Write tests for evaluation caching
- [ ] Write tests for diversity metrics

**Implementation:**
- [ ] Parallelize game evaluation
- [ ] Optimize LLM token usage
- [ ] Add caching for repeated evaluations
- [ ] Implement diversity preservation mechanisms
- [ ] Performance profiling and optimization
- [ ] Verify all tests pass

**Deliverables:**
- Parallel evaluation system
- Evaluation caching
- Diversity preservation mechanisms
- Performance profiling results
- Optimized token usage
- All tests passing

## Technical Stack

### Core Components
- **Language**: Python 3.10+
- **LLM API**: OpenAI API (GPT-4 for Root, GPT-3.5/4 for Children) or Anthropic Claude
- **Game Engine**: PufferLib (gymnasium-compatible RL environment library)
- **Database**: SQLite for program storage
- **REPL**: Python `exec()` based sandbox (from RLM framework)
- **Testing**: pytest with test-driven development (TDD) approach

### Libraries
```
- pufferlib: Tetris game environment (gymnasium-compatible)
- numpy: Array operations and state management
- openai / anthropic: LLM API calls
- sqlite3: Program database
- multiprocessing: Parallel evaluation
- matplotlib/plotly: Visualization
- pytest: Test framework (TDD approach)
- pydantic: Data validation
- gymnasium: RL environment interface (used by PufferLib)
```

## Configuration

### System Parameters
```yaml
evolution:
  initial_population_size: 30
  num_generations: 100
  games_per_evaluation: 100
  # Note: selection size is dynamic, decided by Root LLM each generation

  # Guidance for Root LLM (not strict constraints)
  suggested_selection_range: [5, 15]  # Root can choose outside this range
  suggested_rllms_per_top_program: [2, 4]  # Multiple RLLMs per promising solution

  mutation_distribution_guidance:  # Suggestions, not requirements
    exploitation: 0.6  # Improve top performers
    exploration: 0.3   # Novel approaches
    crossover: 0.1     # Recombination

llm:
  root_model: "gpt-4"
  child_model: "gpt-4"  # or "gpt-3.5-turbo" for cost
  temperature_root: 0.7
  temperature_child: 0.8
  max_tokens: 4000

tetris:
  board_width: 10
  board_height: 20
  preview_pieces: 5  # Number of next pieces visible
  enable_hold: true

evaluation:
  num_games: 100
  parallel_games: 10
  timeout_per_game: 300  # seconds
  max_moves_per_game: 10000
```

## Root LLM Prompting Strategy

### Initial Prompt Template
```
You are the Root LLM in an evolutionary system designed to evolve optimal
Tetris-playing code. You have access to a program database containing all
programs from previous generations, their performance metrics, and metadata.

Your responsibilities:
1. Analyze the current generation's performance
2. Select the best programs to advance
3. Identify patterns in successful programs
4. Design mutation strategies for the next generation
5. Spawn Child LLMs with specific directives to generate new code

Current Generation: {generation}

Available Functions:
- spawn_rllm(parent_program, focus_area, mutation_directive, context)
- evaluate_program(code, num_games)
- get_performance_analysis(generation)
- advance_generation(selected_programs)

Current Generation Data:
{program_data}

Performance Summary:
{metrics_summary}

Please analyze this generation and decide:
1. How many programs should advance to the next generation?
2. Which specific programs should advance?
3. For each selected program, which focus areas should RLLMs explore?
4. Should any program have multiple RLLMs working on different improvements?

Think step-by-step and use the available functions to implement your strategy.
You have full control over selection size and RLLM assignments.
```

### Recursive Child LLM Prompt Template
```
You are a Recursive Child LLM specialized in generating Tetris-playing code.
You have been assigned a specific focus area by the Root LLM.

**IMPORTANT**: You may be one of multiple RLLMs working on the SAME parent program.
Each RLLM has a different focus area. Your job is to explore YOUR specific focus.

Parent Program (Generation {gen}, Score: {score}):
{parent_code}

Parent Performance:
{parent_metrics}

Your Focus Area: {focus_area}
Examples: "hole_management", "lookahead_depth", "speed_optimization", "piece_placement_scoring"

Mutation Directive for YOUR focus:
Strategy: {strategy}
Guidance: {guidance}
Constraints: {constraints}

Context:
{context}

Your task:
Generate an improved version of the parent program focusing SPECIFICALLY on {focus_area}.
Other RLLMs may be exploring different aspects of the same parent.

Your code must implement the TetrisPlayer interface:
- select_action(game_state) -> action

Return your generated code and a brief explanation of changes related to your focus area.
```

## Success Metrics

### System Performance
- **Convergence**: Improvement in average score over generations
- **Diversity**: Variety of approaches in population
- **Efficiency**: Time per generation, cost per generation
- **Code Quality**: Syntactic correctness, runtime stability

### Tetris Performance (Target Goals)
- **Lines Cleared**: >500 lines average per game
- **Survival Time**: >5000 moves per game
- **Score**: Context-dependent, aim for top 1% of baseline algorithms

### Evolution Quality
- **Innovation**: Novel strategies discovered
- **Robustness**: Performance across different game seeds
- **Interpretability**: Understandable strategies in evolved code

## Risk Mitigation

### Code Safety
- **Sandboxing**: Execute generated code in isolated environment
- **Timeout**: Limit execution time per move and per game
- **Resource Limits**: Cap memory and CPU usage
- **Validation**: Parse and check code before execution

### Cost Control
- **Token Budgets**: Limit per generation and total
- **Model Selection**: Use cheaper models for Child LLMs when possible
- **Caching**: Reuse evaluations for identical code
- **Early Stopping**: Halt evolution if no improvement after N generations

### Quality Assurance
- **Baseline Comparison**: Maintain hand-written baseline players
- **Regression Testing**: Ensure new generations don't catastrophically fail
- **Human Review**: Periodic inspection of evolved strategies
- **Checkpointing**: Save state frequently for recovery

## Extensions & Future Work

### Short-term Enhancements
1. **Multi-objective Optimization**: Balance score, style, code simplicity
2. **Transfer Learning**: Use insights from Tetris for other games
3. **Interactive Evolution**: Allow human guidance/feedback
4. **Visualization Dashboard**: Real-time evolution monitoring

### Long-term Research Directions
1. **Self-Modifying Evolution**: Root LLM modifies its own selection strategy
2. **Hierarchical Specialization**: Multi-level RLM tree (depth > 1)
3. **Meta-Learning**: Learn mutation strategies that work best
4. **Curriculum Learning**: Start with simplified Tetris, increase difficulty
5. **Ensemble Players**: Combine multiple evolved programs

## Conclusion

This system uniquely combines evolutionary coding (AlphaEvolve) with hierarchical
LLM decision-making (RLM) to evolve Tetris-playing code. The Root LLM provides
strategic intelligence for evolution, while Child LLMs provide creative code
generation. This separation of concerns leverages each LLM's strengths:

- **Root LLM**: Strategic thinking, pattern recognition, long-term planning
- **Child LLMs**: Creative coding, mutation implementation, diverse exploration

The result is an evolution system that is both more intelligent in selection
and more creative in generation than traditional evolutionary algorithms or
single-LLM approaches.
