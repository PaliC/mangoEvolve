"""
Prompt templates for mango_evolve.

Contains template functions that build prompts from ProblemSpec, documenting
available functions and guiding the evolution process. Structured for optimal
prompt caching.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ChildLLMConfig
    from ..problem import ProblemSpec


# Evolution API documentation - problem-agnostic, reused across all problems
EVOLUTION_API_DOCS = '''## Available Functions

### spawn_children(children: list[dict]) -> list[TrialView]
Spawn child LLMs in parallel. Each child dict has:
- `prompt` (str, required) - Use `trials["trial_X_Y"].code` to include code in your prompt
- `parent_id` (str, optional) - set to track lineage when improving a trial
- `model` (str, optional) - alias from available child LLMs
- `temperature` (float, optional, default 0.7)
Returns list of TrialView objects with: trial_id, code, score, success, reasoning, error, etc.

Example:
```python
# Improve the best trial from previous generation
best = trials.filter(success=True, sort_by="-score", limit=1)[0]
results = spawn_children([{
    "prompt": f"Improve this solution (score={best.score}):\\n{best.code}\\nTry to increase the score.",
    "parent_id": best.trial_id
}])

# Combine two approaches
t1, t2 = trials.filter(success=True, sort_by="-score", limit=2)
spawn_children([{
    "prompt": f"Combine these two approaches:\\n# Approach 1 ({t1.score}):\\n{t1.code}\\n\\n# Approach 2 ({t2.score}):\\n{t2.code}"
}])
```

### query_llm(queries: list[dict]) -> list[dict]
Query child LLMs for analysis without code evaluation or trial records. Use this to:
- **Compare trials**: "Why does trial_0_5 score higher than trial_0_3?"
- **Understand methodology**: "What optimization technique is used in this code?"
- **Explore diversity**: "How do these two approaches differ conceptually?"
- **Plan strategy**: "Given these results, what should I try next?"
- **Find patterns**: "What do the top 5 trials have in common?"

Each query dict has:
- `prompt` (str, required) - Use `trials["trial_X_Y"].code` to include code in your prompt
- `model` (str, optional) - alias from available child LLMs
- `temperature` (float, optional, default 0.7)

Returns list of dicts with: model, prompt, response, temperature, success, error.

Examples:
```python
# Analyze top performing trials
top = trials.filter(success=True, sort_by="-score", limit=3)
analysis = query_llm([{
    "prompt": f"Compare these approaches and identify what makes the best one work:\\n" +
              "\\n---\\n".join(f"# {t.trial_id} (score={t.score})\\n{t.code}" for t in top),
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])

# Ask about a specific trial
best = trials.filter(success=True, sort_by="-score", limit=1)[0]
query_llm([{"prompt": f"What optimization technique does this use?\\n{best.code}"}])

# Compare all trials of a generation
generation = trials.filter(generation=0)
analysis = query_llm([{
    "prompt": f"Compare all trials of generation 0 and identify what makes the best ones work vs the rest:\\n" +
              "\\n---\\n".join(f"# {t.trial_id} (score={t.score})\\n{t.code}" for t in generation),
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])

# Find diversity in a generation
generation = trials.filter(generation=0)
analysis = query_llm([{
    "prompt": f"Look throug the various approaches and identify which ones may not be the best, but may be interesting to try again in the future:\\n" +
              "\\n---\\n".join(f"# {t.trial_id} (score={t.score})\\n{t.code}" for t in generation),
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])

# Identify bugs in a trial
trial = trials["trial_0_5"]
analysis = query_llm([{
    "prompt": f"Identify bugs in the following code:\\n{trial.code}",
    "model": "model_alias_of_your_choice"
}])
print(analysis[0]["response"])
```

### `scratchpad` - Persistent notes

A mutable scratchpad for tracking insights across generations:
- `scratchpad.content` - Read current content
- `scratchpad.content = "New notes"` - Replace content (auto-persists)
- `scratchpad.append("\\n## New section")` - Append content
- `scratchpad.clear()` - Clear all content
- `print(scratchpad)` - Print current content
- `"grid" in scratchpad` - Check if text exists
- `len(scratchpad)` - Get character count

The scratchpad is shown in Evolution Memory and persists across generations.

### update_scratchpad(content: str) -> dict
Alternative function to update persistent notes (same as `scratchpad.content = content`).

### terminate_evolution(reason: str, best_program: str = None) -> dict
End evolution early.

## Evolution Flow

1. (Optional) Run analysis in one or more ```python``` blocks
2. When ready, spawn children using the `spawn_children` function with diverse prompts
2. Using the trials variable do some analysis on the results in order to inform your strategy. We recommend at lookingings at all of the results. As the contents of trials may be large, you can use the `query_llm` function to analyze the results. Note you have access and are encouraged to consider all past trials not just the past generation.
3. After spawning, you SELECT which trials to carry forward (performance, diversity, potential)
3. Repeat until max_generations or you call terminate_evolution()

## Selection Format

```selection
{
  "selections": [
    {"trial_id": "trial_0_2", "reasoning": "Best score", "category": "performance"},
    {"trial_id": "trial_0_5", "reasoning": "Different approach", "category": "diversity"}
  ],
  "summary": "Brief summary"
}
```

## Historical Trial Access

**You can access and mutate ANY historical trial**, not just those from the current generation:
- Use the `trials` variable: `trials["trial_0_5"].code` to retrieve code from any past trial

## Custom Analysis Functions

You can define helper functions in Python that persist throughout this evolution session:

```python
# Define a reusable analysis function
def compute_score_stats(scores_list):
    """Compute statistics for a list of scores."""
    import statistics
    return {
        "mean": statistics.mean(scores_list),
        "max": max(scores_list),
        "min": min(scores_list),
        "stdev": statistics.stdev(scores_list) if len(scores_list) > 1 else 0
    }

# Store results for later analysis
generation_bests = []  # Persists across code executions

# Track best scores as you go
generation_bests.append(2.61)  # After gen 0
generation_bests.append(2.62)  # After gen 1
print(compute_score_stats(generation_bests))
```

Available modules: math, random, json, numpy, scipy, collections, itertools, functools, statistics
Functions and variables you define persist across all generations within this run.

## REPL Variables

### `trials` - Query all trials
A live view of all trials across all generations. Use this for flexible analysis.

**Basic access:**
```python
trials["trial_0_5"]  # Get specific trial by ID
len(trials)          # Total trial count
for t in trials: ... # Iterate all trials
"trial_0_5" in trials  # Check if trial exists
```

**Filtering with `trials.filter()`:**
```python
# Top 5 by score
trials.filter(success=True, sort_by="-score", limit=5)

# All from generation 2
trials.filter(generation=2)

# Custom predicate (lambda)
trials.filter(predicate=lambda t: t.score > 2.4 and "grid" in t.reasoning)

# All descendants of a trial
trials.filter(descendant_of="trial_0_3")

# All ancestors of a trial
trials.filter(ancestor_of="trial_2_5")

# Combined: top 3 successful from gen 1
trials.filter(success=True, generation=1, sort_by="-score", limit=3)

# Filter by model
trials.filter(model_alias="fast", success=True)
```

**Filter parameters:**
- `success`: bool - Filter by success/failure
- `generation`: int - Filter by generation number
- `parent_id`: str - Filter by direct parent
- `model_alias`: str - Filter by child LLM model
- `descendant_of`: str - All trials descending from this trial
- `ancestor_of`: str - All ancestors of this trial
- `predicate`: lambda - Custom filter function
- `sort_by`: str - Sort field ("-score" for descending)
- `limit`: int - Max results to return

**Trial attributes:**
`.trial_id`, `.code`, `.score`, `.success`, `.generation`,
`.parent_id`, `.reasoning`, `.error`, `.model_alias`, `.metrics`

**Return values from spawn_children:**
`spawn_children()` returns TrialView objects (same attributes as above).
Use `.to_dict()` if you need dict format.

## Memory Model

**IMPORTANT**: Your conversation history resets at each generation boundary. You start each generation fresh with only:
1. This system prompt
2. Your scratchpad content (shown at start of each generation)
3. The current generation's context

To pass information to future generations, you **MUST** use the scratchpad:
- `scratchpad.append("\\n## Key Insight: ...")` - Add notes
- `scratchpad.content = "..."` - Replace entirely
- `update_scratchpad("...")` - Alternative function form

**What persists across generations:**
1. Your scratchpad content - **this is your primary memory**
2. Functions/variables you define in REPL (e.g., helper functions like `compute_score_stats()`)
3. All trials accessible via `trials.filter(...)` and `trials["trial_X_Y"]`

**What does NOT persist:**
- Your previous messages and reasoning (conversation resets)
- Analysis you did but didn't save to scratchpad

**Best practice**: At the end of each generation, update your scratchpad with:
- Key insights about what approaches work
- Which trials to build on next
- Strategic notes for future generations

Example scratchpad update:
```python
scratchpad.append(\"\"\"
## Generation 2 Insights
- Best score: 2.634 (trial_2_3) using basin-hopping
- Grid approaches plateauing around 2.61
- Next: try combining basin-hopping with different initializations
\"\"\")
```

## Guidelines

**You have full control**: Craft prompts however you see fit - be as specific or open-ended as you want. You're the orchestrator.

**Historical selection is allowed**: You may select any trial_id from any generation (not just the current one).

**Track lineage**: When a child is based on an existing trial, set `parent_id` to that trial_id (choose the primary parent if there are multiple influences).

**Diversity matters**: Especially in early generations, try fundamentally different approaches rather than minor variations of the same idea. If you feel like you are platueing, use diverse tests as well.

**Learn from results**: Use scores and patterns you observe to guide your strategy. If an approach is working, refine it. If you're stuck, try something radically different.

**Use query_llm for analysis**: When you want to understand *why* something works, compare approaches, or get strategic advice, use `query_llm(queries: list[dict])` to analyze trials. This helps you make more informed decisions about what to try next.

**Exploration and Exploitation**: You are encouraged to explore different approaches and to exploit the best ones. Given the amount of trials you are encouraged to allocate some for exploration and exploring ideas with potential or new ideas, and others for refining the best solution(s).
'''

# Dynamic suffix template - appended after the static prefix
ROOT_LLM_SYSTEM_PROMPT_DYNAMIC = """
## Current Run Parameters

- **Max children per generation**: {max_children_per_generation}
- **Max generations**: {max_generations}
- **Current generation**: {current_generation}/{max_generations}
"""

# Available child LLMs template - inserted after dynamic parameters
ROOT_LLM_CHILD_MODELS_TEMPLATE = """
## Available Child LLMs

{child_llm_list}

**Default model**: {default_child_llm}
"""

# Timeout constraint - simple exposure of the limit
ROOT_LLM_TIMEOUT_CONSTRAINT = """
- **Timeout per trial**: {timeout_seconds}s
"""


def _build_problem_section(spec: "ProblemSpec") -> str:
    """Build the problem definition section from ProblemSpec."""
    lines = [
        f"You are an expert algorithm designer. You are orchestrating an evolutionary process to develop algorithms for {spec.name.lower()}.",
        "",
        "## Problem",
        "",
        spec.description,
        "",
    ]

    # Add best known solution if available
    if spec.best_known_solution is not None:
        lines.append(
            f"The best known solution is {spec.best_known_solution}. "
            f"Aim to achieve as {'high' if spec.objective == 'maximize' else 'low'} "
            f"of a {spec.metric_name} as possible."
        )
        lines.append("")

    return "\n".join(lines)


def _build_code_format_section(spec: "ProblemSpec") -> str:
    """Build the code format section from ProblemSpec."""
    lines = ["## Code Format", ""]

    # Entry function
    lines.append(f"Child LLMs must produce code with an entry function `{spec.entry_function}()`.")

    # Helper functions
    if spec.helper_functions:
        helpers = ", ".join(f"`{f}()`" for f in spec.helper_functions)
        lines.append(f"Optional helper functions: {helpers}")

    lines.append("")

    # Return format
    lines.append("**Return format:**")
    lines.append(spec.return_description)
    lines.append("")

    # Allowed modules
    if spec.allowed_modules:
        lines.append(f"**Allowed modules:** {', '.join(spec.allowed_modules)}")
        lines.append("")

    # Constraints
    if spec.constraints:
        lines.append("**Constraints:**")
        for constraint in spec.constraints:
            lines.append(f"- {constraint}")
        lines.append("")

    # Example code
    if spec.example_code:
        lines.append("**Example structure:**")
        lines.append("```python")
        lines.append(spec.example_code)
        lines.append("```")
        lines.append("")

    # Reference code for optimization problems
    if spec.reference_code:
        lines.append("## Reference Implementation")
        lines.append("")
        lines.append("The following is the baseline implementation to optimize:")
        lines.append("```python")
        lines.append(spec.reference_code)
        lines.append("```")
        lines.append("")
        if spec.reference_context:
            lines.append(f"**Context:** {spec.reference_context}")
            lines.append("")

    # Secondary metrics
    if spec.secondary_metrics:
        lines.append(
            f"**Additional metrics tracked:** {', '.join(spec.secondary_metrics)}"
        )
        lines.append("")

    return "\n".join(lines)


def build_root_system_prompt_static(spec: "ProblemSpec") -> str:
    """
    Build the static portion of the Root LLM system prompt from ProblemSpec.

    This contains all the stable content that doesn't change between calls.
    Dynamic values (generation counts) are appended separately.

    Args:
        spec: Problem specification from evaluator

    Returns:
        Static system prompt string
    """
    problem_section = _build_problem_section(spec)
    code_format_section = _build_code_format_section(spec)

    return problem_section + code_format_section + EVOLUTION_API_DOCS


def build_child_system_prompt(spec: "ProblemSpec") -> str:
    """
    Build system prompt for child LLMs from ProblemSpec.

    Minimal prompt to give child LLMs freedom to explore.

    Args:
        spec: Problem specification from evaluator

    Returns:
        Child LLM system prompt string
    """
    lines = [
        f"You are an expert algorithm designer working on: {spec.name}",
        "",
        "## Task",
        "",
        spec.description,
        "",
    ]

    # Best known solution
    if spec.best_known_solution is not None:
        lines.append(
            f"The best known solution is {spec.best_known_solution}. "
            f"Aim to achieve as {'high' if spec.objective == 'maximize' else 'low'} "
            f"of a {spec.metric_name} as possible."
        )
        lines.append("")

    # Reference code for optimization problems
    if spec.reference_code:
        lines.append("## Reference Implementation")
        lines.append("")
        lines.append("```python")
        lines.append(spec.reference_code)
        lines.append("```")
        lines.append("")
        if spec.reference_context:
            lines.append(f"**Context:** {spec.reference_context}")
            lines.append("")

    # Output format
    lines.append("## Output")
    lines.append("")
    lines.append(f"Provide a Python solution with entry function `{spec.entry_function}()`.")
    lines.append("")
    lines.append("**Return format:**")
    lines.append(spec.return_description)
    lines.append("")

    # Allowed modules
    if spec.allowed_modules:
        lines.append(f"You may use: {', '.join(spec.allowed_modules)}.")

    # Constraints
    if spec.constraints:
        lines.append("")
        for constraint in spec.constraints:
            lines.append(f"- {constraint}")

    return "\n".join(lines)


def _format_timeout_constraint(timeout_seconds: int | None) -> str:
    """
    Format timeout constraint for the system prompt.

    Args:
        timeout_seconds: Timeout limit in seconds, or None to omit

    Returns:
        Formatted timeout constraint string, or empty string if None
    """
    if timeout_seconds is None:
        return ""

    return ROOT_LLM_TIMEOUT_CONSTRAINT.format(timeout_seconds=timeout_seconds)


def get_root_system_prompt(
    spec: "ProblemSpec",
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
    timeout_seconds: int | None = None,
) -> str:
    """
    Get the Root LLM system prompt with configuration values.

    Args:
        spec: Problem specification from evaluator
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)
        timeout_seconds: Optional timeout limit per trial in seconds

    Returns:
        Formatted system prompt string (static + dynamic parts combined)
    """
    static_part = build_root_system_prompt_static(spec)
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    timeout_part = _format_timeout_constraint(timeout_seconds)
    return static_part + dynamic_part + timeout_part


def get_root_system_prompt_parts(
    spec: "ProblemSpec",
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
    timeout_seconds: int | None = None,
) -> list[dict]:
    """
    Get the Root LLM system prompt as structured content blocks for caching.

    The static prefix is marked with cache_control for prompt caching.
    The dynamic suffix contains run-specific parameters and timeout constraint.

    Args:
        spec: Problem specification from evaluator
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)
        timeout_seconds: Optional timeout limit per trial in seconds

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
        The first block (static) has cache_control set.
    """
    static_part = build_root_system_prompt_static(spec)
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    timeout_part = _format_timeout_constraint(timeout_seconds)

    return [
        {
            "type": "text",
            "text": static_part,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_part + timeout_part,
        },
    ]


def format_child_mutation_prompt(
    spec: "ProblemSpec",
    parent_code: str,
    parent_score: float,
    guidance: str = "",
) -> str:
    """
    Format a prompt for mutating a parent program.

    This is a helper function - the Root LLM can use this or craft its own.
    Note: The Root LLM is encouraged to write its own shorter, more creative prompts.

    Args:
        spec: Problem specification from evaluator
        parent_code: The parent program code
        parent_score: The parent's score
        guidance: Optional high-level guidance for mutation

    Returns:
        Formatted prompt string
    """
    benchmark_hint = ""
    if spec.best_known_solution is not None:
        benchmark_hint = f" It is possible to achieve a {spec.metric_name} of at least {spec.best_known_solution}."

    prompt = f"""Improve this {spec.name.lower()} solution (current {spec.metric_name}: {parent_score:.16f}).{benchmark_hint}

```python
{parent_code}
```

{guidance if guidance else "Find a way to improve the score."}
"""
    return prompt


def _format_child_llm_list(child_llm_configs: dict[str, "ChildLLMConfig"]) -> str:
    """Format the list of available child LLMs for the system prompt."""
    lines = []
    for alias, cfg in child_llm_configs.items():
        lines.append(
            f"- **{alias}**: `{cfg.model}` ({cfg.provider}) - "
            f"${cfg.cost_per_million_input_tokens:.2f}/M in, "
            f"${cfg.cost_per_million_output_tokens:.2f}/M out"
        )
    return "\n".join(lines)


def get_root_system_prompt_parts_with_models(
    spec: "ProblemSpec",
    child_llm_configs: dict[str, "ChildLLMConfig"],
    default_child_llm_alias: str | None = None,
    max_children_per_generation: int = 10,
    max_generations: int = 10,
    current_generation: int = 0,
    timeout_seconds: int | None = None,
) -> list[dict]:
    """
    Get the Root LLM system prompt with child LLM info as structured content blocks.

    Args:
        spec: Problem specification from evaluator
        child_llm_configs: Dict of alias -> ChildLLMConfig
        default_child_llm_alias: Default model alias
        max_children_per_generation: Maximum children that can be spawned per generation
        max_generations: Maximum number of generations
        current_generation: Current generation number (0-indexed)
        timeout_seconds: Optional timeout limit per trial in seconds

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
    """
    static_part = build_root_system_prompt_static(spec)
    dynamic_part = ROOT_LLM_SYSTEM_PROMPT_DYNAMIC.format(
        max_children_per_generation=max_children_per_generation,
        max_generations=max_generations,
        current_generation=current_generation,
    )
    timeout_part = _format_timeout_constraint(timeout_seconds)

    # Build child LLM info
    child_llm_list = _format_child_llm_list(child_llm_configs)
    default_llm = default_child_llm_alias or (
        list(child_llm_configs.keys())[0] if child_llm_configs else "none"
    )
    child_models_part = ROOT_LLM_CHILD_MODELS_TEMPLATE.format(
        child_llm_list=child_llm_list,
        default_child_llm=default_llm,
    )

    return [
        {
            "type": "text",
            "text": static_part,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_part + timeout_part + child_models_part,
        },
    ]


# Calibration system prompt - used during calibration phase
def _build_calibration_problem_reference(spec: "ProblemSpec") -> str:
    """Build problem reference for calibration prompt."""
    lines = [
        "## Problem (for reference)",
        "",
        f"The main task is {spec.name.lower()}:",
        spec.description,
    ]
    if spec.best_known_solution is not None:
        lines.append(
            f"- The best known solution is {spec.best_known_solution}. "
            f"Aim to achieve as {'high' if spec.objective == 'maximize' else 'low'} "
            f"of a {spec.metric_name} as possible."
        )
    return "\n".join(lines)


CALIBRATION_SYSTEM_PROMPT_TEMPLATE = """You are orchestrating a calibration phase for an evolutionary optimization process.

## Purpose

Before evolution begins, you have the opportunity to test the available child LLMs to understand their capabilities. **You can send ANY prompt you want** - not just {problem_name} tasks. Use this to evaluate:

- Reasoning depth and quality (ask them to explain their approach)
- Code style and correctness (test with simple problems first)
- Mathematical reasoning (geometry, optimization concepts)
- Instruction following (give specific constraints)
- Creativity vs precision tradeoffs at different temperatures
- How they handle ambiguous or open-ended prompts

The goal is to understand what each model is good at so you can use them strategically during evolution.

{problem_reference}

## How to Call Functions

**IMPORTANT**: Write Python code in ```python blocks to call functions. Example:

```python
query_llm([
    {{"prompt": "Explain quicksort", "model": "sonnet", "temperature": 0.5}},
    {{"prompt": "What is 2+2?", "model": "gpt41", "temperature": 0.3}}
])
```

## Available Functions

### query_llm(queries: list[dict]) -> list[dict]
Query child LLMs with ANY prompts (no code evaluation - just get responses). Each dict has:
- `prompt` (str, required) - Any question or task, not limited to {problem_name}!
- `model` (str, optional) - alias from available child LLMs
- `temperature` (float, optional, default 0.7)

Returns list of dicts with: model, prompt, response, temperature, success, error.

Example:
```python
results = query_llm([
    {{"prompt": "Explain gradient descent in 2 sentences", "model": "sonnet"}},
    {{"prompt": "Write a Python function to compute factorial", "model": "gpt41"}}
])
for r in results:
    print(f"{{r['model']}}: {{r['response'][:200]}}")
```

### get_calibration_status() -> dict
Check remaining calibration calls per model.

```python
get_calibration_status()
```

### update_scratchpad(content: str) -> dict
Record your observations about each model's behavior. These notes will persist into evolution.

```python
update_scratchpad(\"\"\"
## Model Observations
- sonnet: Strong reasoning, verbose responses
- gpt41: Concise, good at math
\"\"\")
```

### end_calibration_phase() -> dict
Finish calibration and begin the evolution phase. Call this when you've learned enough.

```python
end_calibration_phase()
```

## Guidelines

1. **Ask diverse questions**: Test reasoning, math, code quality - not just {problem_name}. Your notes should be generic and not specific to the {problem_name} task.
2. **Compare models**: Give the same prompt to different models to compare their responses
3. **Experiment with temperatures**: Generally 0 is considered the most focused / reproducible, 1 is the most creative.
4. **Record detailed observations**: Note strengths/weaknesses of each model
5. **Be strategic**: Your notes will guide which model you choose for different tasks during evolution
"""


def get_calibration_system_prompt_parts(
    spec: "ProblemSpec",
    child_llm_configs: dict[str, "ChildLLMConfig"],
) -> list[dict]:
    """
    Get the calibration system prompt as structured content blocks.

    Args:
        spec: Problem specification from evaluator
        child_llm_configs: Dict of alias -> ChildLLMConfig

    Returns:
        List of content blocks suitable for Anthropic API system parameter.
    """
    problem_reference = _build_calibration_problem_reference(spec)
    static_part = CALIBRATION_SYSTEM_PROMPT_TEMPLATE.format(
        problem_name=spec.name.lower(),
        problem_reference=problem_reference,
    )

    # Build child LLM info with calibration budgets
    lines = ["## Available Child LLMs", ""]
    for alias, cfg in child_llm_configs.items():
        lines.extend(
            [
                f"### {alias}",
                f"- **Model**: `{cfg.model}`",
                f"- **Provider**: {cfg.provider}",
                f"- **Calibration budget**: {cfg.calibration_calls} calls",
                f"- **Cost**: ${cfg.cost_per_million_input_tokens:.2f}/M input, "
                f"${cfg.cost_per_million_output_tokens:.2f}/M output",
                "",
            ]
        )

    child_models_part = "\n".join(lines)

    return [
        {
            "type": "text",
            "text": static_part,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": child_models_part,
        },
    ]


# Legacy compatibility - keep old constants for backwards compatibility during migration
# These will be removed after all callers are updated to use the new functions
ROOT_LLM_SYSTEM_PROMPT_STATIC = """[DEPRECATED] Use build_root_system_prompt_static(spec) instead."""
CHILD_LLM_SYSTEM_PROMPT = """[DEPRECATED] Use build_child_system_prompt(spec) instead."""
