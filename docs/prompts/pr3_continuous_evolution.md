# Task: Refactor to Continuous Evolution Loop

## Objective
Remove the rigid generation-based advancement pattern and allow continuous evolution where trials are added to the database immediately. The Root LLM controls its own exploration strategy via the REPL.

## Context
Currently, MangoEvolve forces a pattern:
1. Spawn N children
2. Request selection from Root LLM
3. Advance generation
4. Repeat

AlphaEvolve uses continuous evolution - trials are evaluated and added to the database immediately, with no forced boundaries.

## Requirements

### 1. Simplify the Main Loop

Refactor `RootLLMOrchestrator.run()` to:

```python
def run(self) -> OrchestratorResult:
    """Run continuous evolution loop."""
    # Calibration phase (unchanged)
    ...

    # Build initial message
    self.messages = [{
        "role": "user",
        "content": self._build_initial_prompt()
    }]

    # Simple loop: call LLM, execute code, repeat
    while not self._should_terminate():
        # Call Root LLM
        response = self.root_llm.generate(
            messages=self._prepare_messages_with_caching(self.messages),
            system=self._get_system_prompt(),
            max_tokens=4096,
            temperature=0.7,
        )

        self._append_and_log("assistant", response.content)

        # Execute code blocks
        code_blocks = self.extract_code_blocks(response.content)
        execution_results = []
        for code in code_blocks:
            result = self.execute_code_in_repl(code)
            execution_results.append(f"```\n{code}\n```\n\nResult:\n{result}")
            self._log_execution(code, result)

        # Check for termination
        if self.evolution_api.is_terminated:
            break

        # Build feedback message
        feedback = self._build_feedback_message(execution_results)
        self._append_and_log("user", feedback)

        # Prune history if needed
        self._prune_message_history()

    return self._build_result()
```

### 2. Remove Generation-Based Logic

**Remove or simplify:**
- `_advance_generation()` - No longer needed for forced advancement
- `_build_selection_request_message()` - No forced selection
- `_handle_trial_selection()` - No forced selection
- `GenerationSummary.selected_trial_ids` - Keep for logging but don't require
- The concept of "current_generation" - Keep for organization/logging but don't enforce

**Keep:**
- `self.evolution_api.current_generation` - Useful for organizing trials
- Generation folders in output - Good for observability
- `_build_evolution_memory()` - Still useful context

### 3. Add Checkpoint/Milestone Concept (Optional)

Instead of forced generations, allow the LLM to create explicit checkpoints:

```python
def checkpoint(self, name: str = None) -> dict:
    """
    Create a checkpoint of current evolution state.

    Useful for marking milestones, switching strategies, or
    organizing the trial history.
    """
    self.current_generation += 1
    checkpoint_name = name or f"checkpoint_{self.current_generation}"

    # Log the checkpoint
    self.logger.log_checkpoint(
        generation=self.current_generation,
        name=checkpoint_name,
        best_trial=self._get_best_trial(),
        total_trials=len(self.all_trials),
    )

    # Create new generation folder
    self.generations.append(GenerationSummary(generation_num=self.current_generation, trials=[]))

    return {"checkpoint": checkpoint_name, "generation": self.current_generation}
```

### 4. Update System Prompt

Remove references to "spawn N children per generation" and emphasize continuous exploration:

```python
"""
## Evolution Loop

You are in a continuous evolution loop. Each iteration:
1. Analyze current trials using `trials` variable
2. Decide what to explore next
3. Spawn children with `spawn_child_llm()` or `spawn_children_parallel()`
4. Update `scratchpad` with insights
5. Repeat until you achieve the target or run out of budget

There are no forced generation boundaries. You control the exploration strategy.

Use `checkpoint("milestone_name")` to mark significant progress points.
Use `terminate_evolution(reason)` when done.

### Budget
- Max iterations: {max_iterations}
- Max total trials: {max_trials}
- Cost budget: ${budget}
"""
```

### 5. Simplify Trial ID Scheme

Instead of `trial_{gen}_{num}`, use sequential IDs:

```python
# Old: trial_0_5, trial_1_3, trial_2_0
# New: trial_0, trial_1, trial_2, ... trial_47
trial_id = f"trial_{len(self.all_trials)}"
```

Or keep generation-based naming but auto-increment generation on checkpoint.

### 6. Maintain Observability

**Critical**: Keep all existing logging and file outputs:
- Trial JSON files: `generations/gen_X/trial_X_Y.json`
- Scratchpad files: `generations/gen_X/scratchpad.json`
- Root LLM logs: `root_llm_log.jsonl`
- Cost tracking: `cost_tracking.json`
- Experiment summary: `experiment.json`

The generation folders become checkpoints. Trials are written to the current checkpoint folder.

### 7. Update Feedback Message

Simplify to just show execution results and current state:

```python
def _build_feedback_message(self, execution_results: list[str]) -> str:
    lines = []

    if execution_results:
        lines.append("## Execution Results")
        lines.extend(execution_results)
        lines.append("")

    # Current state summary
    lines.append("## Current State")
    lines.append(f"- Total trials: {len(self.evolution_api.all_trials)}")
    lines.append(f"- Successful: {sum(1 for t in trials if t.success)}")
    best = self.evolution_api._get_best_trials(1)
    if best:
        lines.append(f"- Best score: {best[0]['metrics'].get('score', 0):.10f}")
    lines.append(f"- Budget remaining: {self._get_remaining_budget()}")
    lines.append("")

    # Include evolution memory
    lines.append(self._build_evolution_memory())

    return "\n".join(lines)
```

### 8. Remove/Deprecate

Consider removing or marking as deprecated:
- `max_children_per_generation` config - No longer enforced
- `_auto_select_trials()` - No forced selection
- Selection parsing code in `root_llm.py`

Keep `max_generations` as a way to limit checkpoints if desired.

## Files to Modify
- `src/mango_evolve/root_llm.py` - Major refactor of run loop
- `src/mango_evolve/evolution_api.py` - Add checkpoint(), simplify advancement
- `src/mango_evolve/config.py` - Update config options
- `src/mango_evolve/llm/prompts.py` - Update system prompt
- `tests/test_root_llm.py` - Update for new flow
- `tests/test_e2e.py` - Update E2E tests

## Acceptance Criteria
1. Evolution runs without forced generation advancement
2. Root LLM can spawn trials freely until budget exhausted
3. `checkpoint()` creates explicit milestones
4. All trial files still written to disk
5. Scratchpad persistence unchanged
6. Cost tracking unchanged
7. Experiment summary includes all trials
8. Existing E2E tests adapted to new flow

## Migration Notes
- Old configs with `max_children_per_generation` should still work (just not enforced)
- Add `max_total_trials` as new config option
- Document the change in README
