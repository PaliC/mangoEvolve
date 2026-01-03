# Task: Plan Mode - Allow Root LLM to Query Child LLMs Without Side Effects

## Enter Plan Mode

I need to design a feature that allows the Root LLM to call child LLMs for auxiliary queries (not for creating trial programs) without affecting the generation state or trial count.

## Use Cases

1. **Ask for analysis**: Root LLM wants a child LLM to analyze why certain trials failed
2. **Generate prompts**: Root LLM wants help crafting better mutation prompts
3. **Brainstorm**: Root LLM wants ideas for new approaches without committing to trials
4. **Code review**: Root LLM wants feedback on a specific trial's code
5. **Summarization**: Root LLM wants a child LLM to summarize patterns across trials

## Design Questions to Explore

### 1. API Design
What should the function signature look like?

Options:
- `query_child_llm(prompt, model=None)` - Simple query, returns response text
- `llm_call(prompt, model=None, purpose="query")` - Explicit purpose parameter
- `child_llm.query(prompt)` vs `child_llm.spawn(prompt)` - Separate object

### 2. Cost Tracking
How should these queries be tracked?
- Separate budget category: `query_budget` vs `spawn_budget`?
- Same budget but labeled differently in cost tracking?
- No budget limit for queries (dangerous)?

### 3. Logging
Where should query responses be logged?
- New file: `child_llm_queries.jsonl`?
- Same root LLM log with a "query" type?
- In the scratchpad (LLM can save useful responses)?

### 4. Observability
How do we maintain visibility into these queries?
- Log to terminal with different prefix?
- Include in experiment summary?
- Track query count separately?

### 5. Rate Limiting
Should we limit query frequency?
- Max queries per generation/iteration?
- Delay between queries?
- Just rely on cost budget?

### 6. Response Format
Should the response be structured?
- Raw text response
- Parsed JSON if available
- Object with `.text`, `.reasoning`, `.usage`

## Current Architecture Context

Currently in `evolution_api.py`:
- `spawn_child_llm()` creates a trial, calls evaluator, records to database
- `spawn_children_parallel()` does the same in parallel
- Both increment trial count and affect generation state

The child LLM call itself is in lines 351-357:
```python
response = child_llm.generate(
    messages=[{"role": "user", "content": substituted_prompt}],
    max_tokens=4096,
    temperature=temperature,
)
```

## Exploration Tasks

1. Read `evolution_api.py` to understand current spawn flow
2. Read `cost_tracker.py` to understand budget tracking
3. Read `logger.py` to understand logging patterns
4. Look at how calibration calls work (they're non-trial LLM calls)
5. Consider if this should reuse calibration infrastructure

## Expected Deliverable

A design document with:
1. Recommended API design
2. Cost tracking approach
3. Logging approach
4. Any config changes needed
5. Implementation sketch
6. Test plan

After planning, we'll implement in a separate PR.
