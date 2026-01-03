# Task: Expose Scratchpad as REPL String Variable

## Objective
Make the scratchpad directly accessible in the REPL as a mutable `scratchpad` variable (string), allowing the Root LLM to read and modify it naturally.

## Context
Currently, the Root LLM must call `update_scratchpad(content)` to modify the scratchpad, and the scratchpad content is only shown in the "Evolution Memory" section. Direct variable access enables more natural interaction.

## Requirements

### 1. Create ScratchpadProxy Class

Create a simple wrapper that provides string-like access with auto-persistence:

```python
# In src/mango_evolve/evolution_api.py or src/mango_evolve/repl_proxies.py

class ScratchpadProxy:
    """
    Mutable scratchpad wrapper for REPL access.

    Usage in REPL:
        scratchpad.content  # Read current content
        scratchpad.content = "New content"  # Triggers persistence
        scratchpad.append("More text")  # Append and persist
        str(scratchpad)  # Get as string
        print(scratchpad)  # Prints content
    """

    def __init__(self, api: "EvolutionAPI"):
        self._api = api

    @property
    def content(self) -> str:
        return self._api.scratchpad

    @content.setter
    def content(self, value: str) -> None:
        self._api.update_scratchpad(value)

    def append(self, text: str) -> None:
        """Append text to scratchpad."""
        self._api.update_scratchpad(self._api.scratchpad + text)

    def clear(self) -> None:
        """Clear the scratchpad."""
        self._api.update_scratchpad("")

    def __str__(self) -> str:
        return self._api.scratchpad

    def __repr__(self) -> str:
        content = self._api.scratchpad
        if len(content) == 0:
            return "<scratchpad: empty>"
        preview = content[:100].replace('\n', '\\n')
        if len(content) > 100:
            return f"<scratchpad: {len(content)} chars>\n{preview}..."
        return f"<scratchpad: {len(content)} chars>\n{content}"

    def __len__(self) -> int:
        return len(self._api.scratchpad)

    def __contains__(self, item: str) -> bool:
        return item in self._api.scratchpad

    def __add__(self, other: str) -> str:
        """Allow scratchpad + "text" but don't auto-persist (returns new string)."""
        return self._api.scratchpad + other
```

### 2. Inject into REPL Namespace

Update `get_repl_namespace()` (or `get_api_functions()` if not yet renamed):

```python
def get_repl_namespace(self) -> dict[str, Any]:
    return {
        # ... existing functions ...
        "trials": TrialsProxy(self),
        "scratchpad": ScratchpadProxy(self),  # New
    }
```

### 3. Update System Prompt

Add documentation in `src/mango_evolve/llm/prompts.py`:

```python
"""
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
Max 8000 characters.
"""
```

### 4. Keep update_scratchpad() Function

For backward compatibility, keep the `update_scratchpad()` function available. The ScratchpadProxy uses it internally.

### 5. Maintain Observability

The scratchpad continues to be:
- Saved to `generations/gen_X/scratchpad.json` on each update
- Included in the Evolution Memory section
- Logged in experiment.json

No changes to the persistence logic - ScratchpadProxy just wraps the existing `update_scratchpad()` method.

### 6. Tests

Add tests:
- Test ScratchpadProxy.content getter/setter
- Test ScratchpadProxy.append()
- Test persistence is triggered on modification
- Test __contains__, __len__, __str__, __repr__
- Test integration with REPL execution

## Files to Modify
- `src/mango_evolve/evolution_api.py` - Add ScratchpadProxy, update get_repl_namespace()
- `src/mango_evolve/llm/prompts.py` - Document scratchpad variable
- `tests/test_evolution_api.py` - Add ScratchpadProxy tests

## Acceptance Criteria
1. Root LLM can access `scratchpad.content` in REPL
2. `scratchpad.content = "..."` triggers persistence
3. `scratchpad.append()` works
4. Existing `update_scratchpad()` function still works
5. Scratchpad files continue to be written correctly
6. All existing tests pass
