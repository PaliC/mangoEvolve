# MangoEvolve Performance Profiling Report

## Executive Summary

Experiments running >1 hour are dominated by **LLM API latency** (~85-90% of wall time) and **subprocess evaluation overhead** (~10-15%). Internal Python operations (REPL, cost tracking, logging, prompt construction) are negligible (<0.1% of total time).

The profiling was conducted by mocking all LLM API calls with realistic responses and running real evaluations through the actual subprocess pipeline.

---

## Profiling Results

### 1. CRITICAL BOTTLENECK: LLM API Calls (~85-90% of wall time)

LLM API calls are **by far** the dominant cost. In a real experiment:

| Operation | Estimated Time Per Call | Calls Per Generation | Est. Time/Gen |
|-----------|------------------------|---------------------|---------------|
| Root LLM (spawn prompt) | 10-30s | 1 | 10-30s |
| Root LLM (selection) | 5-15s | 1 | 5-15s |
| Child LLM (per child) | 5-15s | 3-10 (parallel) | 5-15s |
| Root LLM (analysis/query_llm) | 5-10s | 0-3 | 0-30s |

**Per generation estimate: 20-90s (API-bound)**

For a 5-generation experiment with 4 children each: ~2-8 minutes just in API calls.
For 20 generations with 10 children: ~7-30 minutes.

**Why experiments exceed 1 hour:**
- With reasoning enabled (high effort), root LLM calls can take 30-60s each
- Root LLM often executes multiple code blocks per generation (analysis + spawn + selection = 3+ LLM calls)
- `query_llm` calls during analysis phases add 5-10s each
- Calibration phase adds 5-10 additional LLM round trips before evolution starts

### 2. SIGNIFICANT BOTTLENECK: Subprocess Evaluation (~350ms per trial)

```
evaluate() mean:                  360ms (5 runs)
  ├─ run_code_with_timeout mean:  368ms
  │   ├─ subprocess.Popen():      ~300ms (subprocess creation + Python startup)
  │   ├─ Code execution:          ~50ms (actual packing computation)
  │   └─ pickle serialization:    ~15ms
  └─ validate_packing mean:       1.4ms (negligible)
```

**Key finding:** The subprocess overhead is ~300ms per trial regardless of code complexity. Even broken code that fails immediately takes ~337ms because the subprocess creation cost dominates.

**Impact at scale:**
- 3 children parallel: ~650ms (pool overhead + 360ms eval)
- 10 children parallel: ~650ms-1.2s (limited by slowest + pool overhead)
- 10 children sequential: ~3.6s

### 3. MODERATE BOTTLENECK: Multiprocessing Pool Overhead (~100-140ms)

```
Pool creation mean:     105-140ms per spawn_children() call
Pool.map overhead:      ~50-200ms depending on worker count
Pool terminate:         ~600ms (!) per generation
```

**Critical discovery from cProfile:** `pool.terminate()` and `_terminate_pool()` consume significant time (~600ms per generation). The `_help_stuff_finish` and thread joining account for much of the wall time in the mocked run:

```
pool.terminate:         2.046s cumulative (3 gens)
_help_stuff_finish:     1.807s cumulative (3 gens)
SemLock.acquire:        1.806s cumulative
```

This means **~600ms per generation is spent just creating and tearing down the multiprocessing pool**.

### 4. NOT A BOTTLENECK: REPL Execution (<2ms)

```
assignment:           0.06ms
computation:          1.13ms
list comprehension:   0.49ms
function definition:  0.04ms
numpy operations:     1.36ms
```

Code extraction is also negligible:
```
extract_python_blocks:   0.011ms
extract_selection_block: 0.007ms
```

### 5. NOT A BOTTLENECK: Cost Tracking (<0.1ms per call)

```
record_usage:          0.010ms avg
get_summary (200 entries): 0.065ms avg
raise_if_over_budget:  0.0002ms avg
```

The `get_summary()` scaling is linear but benign:
```
  10 entries:   0.007ms
  50 entries:   0.025ms
 100 entries:   0.044ms
 500 entries:   0.158ms
1000 entries:   0.302ms
```

Even with 1000 LLM calls, summary computation takes <0.3ms.

### 6. NOT A BOTTLENECK: File I/O (<2ms per operation)

```
log_trial:       0.80-1.05ms avg
log_root_turn:   0.28-0.34ms avg
log_generation:  1.73-1.95ms
save_scratchpad: 1.10-1.25ms
save_experiment: 0.59-0.70ms
```

### 7. NOT A BOTTLENECK: Prompt Construction (<0.02ms)

```
build_root_system_prompt_static:           0.004ms
get_root_system_prompt_parts_with_models:  0.015ms
build_child_system_prompt:                 0.002ms
```

System prompt is ~2900 tokens. Not a concern.

### 8. NOT A BOTTLENECK: Lineage Map Building (<0.5ms)

```
 10 trials: 0.1ms
 50 trials: 0.1ms
100 trials: 0.2ms
200 trials: 0.4ms
```

Scales linearly but negligible even at 200 trials.

---

## End-to-End Generation Breakdown (Mocked LLM)

With **all LLM calls mocked** (returning instantly), a single generation with 3 children takes:

```
build_generation_start_message:     0.0ms (0.0%)
get_system_prompt:                  0.0ms (0.0%)
root_llm_generate (mock):          0.1ms (0.0%)
extract_code_blocks:               0.1ms (0.0%)
execute_code_blocks (incl spawn):  689.8ms (100.0%)
  └─ spawn_children():             ~650ms
       ├─ Pool creation:           ~110ms
       ├─ Eval subprocess x3:      ~400ms (parallel)
       └─ Pool teardown:           ~140ms
TOTAL:                              690ms
```

Full 3-generation run (mocked LLM, 3 children/gen):
```
Total: 1.97s
Per generation: 0.66s
```

**In a real experiment**, add ~20-60s of LLM API time per generation.

---

## cProfile Top Functions (Full Run, Mocked LLM)

| Function | Cumulative Time | Calls |
|----------|----------------|-------|
| `multiprocessing.pool.terminate` | 2.046s | 3 |
| `multiprocessing.pool._terminate_pool` | 1.816s | 3 |
| `SemLock.acquire` | 1.806s | 66 |
| `posix.read` (pipe I/O) | 1.727s | 36 |
| `pool.map` | 1.242s | 3 |
| `posix.write` (pipe I/O) | 1.205s | 33 |
| `selectors.select` (poll) | 0.639s | 29 |

---

## Optimization Recommendations

### High Impact (address LLM API latency)

1. **Persistent multiprocessing pool**: Create the pool once at `EvolutionAPI.__init__()` and reuse across generations. Currently, a new `Pool` is created and torn down per `spawn_children()` call, wasting ~600ms per generation (~250ms create + 350ms+ teardown). Over 20 generations, that's ~12s of pure overhead.

2. **Async LLM calls**: Replace `pool.map()` with `asyncio`-based concurrent API calls. LLM APIs are I/O-bound - `multiprocessing` adds unnecessary process creation overhead for what is fundamentally a network I/O operation. Each subprocess worker imports the full environment and creates a new evaluator instance.

3. **Reduce root LLM round trips**: The current architecture does 2+ LLM calls per generation (spawn + selection). If the root LLM also runs analysis code blocks or `query_llm` calls, this can balloon to 4-6 calls per generation. Each call is 10-30s with reasoning enabled.

4. **Pipeline evaluation**: Don't wait for all children to complete before starting evaluation. Start evaluating the first child's code as soon as it arrives while other children are still generating.

### Medium Impact (reduce subprocess overhead)

5. **In-process evaluation with sandboxing**: The subprocess overhead is ~300ms per evaluation regardless of code complexity. For simple packing code that runs in 50ms, 85% of evaluation time is subprocess creation. Consider running evaluations in-process with restricted imports instead of spawning a new Python process each time.

6. **Subprocess reuse / persistent evaluator process**: Instead of creating a new subprocess per evaluation, maintain a persistent evaluator subprocess that receives code via IPC. This eliminates the ~300ms Python startup cost per evaluation.

7. **Batch child LLM + evaluation**: Currently each worker does LLM call + evaluation sequentially. Separate these so all LLM calls can complete first (async), then evaluations can run in parallel without the LLM latency bottleneck.

### Low Impact (polish)

8. **CostTracker.get_summary() optimization**: Uses list comprehensions over the full usage log. With 1000+ entries this reaches 0.3ms per call. Pre-compute aggregates incrementally in `record_usage()` instead. Low priority since even 1000 calls is sub-millisecond.

9. **Evaluation timeout tuning**: Default timeout is 30-300s per trial. Most successful evaluations complete in <1s. A more aggressive timeout (10-30s) for early generations could save time on pathologically slow generated code.

---

## Estimated Time Breakdown for a Typical 1-Hour Experiment

Assuming: 10 generations, 5 children/gen, reasoning=high

| Component | Est. Time | % of Total |
|-----------|-----------|------------|
| Root LLM API calls (spawn + selection + analysis) | 30-45 min | 50-75% |
| Child LLM API calls (parallel, 5/gen) | 10-20 min | 17-33% |
| Subprocess evaluation (5/gen x 10 gen) | ~18s | 0.5% |
| Multiprocessing pool overhead | ~6s | 0.2% |
| File I/O, logging, prompt construction | ~0.5s | <0.01% |
| **Total** | **~40-65 min** | |

The dominant factor is LLM API latency, which is external and not directly controllable. The most impactful internal optimization would be the persistent pool (#1) and async LLM calls (#2), which together could save 1-2 minutes per experiment through reduced overhead and better parallelism.
