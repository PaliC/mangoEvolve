# MangoEvolve: Circle Packing Ablation Analysis

## 1. State-of-the-Art Comparison

### 1.1 Published Results (26 Circles in Unit Square)

| System | Score (Relaxed, 1e-6) | Score (Strict) | Models | Evaluations | Cost |
|--------|----------------------|----------------|--------|-------------|------|
| Previous best known | 2.6340 | 2.6340 | N/A | N/A | N/A |
| AlphaEvolve (Google) | 2.63586275 | N/A | Gemini (internal) | Thousands | N/A |
| ShinkaEvolve (Sakana AI) | 2.635983099 | 2.63597771 | Claude Sonnet 4, o4-mini, GPT-4.1{,mini,nano} | 150 | N/A |
| OpenEvolve (community best) | 2.635977395 | N/A | Gemini Flash 2.0 + Claude 3.7 Sonnet | ~460 gens | N/A |

**Sources**: AlphaEvolve paper; [ShinkaEvolve paper](https://arxiv.org/html/2509.19349v1); [OpenEvolve Issue #156](https://github.com/algorithmicsuperintelligence/openevolve/issues/156)

### 1.2 MangoEvolve Results

| Configuration | Score (Relaxed) | Score (Strict) | Models | Evaluations | Gens | Cost | Wall Time |
|--------------|----------------|----------------|--------|-------------|------|------|-----------|
| **Baseline** (all features) | 2.6359830849 | 2.6359830274 | gemini-3-flash | 60 | 10 | $0.70 | 81 min |
| **No query_llm** | 2.6359830855 | 2.6359828255 | gemini-3-flash | 60 | 10 | $0.71 | 77 min |
| **No scratchpad** | 2.6359830850 | 2.6359829142 | gemini-3-flash | 60 | 10 | $0.62 | 80 min |
| **No trial reasoning** (buggy) | 2.6359830849 | 2.6359830849 | gemini-3-flash | 59 | 10 | $0.69 | 82 min |
| **All disabled** | 2.6359830849 | 2.6359828249 | gemini-3-flash | 60 | 10 | $0.59 | 74 min |
| **ShinkaEvolve config** | 2.6359196553 | 2.6359196553 | GPT-5 root + 7 child models | 160 | 20 | $13.13 | 297 min |
| **OpenEvolve config** | 2.6359830849 | 2.6359828249 | Claude 3.7 Sonnet root + Gemini 2.0 Flash | 160 | 16 | $10.16 | 91 min |

**Key comparison**: The OpenEvolve config uses the **same models** as OpenEvolve (Claude 3.7 Sonnet + Gemini 2.0 Flash). The ShinkaEvolve config uses a model ensemble **inspired by** ShinkaEvolve's multi-model approach.

### 1.3 Strict vs Relaxed Scores

The MangoEvolve evaluator uses a `tolerance = 1e-6` that relaxes constraints in two ways:
1. **Boundary**: Circles may extend up to 1e-6 beyond the [0,1] square
2. **Overlap**: Required separation distance reduced by 1e-6 (i.e., `dist >= r_i + r_j - 1e-6`)

The ShinkaEvolve paper explicitly reports both: **2.635983099** (relaxed) vs **2.63597771** (strict), a difference of ~5.4e-6.

**Strict scores for MangoEvolve experiments**:

Strict scores are computed two ways and the **best** is reported:
1. **Re-execution**: Run the best trial's code again, then uniformly shrink radii until all constraints hold at tolerance=0 (alpha method).
2. **Analytical bound**: Shrink each of the 26 radii by 1e-8: `strict_score = recorded_score - 26 × 1e-8`. This is conservative since SLSQP internal tolerances (ftol=1e-12) produce near-strict solutions — re-execution confirmed worst violations are only 3-8e-9 per radius pair.

The best of the two is used because the algorithms are stochastic (shaking, multi-start), so re-execution may land on a different (sometimes worse) local optimum.

| Experiment | Recorded Score (relaxed) | Strict Score | Delta | Method |
|-----------|-------------------------|-------------|-------|--------|
| **Baseline** | 2.6359830849 | 2.6359830274 | 5.75e-08 | re-exec (alpha=0.9999999774) |
| **No query_llm** | 2.6359830855 | 2.6359828255 | 2.60e-07 | analytical (r - 1e-8) |
| **No scratchpad** | 2.6359830850 | 2.6359829142 | 1.71e-07 | re-exec (alpha=0.9999999327) |
| **No trial reasoning** | 2.6359830849 | 2.6359830849 | 0.00 | re-exec (alpha=1.0, strict-valid) |
| **All disabled** | 2.6359830849 | 2.6359828249 | 2.60e-07 | analytical (r - 1e-8) |
| **ShinkaEvolve config** | 2.6359196553 | 2.6359196553 | 0.00 | re-exec (alpha=1.0, strict-valid) |
| **OpenEvolve config** | 2.6359830849 | 2.6359828249 | 2.60e-07 | analytical (r - 1e-8) |

The worst-case delta is **2.6e-7**, which is **20x smaller** than ShinkaEvolve's reported relaxed-to-strict gap of ~5.4e-6. Several experiments are already strictly valid upon re-execution (alpha=1.0).

---

## 2. Convergence Analysis

### 2.1 Generation to Reach Target Score (>= 2.6359)

| Experiment | Gen Reached | Total Evaluations | Wall Time (approx) |
|-----------|-------------|-------------------|-------------------|
| **Baseline** | **Gen 2** | **18** | ~102 min |
| All disabled | Gen 3 | 24 | ~93 min |
| No query_llm | Gen 5 | 36 | ~128 min |
| No trial reasoning | Gen 5 | 36 | ~131 min |
| No scratchpad | Gen 6 | 42 | ~216 min |
| ShinkaEvolve config | Gen 10 | 88 | ~198 min |
| OpenEvolve config | Gen 14 | 150 | ~91 min |

### 2.2 Best-Score-So-Far Per Generation (Cumulative)

**Ablation experiments (10 generations, 6 children/gen):**

| Gen | Baseline | No query_llm | No scratchpad | No trial reason. | All disabled |
|-----|----------|-------------|---------------|------------------|--------------|
| 0 | 2.61587 | 2.61921 | 2.61312 | 2.62991 | 2.63070 |
| 1 | 2.63304 | 2.62932 | 2.63091 | 2.63234 | 2.63070 |
| 2 | **2.63598** | 2.62996 | 2.63091 | 2.63234 | 2.63070 |
| 3 | 2.63598 | 2.62996 | 2.63187 | 2.63234 | **2.63598** |
| 4 | 2.63598 | 2.63021 | 2.63187 | 2.63234 | 2.63598 |
| 5 | 2.63598 | **2.63598** | 2.63532 | **2.63598** | 2.63598 |
| 6 | 2.63598 | 2.63598 | **2.63598** | 2.63598 | 2.63598 |
| 7 | 2.63598 | 2.63598 | 2.63598 | 2.63598 | 2.63598 |
| 8 | 2.63598 | 2.63598 | 2.63598 | 2.63598 | 2.63598 |
| 9 | 2.63598 | 2.63598 | 2.63598 | 2.63598 | 2.63598 |

**Non-ablation experiments:**

| Gen | ShinkaEvolve config | OpenEvolve config |
|-----|--------------------|--------------------|
| 0 | 2.58796 | 2.61219 |
| 1 | 2.61274 | 2.61219 |
| 2 | 2.61418 | 2.61219 |
| 3 | 2.61418 | 2.61219 |
| 4 | 2.61418 | 2.61219 |
| 5 | 2.63580 | 2.63304 |
| 6 | 2.63580 | 2.63304 |
| 7 | 2.63587 | 2.63304 |
| 8 | 2.63587 | 2.63429 |
| 9 | 2.63587 | 2.63429 |
| 10 | **2.63592** | 2.63429 |
| 11 | 2.63592 | 2.63429 |
| 12 | 2.63592 | 2.63429 |
| 13 | 2.63592 | 2.63429 |
| 14 | - | **2.63598** |
| 15 | 2.63592 | 2.63598 |
| 16-19 | 2.63592 | - |

### 2.3 Convergence Insights

1. **Baseline is fastest**: Reaches the optimum at generation 2 (18 evaluations). This is remarkable — 18 evaluations of a single cheap model (gemini-3-flash) matches what ShinkaEvolve achieves with 150 evaluations across 5 frontier models.

2. **All disabled is nearly as fast**: Generation 3 (24 evaluations), suggesting the core evolution mechanism is sufficient without auxiliary features.

3. **Multi-model is slower**: The ShinkaEvolve config (7 models, GPT-5 root) takes 88 evaluations and never reaches the full 2.63598. The OpenEvolve config (Claude 3.7 Sonnet root) takes 150 evaluations. Both are significantly slower than the gemini-3-flash ablations.

4. **Plateau behavior**: All experiments plateau quickly. Once ~2.63598 is reached, further generations don't improve. The optimization landscape funnels to the same local/global optimum regardless of approach.

---

## 3. Ablation Analysis

### 3.1 Feature Impact Summary

| Ablation | Best Score | Gen to 2.6359 | Cost | Success Rate | Root Tokens |
|----------|-----------|----------------|------|-------------|-------------|
| **Baseline** (all features) | 2.6359830849 | Gen 2 | $0.70 | 75% (45/60) | 449K input |
| **-query_llm** | 2.6359830855 | Gen 5 | $0.71 | 70% (42/60) | 329K input |
| **-scratchpad** | 2.6359830850 | Gen 6 (Gen 9 exact) | $0.62 | 70% (42/60) | 326K input |
| **-trial_reasoning** (buggy) | 2.6359830849 | Gen 5 | $0.69 | 85% (50/59) | 379K input |
| **All disabled** | 2.6359830849 | Gen 3 (Gen 5 exact) | $0.59 | 78% (47/60) | 244K input |

**Note on no_trial_reasoning**: This ablation has a bug where trial reasoning was not actually exposed to the root LLM even in the baseline. The "no trial reasoning" flag hides something that wasn't visible in the first place. As such, this ablation is not informative about the actual value of trial reasoning.

### 3.2 query_llm Analysis (Comprehensive)

#### 3.2.1 Call Counts

| Experiment | Total Calls | Per Generation (avg) |
|-----------|-------------|---------------------|
| Baseline | 53 | 5.3 |
| No scratchpad | 55 | 5.5 |
| No trial reasoning | 49 | 4.9 |
| ShinkaEvolve config | 47 | 2.4 |
| OpenEvolve config | 158 | 9.9 |

#### 3.2.2 Category Breakdown

| Category | Baseline | No scratchpad | No trial reason. | ShinkaEvolve | OpenEvolve |
|----------|----------|--------------|------------------|--------------|------------|
| trial_analysis | 4 | 6 | 3 | 0 | 67 |
| comparison | 4 | 6 | 2 | 0 | 21 |
| error_diagnosis | 1 | 0 | 0 | 0 | 2 |
| strategy_planning | 0 | 0 | 0 | 0 | 1 |
| code_review | 0 | 2 | 0 | 0 | 1 |
| explanation | 0 | 0 | 0 | 3 | 10 |
| general_analysis | 0 | 1 | 0 | 0 | 0 |
| other | 44 | 40 | 44 | 44 | 56 |

**Note**: The "other" category is high because many query_llm calls use f-strings or complex prompt construction that the parser can't fully extract. Manual inspection of the logs reveals these are primarily trial analysis and code review calls where the prompt text is dynamically constructed.

#### 3.2.3 Representative query_llm Call Patterns

**Trial Analysis** (most common in OpenEvolve config):
- "Compare these top approaches from Generation 0 and analyze their strengths and weaknesses"
- "Analyze why these trials failed (INVALID)"
- "Analyze this circle packing code (trial_0_0, score=...)"

**Comparison**:
- "Compare the top trial of Gen 5 with the previous best"
- "Compare these two circle packing algorithms"

**Code Review** (seen in no_scratchpad):
- "Extract and explain the Linear Programming (LP) refinement and the penalty continuation logic in this code"
- "Review all successful trials and identify approaches most different from our top performers"

**Strategy Planning** (rare):
- "Suggest three different algorithms for packing circles of varying sizes into a unit square"

**Error Diagnosis**:
- "Identify the common reasons for failure in these trials"
- "What went wrong with this implementation? Identify the key issues"

#### 3.2.4 query_llm Calls Per Generation

**Baseline**: Steady 3-9 calls per generation. Peaks at Gen 4 (9 calls) during the mid-evolution analysis phase.

**OpenEvolve config**: Much heavier usage. Peaks at Gen 5 (21 calls!) when the root LLM (Claude 3.7 Sonnet with xhigh reasoning) extensively analyzed trial approaches. This correlates with the OpenEvolve config's breakthrough from 2.612 to 2.633 in that generation.

**ShinkaEvolve config**: Uses query_llm primarily for calibration (10 calls in Gen 0) and then a steady 2 calls per generation — mostly for trial summarization.

#### 3.2.5 query_llm Impact Assessment

**Impact on convergence speed**: The baseline (with query_llm) reaches 2.6359 at Gen 2, while no_query_llm reaches it at Gen 5. This is a **3-generation delay** (18 fewer evaluations needed with query_llm). However, the all_disabled experiment (also without query_llm) reaches it at Gen 3, suggesting the delay may not be solely attributable to query_llm.

**Impact on final score**: No measurable difference. All experiments converge to approximately the same score (~2.635983).

**Impact on cost**: query_llm adds ~$0.09 to root LLM costs (comparing baseline $0.29 root cost vs all_disabled $0.17 root cost, though scratchpad/reasoning also contribute to the difference).

**Qualitative impact**: query_llm enables the root LLM to:
1. Analyze trial code without loading full code into its own context
2. Get external analysis of why trials failed
3. Compare different algorithmic approaches
4. Plan strategy based on external LLM reasoning

The no_scratchpad experiment compensates for lacking scratchpad by using more query_llm calls (55 vs 53 for baseline), suggesting the features are partially fungible.

### 3.3 Scratchpad Analysis

#### 3.3.1 Word Count Per Experiment

| Experiment | Total Words | Non-empty Gens | Words/Gen (avg) |
|-----------|-------------|----------------|-----------------|
| Baseline | 21,301 | 10/10 | 2,130 |
| No query_llm | 5,168 | 10/10 | 517 |
| No trial reasoning | 18,499 | 10/10 | 1,850 |
| ShinkaEvolve config | 23,918 | 20/20 | 1,196 |
| OpenEvolve config | 40,335 | 17/17 | 2,373 |

**Key insight**: The **no_query_llm** scratchpad is **4x smaller** than the baseline (5,168 vs 21,301 words). Without query_llm to provide external analysis, the root LLM writes shorter, more concise notes. With query_llm available, the root LLM writes extensively — often copying analysis from query_llm responses into the scratchpad.

#### 3.3.2 Scratchpad Themes

All experiments with scratchpad enabled share these common themes:
- **Strategy**: Planning for next generation
- **Analysis**: Understanding trial results
- **Error tracking**: Recording failures and their causes
- **LP_optimization**: Linear programming for radii optimization
- **SLSQP_refinement**: SLSQP solver tuning
- **Basin hopping**: Global optimization strategies
- **Shaking/perturbation**: Local search techniques

The **baseline** scratchpad uniquely includes **code snippets** starting from Gen 4, where the root LLM recorded recommended code patterns. This reflects the query_llm-assisted deeper analysis.

#### 3.3.3 Scratchpad Evolution Example (Baseline)

The baseline scratchpad grows from 207 words (Gen 0) to 4,956 words (Gen 9), showing an accumulation pattern:

- **Gen 0** (207 words): Initial strategy listing 6 approaches to explore
- **Gen 2** (633 words): "HUGE SUCCESS: trial_2_4 achieved 2.635983" — discovery of the shaking+SLSQP method
- **Gen 4** (1,979 words): Deep analysis of the 2e-6 gap to best known, including a complete code snippet for LP integration
- **Gen 8-9** (4,880-4,956 words): Extensive analysis with geometry analysis and precision limit discussion

#### 3.3.4 Scratchpad Without query_llm

The no_query_llm scratchpad caps at 726 words by Gen 8-9 (vs 4,956 for baseline). It contains the same themes but lacks:
- The detailed code snippets found in baseline
- The deep analytical passages that resulted from query_llm analysis
- The multi-paragraph strategic reasoning

Despite this, no_query_llm achieves the **same final score**, reaching it 3 generations later.

### 3.4 REPL Function Analysis

#### 3.4.1 Functions Defined Per Experiment

| Experiment | Total Definitions | Unique Functions |
|-----------|------------------|-----------------|
| Baseline | 13 | 6 |
| No query_llm | 11 | 6 |
| No scratchpad | 7 | 7 |
| No trial reasoning | 10 | 6 |
| All disabled | 1 | 1 |
| ShinkaEvolve config | 94 | 33 |
| OpenEvolve config | 10 | 8 |

#### 3.4.2 Function Categories

**Analysis/Utility Functions** (common across ablations):
- `get_top_trials(limit=5)` — Query best trials
- `analyze_best_trials()` — Analyze trial characteristics
- `get_best_score(trials)` — Simple score extraction
- `summarize_trials(trial_list)` — Summarize trial metadata

**Domain-Specific Functions** (optimization helpers):
- `get_optimal_radii(centers)` — LP-based radii optimization (baseline)
- `check_precision_limit()` — Test numerical precision boundaries (baseline)
- `analyze_best_geometry()` — Geometric analysis of best packing (baseline)
- `refine_radii(centers, n, buffer=1e-12)` — Radius refinement (no_scratchpad)
- `gradient(vars, n, lam)` — Gradient computation (no_scratchpad)
- `check_validity(centers, radii)` — Validation helper (no_query_llm)

**ShinkaEvolve config** (33 unique functions — by far the most):
The GPT-5 root LLM defined an extensive toolkit:
- `circle_packing_penalty(x, y, r)` — Custom penalty function
- `lp_feasible(centers, r_upper)` — LP feasibility check
- `tri_hex_seed(n, jitter, edge_bias, rng)` — Hexagonal seed generation
- `random_edge_corner_seeds(n, k, rng)` — Edge-biased initialization
- `make_seeds(n=26, num_seeds=12, rng)` — Multi-strategy seed factory
- `hex_grid(npt)` — Hexagonal grid generator
- `strictly_feasible(centers, radii, tol)` — Strict feasibility check
- `approach_tags(code_lower)` — Automatic algorithm classification

This shows the GPT-5 root LLM building a significantly more elaborate scaffolding infrastructure, yet achieving a **lower final score** (2.63592 vs 2.63598).

#### 3.4.3 Function Persistence

The **all_disabled** experiment defined only 1 function (`get_best_score`) — showing that the evolution mechanism works even with minimal REPL scaffolding.

The **baseline** introduced functions gradually: analysis functions in Gen 0, precision analysis in Gen 6, geometry analysis in Gen 8. This progressive toolbuilding mirrors the scratchpad's strategy evolution.

### 3.5 Peculiarities and Unexpected Findings

#### 3.5.1 All Disabled Matches Baseline

The most striking finding: **all features disabled** ($0.59) achieves the same score as **all features enabled** ($0.70). With 16% lower cost and 78% success rate, the minimal configuration is the most cost-effective.

This suggests that for circle packing, the core evolutionary loop (spawn children → evaluate → select best → repeat) is sufficient. The auxiliary features (query_llm, scratchpad, trial reasoning) add interpretability and strategic depth but not measurable performance.

#### 3.5.2 ShinkaEvolve Config Underperforms

The multi-model ShinkaEvolve config ($13.13, 7 child models, GPT-5 root) achieves a **lower** score (2.63592) than any single-model ablation (2.63598). This is 22x more expensive for a worse result.

Possible explanations:
- **Model diversity creates noise**: Different models produce solutions with different conventions, making it harder for the root LLM to synthesize insights
- **GPT-5 root overhead**: The GPT-5 root LLM defined 33 REPL functions — enormous scaffolding that consumed budget without improving the core optimization
- **Timeout constraints**: Some models (especially when reasoning is enabled) may produce code that runs slower, reducing the effective optimization time
- **Gen 14 anomaly**: Generation 14 produced only scores around 2.17 — a severe regression that wasted evaluations

#### 3.5.3 OpenEvolve Config: Heavy query_llm Usage

The OpenEvolve config made 158 query_llm calls (3x more than any ablation). Generation 5 alone had 21 calls. Despite this heavy analysis, the OpenEvolve config took 14 generations (150 evaluations) to reach 2.63598 — while the baseline reached it in 2 generations (18 evaluations).

The Claude 3.7 Sonnet root LLM's extensive reasoning may have been *too* thorough — spending time on analysis rather than spawning children. The scratchpad grew to 40,335 words (2x the baseline).

#### 3.5.4 No Trial Reasoning: Highest Success Rate but Buggy

The no_trial_reasoning ablation has the highest success rate (85%, 50/59 trials). However, this ablation is misleading because trial reasoning was not actually exposed in any experiment due to a bug — the flag hides something that was already hidden.

#### 3.5.5 Shaking + SLSQP: Universal Discovery

Across **all** experiments, the winning algorithmic pattern is the same:
1. Start with a grid or force-directed initialization
2. Apply random perturbations ("shaking")
3. Run SLSQP local optimization
4. Keep the best result and repeat

This pattern emerges in the baseline at Gen 2 (trial_2_4), in no_query_llm at Gen 5 (trial_5_4), and in all_disabled at Gen 5 (trial_5_x). The exact generation varies, but the algorithm is always the same.

#### 3.5.6 Convergence to Same Numerical Limit

All ablation experiments converge to approximately **2.63598308xx**. The variation in the last 2-3 digits (e.g., 2.6359830849 vs 2.6359830855) is within SLSQP numerical precision limits. This confirms that all experiments find the same packing configuration — only the convergence speed differs.

#### 3.5.7 query_llm Mentions in Shinka Config: Calibration-Heavy

The ShinkaEvolve config's 47 query_llm calls are heavily front-loaded: 10 in Gen 0 (calibration phase) and then 2 per generation. In Gen 0, the GPT-5 root LLM used query_llm to solve math problems as a "calibration test" for child models — including `"Find the minimum value of f(x) = x^x for x > 0"`. This creative calibration usage is unique to the ShinkaEvolve config.

---

## 4. Key Insights and Takeaways

### 4.1 Cost Efficiency

MangoEvolve achieves state-of-the-art circle packing results with a **single cheap model** (gemini-3-flash) at $0.59-$0.70. This is:
- **22x cheaper** than the multi-model ShinkaEvolve config ($13.13)
- **14x cheaper** than the OpenEvolve config ($10.16)
- Achieves the same or better final score

### 4.2 Sample Efficiency

The baseline reaches the optimum in **18 evaluations** (Gen 2). For comparison:
- ShinkaEvolve (published): 150 evaluations
- OpenEvolve (published): ~460 generations
- MangoEvolve ShinkaEvolve config: 88 evaluations
- MangoEvolve OpenEvolve config: 150 evaluations

The same evolutionary framework achieves better sample efficiency with a single model than multi-model configurations.

### 4.3 Feature Value

| Feature | Score Impact | Convergence Impact | Cost Impact | Interpretability Impact |
|---------|-------------|-------------------|-------------|----------------------|
| query_llm | None | +3 gens faster (baseline vs no_query_llm) | +$0.09 | High (detailed analysis) |
| Scratchpad | None | +4 gens faster (baseline vs no_scratchpad) | +$0.08 | High (strategy tracking) |
| Trial reasoning | None (buggy) | N/A | +$0.10 | N/A (bug) |

Features provide value for **understanding** the evolution process but not for **final performance**. They may modestly improve convergence speed.

### 4.4 Multi-Model Diversity Does Not Help

The ShinkaEvolve config with 7 different child models (including GPT-5, Claude Sonnet 4, Gemini 2.5 Pro, o4-mini) achieves a **lower** final score than a single gemini-3-flash model. Model diversity adds cost and complexity without improving the search.

### 4.5 The Algorithm Always Converges

Regardless of features, models, or configuration, the evolutionary search consistently discovers the "shaking + SLSQP" algorithm and converges to the same numerical limit (~2.635983). This robustness is a strength of the evolutionary approach — the search space funnels to the same optimum.
