## Results

We evaluate MangoEvolve on the circle packing benchmark introduced by AlphaEvolve: pack 26 circles of variable radii into the unit square to maximize the sum of their radii.

Our most cost-efficient run used **Gemini 3 Flash** as the Root LLM and **Gemini 2.0 Flash** as the sole child model. It achieved a score of **2.635983**, exceeding AlphaEvolve (2.635863), OpenEvolve (2.635977), and ShinkaEvolve (2.635978) — in **102 program evaluations**, under **36 minutes of wall-clock time**, at a total cost of **$1.80**.

### Setup

| Component | Model | Cost per M tokens (in/out) |
|-----------|-------|---------------------------|
| Root LLM (orchestrator) | Gemini 3 Flash | $0.50 / $3.00 |
| Child LLM (code generation) | Gemini 2.0 Flash | $0.10 / $0.40 |

The Root LLM ran with `xhigh` reasoning effort. Budget was capped at $10; the run consumed only $1.80 (18% of budget). Of that, $1.74 went to the Root LLM's reasoning overhead and just $0.07 to the 102 child LLM code-generation calls. A second child model (Claude 3.7 Sonnet via OpenRouter) was configured but failed due to a routing error, so the entire run used a single child model.

### Convergence

The evolution progressed through 13 generations over 102 total program evaluations:

```
Score
2.636 |                                                          *  <- Gen 12: LP Polish
      |
2.633 |                         *  <- Gen 5: Sobol init
      |               *  <- Gen 3: parameter tuning
2.625 |          *  <- Gen 2: Basin Hopping
      |
2.484 |     *  <- Gen 1: Greedy + SLSQP
      |
0.000 |*  <- Gen 0: diagnostic
      +-----+----+----+----+----+----+----+----+----+----+----+----+--
       0   10   20   30   40   50   60   70   80   90  100  110
                         Cumulative Program Evaluations
```

| Phase | Generations | Evals | Score Range | Key Event |
|-------|-----------|-------|-------------|-----------|
| Bootstrap | 0 | 1 | 0.0 | System connectivity test |
| Exploration | 1 | 10 | 0 → 2.484 | 8 strategies tried, 3 valid. Greedy + SLSQP wins. |
| Rapid convergence | 2–3 | 20 | 2.484 → 2.632 | Basin Hopping added (+0.14 in one generation) |
| Plateau | 4–11 | 61 | 2.632 → 2.633 | 8 generations of failed refinement attempts |
| Breakthrough | 12 | 10 | 2.633 → **2.636** | LP radii polish closes the final gap |

Only 36% of trials (37/102) produced valid packings — the majority failed due to timeouts (21), coding bugs, or solver infeasibility. This high failure rate is characteristic of evolutionary code generation: the system explores aggressively, and most mutations are harmful. What matters is that each generation's few successes compound.

### How the Root LLM solved it

The Root LLM's strategic reasoning, recorded in its scratchpad, reveals a clear narrative:

**Generations 1–3: Cast a wide net, then focus.** The Root LLM spawned 8 diverse strategies in Gen 1 (grid-based, force-directed, Basin Hopping, greedy filling, LP, coordinate descent). Most failed outright. By Gen 2, it identified Basin Hopping as the key mechanism and focused all 10 children on variants of the Greedy + Basin Hopping pipeline, jumping from 2.48 to 2.63.

**Generations 4–11: Plateau and strategic dead ends.** For 8 generations the Root LLM tried to close a 0.003 gap: relaxing radius bounds (caused infeasibility), increasing iteration counts (caused timeouts), hybrid BH+LP during optimization (implementation failures), CMA-ES (module not found), differential evolution (solver failures). Gen 5 contributed the last incremental gain — Sobol sequence initialization — but the improvement was just +0.001. Generations 9 and 10 were total failures (zero valid solutions).

**Generation 12: The insight.** After 8 generations of failed attempts to integrate LP *during* Basin Hopping, the Root LLM made a conceptual leap: decouple the two problems. Run the proven Basin Hopping pipeline unchanged with its conservative 0.15 radius cap (which kept SLSQP numerically stable), then solve a Linear Program *after* to find mathematically optimal radii for the fixed centers. The LP formulation — maximize sum of radii subject to `r_i + r_j ≤ dist(i,j)` and `r_i ≤ min(x, y, 1−x, 1−y)` — is convex, so `linprog` finds the global optimum. This single addition bridged the entire 0.003 gap.

The winning solution's lineage:

```
Gen 1:  Greedy Filling + SLSQP                    → 2.484
Gen 2:  + Basin Hopping global search              → 2.625
Gen 3:  + parameter tuning                         → 2.632
Gen 5:  + Sobol sequence initialization            → 2.633
Gen 12: + post-optimization LP radii polish        → 2.636
```

Five innovations, each building on the last. The final algorithm is a 3-phase pipeline: (1) greedy placement from 4,096 Sobol points, (2) Basin Hopping with SLSQP over all 78 parameters, (3) LP solve for optimal radii.

### Comparison with prior work

| System | Best Score | Program Evals | Wall Time | Cost | LLMs Used |
|--------|-----------|---------------|-----------|------|-----------|
| AlphaEvolve (DeepMind) | 2.635863 | ~thousands (undisclosed) | undisclosed | undisclosed | 2 (Gemini 2.0 Flash + Pro) |
| OpenEvolve (community) | 2.635977 | ~450 | undisclosed | undisclosed | 2 (Gemini Flash + Claude Sonnet) |
| ShinkaEvolve (Sakana AI) | 2.635978 | ~150 | undisclosed | undisclosed | 6 frontier LLMs |
| **MangoEvolve** | **2.635983** | **102** | **36 min** | **$1.80** | **1 (Gemini 2.0 Flash)** |

MangoEvolve exceeds all three comparison systems in score while using fewer program evaluations than any of them. Notably:

- **vs. AlphaEvolve**: +1.2×10⁻⁴ better score in ~10–50× fewer evaluations, using a single cheap model rather than a Gemini Pro + Flash ensemble
- **vs. ShinkaEvolve**: +5.4×10⁻⁶ better score in fewer evaluations (102 vs. ~150), using 1 model instead of 6 frontier LLMs with bandit-based selection
- **vs. OpenEvolve**: +6.1×10⁻⁶ better score in ~4.4× fewer evaluations (102 vs. ~450), without MAP-Elites islands or cascade evaluation

### Solution validity

All constraint violations are floating-point rounding artifacts at the 10⁻¹⁵ to 10⁻¹⁴ level — well within IEEE 754 double-precision noise. To produce a solution that is strictly valid with **zero tolerance** (no constraint violations of any magnitude), we can uniformly scale all radii by a factor of 0.999999999991618. The resulting score is 2.635983084894, a loss of 2.2×10⁻¹¹ — negligible, and still exceeding all comparison systems.

### Reproducibility across runs

We ran 6 total circle packing experiments with different Root LLM and child configurations. All 6 exceeded AlphaEvolve, ShinkaEvolve, and OpenEvolve, with the best achieving 2.635985:

| Run | Root LLM | Children | Evals | Best Score |
|-----|----------|----------|-------|------------|
| Opus Thinking #2 | Claude Opus 4.5 | Opus + Sonnet + Gemini Flash | 255 | **2.635985** |
| Opus Thinking #1 | Claude Opus 4.5 | Opus + Sonnet + Gemini Flash | 256 | 2.635983 |
| Opus Thinking #3 | Claude Opus 4.5 | Opus + Sonnet + Gemini Flash + Grok | 319 | 2.635983 |
| Gemini Mixed | Gemini 3 Pro | Gemini 3 Flash + Pro | 315 | 2.635983 |
| **OE Config #1** | **Gemini 3 Flash** | **Gemini 2.0 Flash** | **102** | **2.635983** |
| OE Config #2 | Claude 3.7 Sonnet | Sonnet + Gemini Flash | 160 | 2.635983 |

100% success rate across all runs. The system reliably converges to scores in the range 2.63598–2.63599, suggesting this is at or very near the true optimum for 26 circles.
