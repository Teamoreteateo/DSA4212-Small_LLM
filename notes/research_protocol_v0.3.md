# Research Protocol v0.3: Fixed-Compute Model Size and Data Repetition

**Project:** DSA4212 Small LLM (group of two)
**Status:** Draft for teammate QC. Nothing in this document has been run. No data has been downloaded and no Modal job has been launched.
**Date:** 23 October 2026
**Supersedes:** protocol v0.2 (equal-token design) and its methodological review, which gave a REVISE verdict.

Every number below carries a provenance tag:

| Tag | Meaning |
|---|---|
| `[looked up]` | Checked at the source on 23 Oct 2026 |
| `[paper]` | Reported in a cited paper |
| `[derived]` | Arithmetic from the other numbers |
| `[assumed]` | Our estimate, to be replaced by the pilot |

---

## 1. Research questions

**Primary question (current).**
> Under a fixed compute budget, how do the optimal model size and the value of repeating data change as unique training data shrinks?

**Original question (retained as the secondary question).**
> How does model capacity affect the point of diminishing returns from repeated training data in small language models?

The primary question is answered by comparing model sizes at equal compute (section 4.1). The original question is answered by within-size repetition comparisons (section 4.2). Section 4 also explains why comparing repetition across sizes is only exploratory.

## 2. Hypotheses

| ID | Hypothesis | Comparison type | Test |
|---|---|---|---|
| H1 (primary) | The compute-optimal size N\*(U) falls as unique data U shrinks | Across sizes, within each U, at fixed C | Bootstrap 95% CI of the slope of log N\* on log U lies above 0, and P_best shifts toward smaller sizes |
| H2 | Within each size, the recovery ratio ρ falls as epochs grow, and ρ crosses 0.5 inside the grid for XS, S and M | Within size, across U | E½ estimated with a CI; reported as a bound if ρ does not cross 0.5 |
| H3 | At the smallest U the loss curve over sizes flattens: the best-minus-worst BPB spread at U4 is smaller than at U0 | Across sizes, at fixed C | Ratio of spreads, with a bootstrap CI below 1 |
| H0 (replication check) | Where a size repeats about 3 times or fewer (M at U1, L at U2), the penalty against fresh data is within ±1% of fresh BPB | Within size | Two one-sided tests (TOST) at α = 0.05 with pooled seed variance |
| H4 (exploratory, original question) | At matched epoch counts, ρ is lower for larger models | Across sizes (confounded, see 4.3) | Descriptive only: ρ against E and against U/N, overlaid by size |

## 3. Fixed-compute (iso-FLOP) methodology

### 3.1 Budget

The per-run compute budget is **C = 2.5e16 training FLOPs**, the same for every primary run, every size and every data condition.

The brief allows the per-run budget to be stated as a measured quantity such as FLOPs `[looked up]`. We use FLOPs rather than GPU-time for two reasons. Small models use the GPU poorly, so a GPU-time budget would penalise them for an implementation artefact. GPU-time also varies between GPU types and between our two workspaces. GPU-time is still logged for every run.

**FLOP accounting.** This convention is fixed before any run:

```
f_N     = 6 * N_total + 12 * n_layer * n_ctx * d_model    # training FLOPs per token
T_N     = C / f_N                                         # training tokens for size N
steps_N = round(T_N / 65,536)                             # batch = 128 sequences x 512 tokens
```

- `N_total` includes the tied embedding matrix, whose matmul is the language-model head.
- The second term is attention.
- Evaluation FLOPs are logged separately and are not part of C.

Chinchilla's compute-optimal rule for this budget gives N_opt ≈ √(C/120) ≈ 14.4M total parameters `[derived]`, from about 20 tokens per parameter with C ≈ 6ND `[paper]`. That lies between S and M, so the fresh-data optimum should fall inside the grid.

### 3.2 Model sizes

The architecture is a GPT-style pre-LayerNorm decoder with a GELU MLP (4× expansion), no biases, and tied input and output embeddings. Context length is 512, one positional scheme is used for every run, and dropout is 0. Every size has head width 64 and width-to-depth ratio 64.

Initialisation is random: weights are drawn from Normal(0, 0.02), with residual projections scaled by 1/√(2·n_layer), seeded by the run seed. **No pretrained weights, embeddings or neural components are used.** The only checkpoints ever loaded are the same run's own, when resuming.

| Size | Layers | Width | Heads | Non-emb N | Embedding | Total N | Emb share | FLOPs/token | T_N (tokens) | Steps | Tokens/param |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XS | 3 | 192 | 3 | 1.33M | 3.15M | 4.47M | 70% | 30.4M | 823.0M | 12,558 | 184 |
| S | 4 | 256 | 4 | 3.15M | 4.19M | 7.34M | 57% | 50.3M | 496.7M | 7,579 | 68 |
| M | 6 | 384 | 6 | 10.62M | 6.29M | 16.91M | 37% | 115.6M | 216.3M | 3,300 | 13 |
| L | 8 | 512 | 8 | 25.17M | 8.39M | 33.55M | 25% | 226.5M | 110.4M | 1,684 | 3 |

All values are `[derived]`. Rounding the step count keeps every run within 0.02% of C.

### 3.3 Data and tokeniser (unchanged from v0.2)

- **Dataset:** FineWeb-Edu `sample-10BT`: about 10B GPT-2 tokens in 9.67M documents, ODC-By 1.0 licence `[looked up]`. It is filtered, educational web text, close in kind to the C4 data in Muennighoff et al. (2023).
- **Splits:** each document is assigned by a stable hash of its `id` to one of four splits: tokeniser (about 200 MB of text), validation (5M tokens), test (10M tokens) and training pool.
- **Deduplication and audit:** exact deduplication of whitespace-normalised text, both across splits and within the pool. A 13-gram overlap audit runs on 1,000 test documents; if overlap exceeds 1%, the split switches to URL host.
- **Tokeniser:** byte-level BPE with 16,384 entries, trained with `tokenizers` on the tokeniser split only, then frozen and hashed. Token IDs are stored as `uint16`. A 50k GPT-2 vocabulary would put about 12.9M parameters in S's embedding, four times its transformer.
- **Pool layout:** documents are shuffled once with data seed 0 and concatenated with an end-of-text token. Every unique budget U is the **first U tokens** of the pool, so the budgets are nested.
- **Sufficiency:** fresh data for XS needs 823M training tokens plus 15M held out, about 1.0B tokens with a deduplication margin. That is roughly 10% of `sample-10BT` `[derived]`.

### 3.4 Training controls (identical for every run)

- **Optimiser:** AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices only, gradient clipping at 1.0, bf16 autocast.
- **Schedule:** linear warmup over 2% of steps (at least 10 steps), then cosine decay to 10% of peak over the run's own step count.
- **Learning rate:** the peak LR for each size is chosen by the sweep (section 6) and then frozen.
- **Data order:** each epoch is a fresh permutation of the U windows (512 inputs plus 1 target), seeded by the pair (run seed, epoch). The last permutation is cut off at T_N, so epoch counts are fractional. The loader logs how many times each window is exposed.
- **Logging per run:** git commit, config hash, GPU type, wall-clock, tokens/s, unique windows, exposure counts, FLOPs, and a hash of the initial weights.

## 4. Confounding between size, tokens processed and repetition

At fixed C, a model's size determines its token budget, `T_N = C / f_N`. Within a data condition U, its epoch count follows:

```
E_N(U) = T_N / U = C / (f_N * U)
```

**Across sizes, at a fixed U and C, three quantities are perfectly collinear.** A larger model has more capacity, processes fewer tokens, and repeats each unique token fewer times. XS sees 7.5× as many tokens as L, and so repeats every unique token 7.5× as often. No comparison across sizes can separate these three factors, and the design cannot fix this. Choosing a size at fixed compute *is* choosing a bundle of capacity, tokens and repetition.

**Within a size, two quantities are collinear.** N, T_N and C are fixed, and only U varies. Because E = T_N / U, fewer unique tokens means more epochs, and the two cannot be varied independently at fixed tokens. That is acceptable, because "repeating a smaller set" is exactly the treatment being studied.

The comparisons below state what each one can and cannot support.

### 4.1 Supports conclusions about optimal model size (primary)

**Comparison:** all four sizes within one condition U, all at the same C.

- **Supports:** which size minimises test BPB for a given compute budget and unique-data budget. How the optimum N\*(U) and the flatness of the curve change as U shrinks (H1, H3). The data level U× at which the fresh-data optimum stops being best. These are statements about the best way to *allocate* fixed compute, which is what the primary question asks.
- **Does not support:** attributing a shift in N\* to capacity alone, or to repetition alone. Language such as "the optimum shifts *because* small models repeat more" is a hypothesis, not a finding.

### 4.2 Supports conclusions about repetition (secondary)

**Comparison:** one size across conditions U0 to U4, at fixed N, T_N and C. The one-pass controls are within the same size, at the same U.

- **Supports:** the value of repetition for that size: penalty P, gain G, recovery ratio ρ, and E½ (H2, H0). These are causal effects of shrinking unique data, with compute held equal. This is the v0.2 equal-token analysis, kept intact.
- **Does not support:** claims about compute allocation.

### 4.3 Exploratory only: comparing repetition across sizes (original question, H4)

**Comparison:** ρ curves of different sizes, overlaid at matched E or matched U/N.

This bears on the original question, but it is confounded. Each size is in a different tokens-per-parameter regime (184 for XS against 3 for L) and covers a different epoch range. Any difference may reflect training regime rather than capacity. These results are reported descriptively, with no confirmatory test. Removing the confound would need an equal-token design across sizes (as in v0.2) at 2 to 6 times the compute for the larger sizes. That is outside this budget and is listed as future work.

### 4.4 Not supported by this design

- Any equal-token comparison across sizes. v0.2's design had one; v0.3 does not.
- Conclusions about GPU-time-optimal sizes. Only the FLOP budget is controlled. A GPU-time view would favour larger models, which use the GPU better.

## 5. Experimental matrix

### 5.1 Primary matrix: 60 runs

4 sizes × 5 unique-data conditions × 3 seeds (0, 1, 2). Every run is exactly C = 2.5e16 FLOPs and reports final-step test BPB.

| Condition | Unique tokens U | XS epochs | S epochs | M epochs | L epochs | Runs |
|---|---|---|---|---|---|---|
| U0 (fresh) | ≥ T_N (no repeats) | 1 | 1 | 1 | 1 | 12 |
| U1 | 75,497,472 | 10.9 | 6.6 | 2.9 | 1.5 | 12 |
| U2 | 37,748,736 | 21.8 | 13.2 | 5.7 | 2.9 | 12 |
| U3 | 18,874,368 | 43.6 | 26.3 | 11.5 | 5.8 | 12 |
| U4 | 9,437,184 | 87.2 | 52.6 | 22.9 | 11.7 | 12 |
| **Total** | | | | | | **60** |

The U values are 302.0M / 2^k for k = 2 to 5, so they are exact multiples of a 65,536-token step. In U0 each size uses the first T_N tokens of the pool.

### 5.2 Supporting runs: 46 runs

| Block | Configuration | Runs |
|---|---|---|
| LR sweep | 4 sizes × 3 peak LRs, fresh data, budget C/4 (step counts: XS 3,140, S 1,895, M 825, L 421) | 12 |
| One-pass controls | 4 sizes × U1 to U4 × seeds {0, 1}; one pass over U (1,152, 576, 288 and 144 steps) | 32 |
| LR × scarcity check | S at U3 (26.3 epochs), peak LR × 0.5 and × 2, seed 0 | 2 |
| **Committed total (primary + supporting)** | | **106** |

### 5.3 Conditional extension U5: 20 runs

U5 adds U = 4,718,592 for 4 sizes × 3 seeds (12 runs), plus one-pass controls × 2 seeds (8 runs). Epochs would be XS 174.4, S 105.3, M 45.8 and L 23.4. U5 runs only if the go/no-go rule in section 9 allows it. Without it, L reaches only 11.7 epochs, and its E½ will probably be reported only as a lower bound.

## 6. Learning-rate selection

The sweep tests three peak LRs per size on fresh data at C/4. Scaling C down by a quarter keeps each size's relative horizon. The candidate values are:

| Size | Peak LRs |
|---|---|
| XS | 3e-3, 6e-3, 1.2e-2 |
| S | 2e-3, 4e-3, 8e-3 |
| M | 1.5e-3, 3e-3, 6e-3 |
| L | 1e-3, 2e-3, 4e-3 |

The LR is selected on **validation** loss. If the best value is at the edge of the range, one extra point is added from the reserve.

The LR × scarcity check tests whether heavy repetition changes the best LR. Small models repeat the most, so a shifted optimum would bias results against them. If either alternative beats the chosen LR by more than 2σ, this is reported as a limitation.

## 7. Evaluation and statistical analysis

### 7.1 Metrics and schedule

| Quantity | Data | When | Role |
|---|---|---|---|
| Test bits per byte (and nats/token) | 10M-token test set, non-overlapping 512-token windows | Final step; also at the best-validation checkpoint | Primary: final step. Secondary: best checkpoint |
| Validation loss | 1M-token validation slice | Every 5% of steps | LR selection, bracketing check, curves |
| Training loss on the run's own subset | Fixed 1M-token sample of U | Same steps as validation | Memorisation gap |
| Checkpoints | Modal Volume | Every 10% of steps | Resuming; best-checkpoint selection |

Bits per byte = (sum of token losses in nats) / (UTF-8 bytes × ln 2). It does not depend on the tokeniser, as the brief recommends.

### 7.2 Validation for selection, test for final evaluation

- **Validation only.** LR choice, best-checkpoint selection, the bracketing check (7.5), go/no-go, the decision to run U5, the LR × scarcity check, and all debugging.
- **Test.** Each run is evaluated once, automatically, and the result is logged. The analysis code is developed on validation numbers. Test numbers go into the analysis only after the pre-registered analysis plan has been merged and every primary run has finished.
- **No early stopping in primary results.** Primary results use the final step, so every run spends exactly C.

### 7.3 Recovery ratio

For size N, with B denoting test BPB:

```
B_fresh(N)  = T_N fresh tokens, one pass                  (condition U0, budget C)
B_rep(N,U)  = T_N tokens as E = T_N/U passes over U       (primary run,  budget C)
B_1p(N,U)   = U tokens, one pass                          (one-pass control, budget C*U/T_N)

P(N,U)   = B_rep(N,U) - B_fresh(N)                        penalty against fresh data
G(N,U)   = B_1p(N,U)  - B_rep(N,U)                        gain from repetition
rho(N,U) = G(N,U) / [ B_1p(N,U) - B_fresh(N) ]            recovery ratio
```

- **Meaning.** The denominator is the improvement that (T_N − U) extra *fresh* tokens buy over one pass of U. The numerator is the improvement that the same compute buys when spent on *repeats* of U. ρ is the share of the fresh-data benefit that repetition delivers. The fresh run is the ceiling and the one-pass control is the floor.
- **Edge cases.**
  - **U ≥ T_N:** there is no repetition, so ρ is undefined and not reported.
  - **Small denominator:** if the denominator is less than 3× its bootstrap standard error, the cell is marked "not estimable". L at U1, with 1.5 epochs, is expected to hit this.
  - **ρ > 1:** reported unclipped. It means noise, or a regularising effect of repetition.
  - **ρ < 0:** repetition did worse than one pass. This is reported as harmful repetition (overfitting).
- **Uncertainty.** Seeds share initialisation across the three terms, so seed indices are resampled jointly to keep that correlation. The one-pass controls, with 2 seeds, are resampled separately. The bootstrap uses 10,000 replicates and reports percentile 95% CIs. Where the denominator's CI approaches 0, a Fieller interval is reported as a check.
- **Threshold.** E½(N) is the epoch count at which ρ crosses 0.5, interpolated linearly in log E between the bracketing conditions. If ρ does not cross 0.5 inside the grid, E½ is reported as a bound.

### 7.4 Optimal model size per data condition

1. For each U, take the mean test BPB of each size over its 3 seeds.
2. **Parabola fit.** Fit BPB = a + b·x + c·x² with x = log N_non-emb, following Chinchilla's approach 2 `[paper]`. N\*(U) is the vertex. With 4 points there is 1 residual degree of freedom, which is reported. If c ≤ 0, or the vertex falls outside [XS, L], report "no interior optimum (edge or flat)".
3. **Discrete optimum.** Report the best size, and P_best: the bootstrap probability that each size is best.
4. **Uncertainty.** Resample seeds within each cell (10,000 replicates) and refit. Report CIs for log N\*, and for the best-minus-runner-up BPB gap, which measures flatness.
5. **Trend (H1).** Fit the inverse-variance-weighted slope of log N\* on log U across conditions, with a bootstrap CI. Separately, report the crossover U×, where the fresh-data optimum and the next smaller size swap order, interpolated in log U.
6. **Sensitivity.** Repeat steps 2 to 5 with total N in place of non-embedding N. Embedding shares range from 70% for XS to 25% for L. A conclusion counts only if it holds under both measures of N.

### 7.5 Pre-registered bracketing check

After seed 0 of U0 finishes, compare **validation** BPB across the four sizes.

- If L is best, add a larger size (9 layers, width 576, about 45M total parameters) at U0 and U1, funded from the reserve, before the other conditions run.
- If XS is best, stop and return to the group to raise C.

### 7.6 Pooled noise, power and pre-registration

- **Pooled noise.** Seed standard deviation σ_N is pooled per size from the primary runs: 5 cells × 2 = 10 degrees of freedom.
- **Power.** Assuming σ ≈ 0.005 to 0.01 nats `[assumed]`, a 3-seed mean has a standard error of 0.003 to 0.006. That makes differences of roughly 0.012 to 0.024 nats detectable. As an illustration only, plugging Chinchilla's constants into the data-constrained law of Muennighoff et al. predicts loss gaps between sizes of 0.03 to 0.3 nats across this grid `[assumed]`. Because those constants come from much larger models, only the predicted *direction* is used, never the size of the effect.
- **Pre-registration.** The hypotheses, criteria, exclusion rules and analysis code are merged through a reviewed PR before the primary grid starts. Anything not pre-registered is labelled exploratory.
- **R_D\*.** A per-size fit of Muennighoff's R_D\* (paper value about 15.4 `[paper]`) has only 4 to 5 points on an uneven epoch grid, so it is **exploratory**. Points where BPB(U) is worse than BPB(2U) beyond noise are excluded from the fit.

## 8. Compute and budget

### 8.1 Assumptions

| Item | Value | Tag |
|---|---|---|
| A100-40GB price | $0.000583/s = $2.10/h | `[looked up]` modal.com/pricing |
| CPU + memory (4 cores, 16 GiB) | $0.32/h, so $2.42/h in total | `[derived]` from looked-up unit prices |
| Starter-plan credits | $30/month per workspace, not transferable | `[looked up]` |
| Effective throughput | XS 20, S 30, M 55, L 80 TFLOPS (about 6% to 26% of the 312 TFLOPS peak) | `[assumed]` |
| Overhead per run (start-up, evaluation, checkpoints) | 120 s | `[assumed]` |
| Measured throughput | none yet; set by the pilot | — |

### 8.2 Per-run cost (assumed)

| Size | Train time | GPU-h/run | $/run | $ per condition (3 seeds) |
|---|---|---|---|---|
| XS | 20.8 min | 0.381 | 0.92 | 2.76 |
| S | 13.9 min | 0.265 | 0.64 | 1.92 |
| M | 7.6 min | 0.160 | 0.39 | 1.16 |
| L | 5.2 min | 0.120 | 0.29 | 0.87 |
| All 4 sizes, one condition | | 0.925 | 2.24 | 6.72 |

At fixed FLOPs, cost per run *falls* as size grows, because larger models use the GPU better. XS and S account for over half of the bill.

### 8.3 Budget and workspace allocation

| Block | Runs | GPU-h | $ | Workspace |
|---|---|---|---|---|
| Pilots (hard cap $1.50 each) | – | – | ≤ 3.00 | A, B |
| LR sweep (C/4) | 12 | 1.00 | 2.41 | A |
| Primary, seed 0 | 20 | 4.63 | 11.19 | A |
| Primary, seed 1 | 20 | 4.63 | 11.19 | B |
| Primary, seed 2 at U0 and U2 | 8 | 1.85 | 4.48 | A |
| Primary, seed 2 at U1, U3 and U4 | 12 | 2.78 | 6.72 | B |
| One-pass controls, seed 0 | 16 | 0.85 | 2.06 | A |
| One-pass controls, seed 1 | 16 | 0.85 | 2.06 | B |
| LR × scarcity check | 2 | 0.53 | 1.28 | A |
| **Committed** | **106** | **≈ 17.1** | **44.40** | A 22.92, B 21.47 |
| Reserve (debugging, reruns, bracketing size) | | | 15.60 | A 7.08, B 8.53 |
| U5 extension (conditional) | 20 | 3.06 | 7.41 | A: seed 0 (2.59). B: seeds 1–2 (4.83) |

Block costs are rounded; totals are computed from unrounded values. The committed main experiment budget is **$44.40 (provisional)**, which is 74% of the $60 combined allowance.

**Allocation rules**

- Seed 2 is split by *condition*, not by size. All four sizes of a condition always run in the same workspace, so the primary within-condition comparison never spans workspaces.
- Both workspaces use the same tagged commit, the same Modal image definition and the same GPU type per size. Each prepares its own data copy, and the token-array SHA-256 checksums must match before training starts.
- A launcher ledger refuses to start a run that would push its workspace past its cap. Each Modal function's timeout is 1.5× its measured runtime. Both workspaces keep their $0 extra-spend limit.
- Workspace A runs the LR sweep first. Workspace B starts its primary runs only after the LR choice has been merged.
- Each member submits their `results/*.jsonl` in their own PR, which the other member reviews.

## 9. Pilot, go/no-go and fallback rules

### 9.1 Pilot (hard cap $1.50 per workspace)

1. **Local and free:** a CPU smoke test on a toy corpus. Check that loss falls, exposure counts are exact, and resuming reproduces the loss.
2. **Workspace A:** data preparation on CPU. Then 200 timed steps of each size on A100-40GB, plus XS and S on L4 and A10, measuring tokens/s, overhead and memory. Finish with a short end-to-end M run.
3. **Workspace B:** rebuild the data and match its checksums. Benchmark M, which must match A within 10%. Repeat A's short M run, which must match within 0.01 nats at every evaluation.
4. Stop either workspace manually once its dashboard shows $1.50 spent.

### 9.2 Go/no-go (from measured throughput)

| Condition | Decision |
|---|---|
| Any correctness gate fails (checksums, exposure counts, resume, cross-workspace match) | **STOP.** Fix it and rerun only that pilot step, from the reserve |
| Projected committed cost ≤ $45, and each workspace ≤ $25 | **GO** |
| Projected committed cost plus U5 ≤ $48 | Also run **U5** (keeps at least $12 in reserve) |
| GO fails, but a fallback step fits | **GO-REDUCED:** apply the smallest fallback step that fits, and record it before the primary grid |
| Even F3 does not fit | **NO-GO:** redesign with the group |

**GPU choice.** Each size uses the GPU type with the lowest measured cost per FLOP, preferring A100-40GB when two are within 10%. Because the budget is in FLOPs, using different GPU types for different sizes does not bias the results. The same size uses the same GPU type in both workspaces.

### 9.3 Fallback ladder (cumulative; costs at assumed throughput)

| Step | Change | Saves | Lost |
|---|---|---|---|
| F1 | One-pass controls with 1 seed only | $2.06 | Precision of ρ |
| F2 | 2 seeds instead of 3 at U1 | $2.24 | Precision at U1 |
| F3 | C = 1.75e16 instead of 2.5e16 (token budgets × 0.7) | about 25% of the remainder | Changes the declared budget; must be decided before the primary grid |

Never cut: any of the four sizes, the U0 fresh condition, or the 3 seeds at U0, U2, U3 and U4. U5 is dropped before any fallback step.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Size, tokens and repetition are collinear across sizes (section 4) | Conclusions limited per section 4; the across-size repetition analysis is labelled exploratory |
| The optimum lands at the edge of the grid | XS added; pre-registered bracketing check (7.5) |
| The best LR changes under repetition | LR × scarcity check; reported as a limitation if it moves |
| One batch size for every size (L has only 1,684 steps) | Kept fixed on purpose to remove a variable; the LR sweep uses the same batch |
| The FLOP-counting convention is ambiguous | Formula fixed in 3.1; both the 6N term and the attention term logged |
| Fractional epochs | Exposure counts logged; the extra exposure is under 1/E of the data |
| Near-duplicate documents across crawl dumps | Exact deduplication, 13-gram audit, URL-host split fallback, duplicate rate reported |
| Tokeniser or validation leaking into test | Separate tokeniser split; test touched only for final evaluation (7.2) |
| Predictions from large-model papers do not transfer | Used only for the direction of effects; the grid spans 9.4M to 75M tokens (4.7M with U5) |
| Optimum is flat at small U | P_best and the gap reported; a flat curve is itself a finding |
| XS throughput is unknown (the largest budget line) | Pilot benchmarks XS first, on three GPU types |
| Environment pitfalls from the repo audit (torch 2.11.0 CUDA 13 wheels; `modal` missing from requirements; `.gitignore` lacks results/checkpoints) | Fixed in the first infrastructure PR and verified in the pilot |

## 11. Unresolved decisions requiring group approval

1. **Budget type:** a FLOP budget (recommended) or a GPU-time budget.
2. **Budget value:** C = 2.5e16 (recommended), or 3.5e16, which costs about 35% more.
3. **Sizes:** XS, S, M and L as in 3.2.
4. **Data grid:** U0 to U4 committed; U5 conditional.
5. **Outcomes:** N\*(U) and U× primary; ρ and E½ secondary; R_D\* and H4 exploratory.
6. **Follow-ups:** drop v0.2's F-REG and F-XL in favour of U5 and the LR × scarcity check.
7. **GPU type:** allow a different GPU type per size if the pilot shows it is cheaper per FLOP.
8. **Go/no-go thresholds:** $45 committed, $25 per workspace, U5 only if the total stays ≤ $48.
9. **Modal balances:** both members confirm their remaining credit and reset date (the plan assumes $30 each).
10. **Roles:** who runs workspace A and who runs workspace B.
11. **Pilot approval:** approve the pilot spend (≤ $1.50 per workspace).
12. **PR responsibilities:** who opens, reviews and merges each implementation PR.

## 12. Implementation sequence (not started)

Each step uses one feature branch and one PR, independently reviewed by the other member before merge.

1. **Environment:** add `modal`; pin a torch build that matches Modal's driver; extend `.gitignore`; fill in `README.md` and `authors.md`.
2. **Data:** `code/prepare_data.py` and `code/make_smoke_corpus.py`, with loader unit tests and checksum output.
3. **Training:** `code/model.py` and `code/train.py`, covering the FLOP-budgeted step count, exposure logging, checkpoints, evaluation and the cost ledger. Add `configs/smoke.json`.
4. **Analysis:** the pre-registered analysis plan, plus `code/analyze.py` tested on synthetic results.
5. **Pilot** (after approval), then a PR recording the go/no-go result and the declared per-run budget.
6. **Grid:** `configs/lr_sweep.json`, `configs/main_grid.json` and `code/run_grid.py`, with a workspace field on every run.

## 13. References

- Alexandre Thiery, DSA4212 Small LLM project brief: https://alexxthiery.github.io/teaching/small-llm/small-llm.html
- DSA4212 assignment guidelines: https://alexxthiery.github.io/teaching/guidelines/guidelines.html
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* arXiv:2203.15556.
- Muennighoff et al. (2023). *Scaling Data-Constrained Language Models.* arXiv:2305.16264.
- Karpathy, *nanoGPT*: https://github.com/karpathy/nanoGPT
- Hugging Face, *The Smol Training Playbook* (starter kit reference in the brief).
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361. Cited from memory; to verify.
- Morris et al. (2025). *How Much Do Language Models Memorize?* arXiv:2505.24832. Cited from memory; to verify before citing in the report.
- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Modal pricing: https://modal.com/pricing (read 23 Oct 2026)
