# Reading notes

Each claim has a section/figure reference.
Status (30 Oct): every claim below was checked against the paper text by Claude (PDFs from arXiv,
nanoGPT from GitHub). Short verbatim quotes are in "quotes". Points where the paper differs from an
earlier report draft are marked [Resolved]; the current report follows the paper on each.

## Muennighoff et al. 2023, Scaling Data-Constrained Language Models (NeurIPS 2023), arXiv 2305.16264

### What they did
- "more than 400 models ranging from 10 million to 9 billion parameters for up to 1500 epochs" (§1);
  §4 gives the largest as 8.7B. Data: "subsets of C4" (§4), nested so that "runs using less data (more
  epochs) always use a subset of the data used in" runs with more data (Fig. 2 caption).
- Split total tokens D into unique tokens U_D and repetitions R_D = epochs - 1 (§3).
- Effective data D' = U_D + U_D * R*_D * (1 - exp(-R_D / R*_D)) (§3.1, Eq. 5): each repetition is worth
  exponentially less than fresh data. Symmetric term for "excess parameters" (Eq. 6).
- Fitted values: "R*_N = 5.309743 and R*_D = 15.387756" (App. A).
- "we do not use early stopping to also explore the extent of overfitting when repeating" (§4);
  "cosine learning rate schedules that decay 10x over the course of training" (§4).
- "we always repeat the entire available data rather than subsets of it" (§4), contrasting with Hernandez.

### Key findings we rely on
1. "training with up to 4 epochs of repeated data yields negligible changes to loss" (abstract). §6: the
   8.7B model at 4 epochs had "only 0.5% higher validation loss than the single-epoch model".
2. "R*_D ~ 15, corresponding to 15 repetitions (or 16 epochs)" (§6): the half-life of repeated data.
   Value keeps decaying after that: "At 40 epochs, repeating is worthless" (Fig. 1 left) and "the value of
   adding compute eventually decays to zero" (abstract). [Resolved] Returns fall by about 1/e by ~16 epochs and are near zero by ~40;
   the report states it this way (Background).
3. "Since R*_D > R*_N, excess parameters decay faster" (App. A); "Allocation is optimized by scaling epochs
   faster than parameters" (§5). With limited data: smaller models, more epochs than Chinchilla. => H1.
4. "Adding parameters and epochs causes the loss to decrease and eventually increase again" (§5); their law
   "significantly underestimates the final test loss of failing models where loss increases midway through"
   (§6; 44-epoch runs). => H2.
5. For 100M unique tokens the one-epoch compute-optimal model "has U_N of approximately 7M parameters" (§5,
   using the Chinchilla laws). Our optimum at U=100M is m3/m4 (4.7M/10.6M non-emb.): same order of magnitude,
   only indicative (different data and tokenizer; Chinchilla counts embeddings, we do not).

### Design choices we copied (and why)
| Their choice (§4) | Ours | Why it matters |
|---|---|---|
| Nested data subsets (Fig. 2) | U = first-U-token prefix of one shuffled pool | differences between U levels come from amount, not content |
| Repeat ENTIRE subset, reshuffle each epoch | EpochLoader: non-overlapping windows, reshuffled per epoch | exact epoch counts; avoids Hernandez-style partial up-sampling |
| Cosine LR, 10x decay | cosine to 10% (min_lr_frac=0.1) | schedule completes in every run |
| No early stopping | final checkpoint for test BPB | lets us observe overfitting (H2) |
| Held-out test loss, not train loss | test BPB on held-out split | train loss is misleading under repetition |

### How our setting differs (important for the Discussion)
- Scale: our models 0.4M-25M params, ~3 orders of magnitude below theirs; budgets of minutes, not GPU-months.
- Budget: they match FLOPs (6ND); we match DEVICE TIME (600 s of T4 throughput). Small models use the
  GPU less efficiently, so per run our FLOPs range ~1.0e15 (m1) to ~7.0e15 (m5).
- Data: TinyStories (simple, synthetic, low-entropy) vs C4 (web text). Overfitting may set in at different epochs.
- Tokenizer: our 4096-token BPE vs GPT-2's 50k vocab.

### Open questions for our project
- Does the ~4-epoch "free repetition" result hold for tiny models on simple text? (our U=25M row:
  m3 5.6 ep, m4 3.5 ep, m5 1.8 ep)
- Does the overfitting onset (epochs) shrink with model size? (H2; their law does not predict this)

## Hoffmann et al. 2022 (Chinchilla), arXiv 2203.15556
- "for compute-optimal training, the model size and the number of training tokens should be scaled equally"
  (abstract). Chinchilla: 70B parameters, 1.4T tokens (Table 1).
- [Resolved] The phrase "20 tokens per parameter" does not appear in the paper. It is derived: Table 3 gives
  1B params -> 20.2B tokens, and 1.4T/70B = 20. The report cites Table 3.
- [Resolved] "we also count embeddings matrices in the total parameter count" (App. F). Chinchilla counts
  embeddings; we (following Kaplan) do not. The report says so and gives the with-embedding counts (footnote, 4.1).
- "the training runs for our analysis have all been trained on less than an epoch of data" (§5), though
  Table A1 notes MassiveWeb and Wikipedia subsets were used for more than one epoch.
- Schedule: "when the cosine cycle overshoots the number of training steps by more than 25%, performance is
  noticeably degraded"; LR "drops by a factor of 10x during training, using a cosine schedule" (App. D.1).
  Our schedule is defined on the fraction of budget used, so it always ends exactly at the end of training.
- Our numbers: tokens/param m1 1083, m2 128, m3 30, m4 8.2, m5 1.8 (non-emb.). ~20 lies between m3 and
  m4, where the U=100M optimum is.

## Kaplan et al. 2020, Scaling Laws for Neural Language Models, arXiv 2001.08361
- "Performance has a power-law relationship with each of the three scale factors N, D, C when not
  bottlenecked by the other two" (§1.1). Exponents (§1.2): alpha_N ~ 0.076, alpha_D ~ 0.095.
- Non-embedding parameters: "this produces significantly cleaner scaling laws" (§2.1); with embeddings
  included "the trend is somewhat obscured" (§3.2). We follow this convention.
- Overfitting: "The performance penalty depends predictably on the ratio N^0.74/D" (§1.1). Relevant to H2:
  bigger models overfit a fixed dataset sooner.
- "Large models are more sample-efficient than small models" (§1.1), and the compute-optimal recipe is
  "training very large models and stopping significantly short of convergence" (§1.1). Chinchilla later
  shifted this towards more tokens per parameter.
- [Resolved] Kaplan does study data-limited training (the N^0.74/D result), so the report attributes the
  under-one-epoch point to Chinchilla's analysis runs only.

## Hernandez et al. 2022, Scaling Laws and Interpretability of Learning from Repeated Data, arXiv 2205.10487
- "by repeating 0.1% of the data 100 times" (abstract), with repeats making up 10% of training tokens,
  "an 800M parameter model can be degraded" to the performance of "a 2x smaller model (400M params)"
  (abstract).
- Mechanism: "there is a range in the middle where the data can be memorized" (intro); repeated data "can
  lead test loss to increase midway through training" (abstract), a double-descent effect; "repeated data
  disproportionately damages induction heads" (intro).
- Contrast with us: they up-sample a small subset; we (like Muennighoff) repeat the whole subset uniformly,
  reshuffled each epoch. Their "memorisation" lens fits m2 at 1.5M ending near uniform-guess BPB.

## Eldan & Li 2023, TinyStories, arXiv 2305.07759
- Generated by "GPT-3.5 and GPT-4" (§2) from "a vocabulary consisting of about 1500 basic words" (§2),
  using only words "a typical 3-year-old child would understand" (§2).
- Models "below 10 million total parameters" produce fluent, consistent stories (abstract); §5 notes this
  works even with "only one transformer block". Justifies the dataset for our 0.4M-25M models.
- Their main evaluation is GPT-4 grading (grammar, creativity, consistency; §3); we use BPB only.
- Limitation for us: simple, synthetic, low-entropy text; overfitting onset may differ on web text.

## nanoGPT (Karpathy), github.com/karpathy/nanoGPT
- Checked in model.py / train.py: pre-LayerNorm blocks (x + attn(ln_1(x))), weight tying
  (wte.weight = lm_head.weight), init normal std 0.02 with residual projections scaled by 1/sqrt(2*n_layer),
  weight decay only on parameters with dim >= 2, beta2 = 0.95, cosine LR with linear warmup, min_lr ~ lr/10
  ("per Chinchilla" in the code comments). Our model.py and train.py follow these.
- The README now marks nanoGPT as "very old and deprecated" in favour of nanochat (Nov 2025 note); citing it
  for the architecture is still fine.

## Papers we found ourselves (required)
- Nakkiran et al. 2021, Deep Double Descent, arXiv 1912.02292 (already in refs.bib)
  - "as we increase model size, performance first gets worse and then gets better" (abstract); double descent
    also occurs "as a function of the number of training epochs" (abstract); defines "effective model
    complexity" (abstract, §1); finds regimes where "increasing (even quadrupling) the number of train
    samples actually hurts test performance" (abstract).
  - Relation to us: our 6M row (m3/m4 worst, m1/m2 and m5 better) is non-monotonic in size, but in our
    setup size and epochs change together, so the mechanism is not the same. Report frames it that way.
