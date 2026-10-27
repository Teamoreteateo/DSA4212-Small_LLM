# Reading notes

Rule: each claim has a section/figure reference. Rewrite in your own words after reading.
AI-drafted starting points are marked [AI draft - verify & rewrite].

## Muennighoff et al. 2023, Scaling Data-Constrained Language Models (NeurIPS 2023) — reader: TODO
[AI draft - verify & rewrite]

### What they did
- 400+ GPT-2-style models, 10M to 9B params, up to 1500 epochs, on nested subsets of C4 (§1, §4, Fig. 2).
- Split total tokens D into unique tokens U_D and repetitions R_D = epochs - 1 (§3).
- Fit a data-constrained scaling law: effective data D' = U_D + U_D * R*_D * (1 - exp(-R_D / R*_D)),
  i.e. each repetition is worth less than fresh data, decaying exponentially (§3.1, Eq. 5).
  Symmetric term for "excess parameters" N' (Eq. 6).

### Key findings we rely on
1. Up to ~4 epochs: loss nearly the same as unique data (abstract; §6: 8.7B model, 4 epochs -> only
   0.5% higher val loss than 1 epoch).
2. Half-life R*_D ~ 15 -> meaningful gains up to ~16 epochs, then returns collapse (§6, Fig. 5).
3. Excess parameters lose value faster than repeated data (R*_N < R*_D) -> with limited data, put extra
   compute into more epochs, i.e. SMALLER models than Chinchilla predicts (§5, Fig. 1 right, Fig. 3).
   => basis for our H1.
4. Too many params/epochs eventually INCREASE loss; their law does not model this (§5, last para;
   §6 notes it underestimates loss of "failing" runs, e.g. 44 epochs).
   => basis for our H2 (overfitting onset).
5. For 100M unique tokens, the Chinchilla one-epoch optimum is ~7M params (§5). Our pilots: optimum
   m3/m4 (4.7M/10.6M) at U=100M -> same order of magnitude (different data/tokenizer, so only indicative).

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

## Hoffmann et al. 2022 (Chinchilla) — reader: TODO
- [verify] Compute-optimal: scale params and tokens equally (Muennighoff §2 confirms "alpha ~ beta").
- [verify] ~20 tokens/param (70B model on 1.4T tokens). CHECK whether their param count includes embeddings.
- Observation: at our budgets, tokens/param = m1 1083, m2 128, m3 30, m4 8.2, m5 1.8 (non-emb params).
  ~20 lies between m3 and m4, where our pilots found the optimum at U=100M.

## Kaplan et al. 2020 — reader: TODO
- [verify] power laws in N, D, C; non-embedding parameter counts (we follow this convention).

## Hernandez et al. 2022 (Repeated data) — reader: TODO
- Repeating a small fraction (0.1%) of data 100x significantly degrades performance (as cited in Muennighoff §8).
- Contrast: we (like Muennighoff) repeat the whole subset uniformly.

## Eldan & Li 2023 (TinyStories) — reader: TODO
- [verify] dataset designed so very small models produce fluent text -> justifies our dataset at this scale.

## nanoGPT / Smol Playbook — reader: TODO
- Pre-LN, weight tying, AdamW betas (0.9, 0.95), weight decay 0.1 on matrices only, cosine + warmup: TODO cite sources.

## Papers we found ourselves (required)
- TODO