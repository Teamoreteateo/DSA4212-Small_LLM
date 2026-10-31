# Report outline (bullet points to write from)

Write each section in your own words from these bullets, replacing the current text in
`report/report.tex`. Numbers come from `results/tables/`; keep them exactly. Target lengths keep the
report at 7 pages including references (current layout: 11pt, 2.2 cm margins).

## Declarations (keep as is, ~8 lines)
- Budget: fixed training tokens per model = 600 s of median T4 throughput; eval, tokenizer, pilots excluded; declared 27 Oct.
- From scratch: random init, no pretrained weights/embeddings; tokenizer trained by us.
- Eval: test BPB on second half of the official TinyStories validation split; never used for tuning.

## 1 Introduction (~2/3 page)
- Scaling laws (Kaplan, Chinchilla) split budget between size and tokens assuming fresh data.
- Data runs out -> spend compute on more epochs or a bigger model?
- Muennighoff: ~4 epochs nearly as good as new data; under data limits, prefer epochs over parameters.
- Our question (quote box): under a fixed per-run budget, how do optimal size and value of repetition change as unique tokens U shrink?
- H1: optimal size shrinks as U shrinks. H2: bigger models stop benefiting from repetition sooner. Both fixed before the grid.
- Preview: 5 sizes (0.4M-25M), U 1.5M-100M, 3 seeds, dropout follow-up. H2 supported; H1 only partly (non-monotonic without regularisation; with dropout unchanged at 6M, smallest model at 1.5M).

## 2 Background (~1/2 page, 3 short paragraphs)
- Kaplan: power laws; non-embedding parameter count (we follow). Chinchilla: scale params and tokens equally, ~20 tokens/param (their Table 3; counts include embeddings); analysis runs < 1 epoch.
- Muennighoff: repeated tokens worth exponentially less; half-life ~16 epochs, worthless ~40; excess params decay faster -> smaller models, more epochs (H1); too many epochs increase loss, not in their law (H2). Hernandez: repeating a small subset is especially harmful; we repeat the whole subset. Nakkiran: test error can be non-monotonic in size/epochs.
- TinyStories: models < 10M params write fluent English -> suits 10-min runs. Code follows nanoGPT. We differ: 1000x smaller, synthetic text, device-time budget.

## 3 Experimental setup (~1.5 pages incl. Table 1)
- Data: TinyStories; 100M-token training pool; U in {100, 25, 6, 1.5}M = first U tokens (nested, like Muennighoff Fig. 2). Val (2.33M tok) for tuning/curves, test (2.56M) for reported numbers.
- Repetition: whole subset, non-overlapping 256-token windows, reshuffled each epoch.
- Tokenizer: byte-level BPE, 4096 vocab; small vocab keeps embeddings from dominating (m1 with 50k vocab: ~6.4M emb vs 0.40M non-emb). 3.92 bytes/token train, 3.96 val.
- Models m1-m5 (Table 1): pre-LN GPT, tied embeddings, GELU 4d, context 256, GPT-2 init (nanoGPT). Dropout 0 except follow-up.
- Optimisation: AdamW (0.9, 0.95), wd 0.1 on matrices, clip 1.0, batch 64x256, fp16; 2% warmup then cosine to 10%, defined on fraction of budget. LR per model from 17 pilot runs on val: 3e-3 m1-m4, 2e-3 m5 (diverged at 3e-3).
- Budget: tokens = 600 s x median T4 throughput; switched from live timing because host speed varied 24-39%. Device time, not FLOPs: FLOPs range 1.0e15 (m1) to 7.0e15 (m5).
- Grid: 5 x 4 x 3 = 60 runs; epochs = budget / U (Table 1 columns); spans <4 epochs, ~16, far beyond.
- Evaluation: test BPB formula, first 2M test tokens, final checkpoint, no early stopping (like Muennighoff); losses logged every 30 s; 3 seeds, 95% t-intervals; decision rules pre-registered (notes/analysis_plan.md); no divergences (seed sd <= 0.015).

## 4 Results (~2.75 pages incl. Fig. 1, Tables 2-3)
### 4.1 H1
- 100M: m4 0.604 and m3 0.605 tie; 8 and 30 tokens/param bracket Chinchilla's ~20; m1 and m5 worse. 25M same ranking (m4 0.615, m3 0.618).
- 100M -> 25M costs 0.5% (m5), 0.7% (m1), 1.8-2.0% (m2-m4): consistent with ~4 epochs nearly free.
- 6M: optimum jumps UP to m5 (0.741); m1/m2 ~0.79; m3/m4 worst (0.88/0.87).
- 1.5M: optimum m1 (1.307); m2 3.04 ~ uniform guess (3.06 = log2 4096 / 3.92).
- Pre-registered rule: H1 supported (m1 at 1.5M vs m4 at 1.5M: 1.307 +/- 0.015 vs 2.433 +/- 0.019), but path is non-monotonic.
### 4.2 H2
- Onset = epochs at min val loss (counted if later rise > 0.005 nats).
- 6M: 23.2, 12.4, 8.2, 5.7 epochs for m2-m5; m1 never overfits in 72 epochs.
- 1.5M: 28.5, 15.8, 9.1, 8.8, 7.3 for m1-m5; val rises 0.9 (m1) to 5.7 (m2) nats afterwards.
- Supported. Caveat: resolution ~1/20 of a run (fine at 6M; m3-m5 indistinguishable at 1.5M). Final gap not monotonic (depends on epochs past onset).
### 4.3 Mechanism
- Fixed device time: bigger = slower = fewer epochs.
- 6M: m3/m4 run 24/15 epochs vs onsets 12/8 -> lose 0.40 nats; m5 runs 7.7 vs onset 5.7 -> loses 0.08. Largest wins because slowness protects it; m1/m2 too small to memorise.
- Early-stop check (best val loss): ranking unchanged at 100M/25M; at 6M m5>m4>m3>m2>m1; at 1.5M m4~m5 best, m1 last. Shrinking optimum depends on training to the end.
### 4.4 Dropout follow-up
- Dropout 0.1, all 5 models at 6M and 1.5M, same budgets/LRs, 30 runs (pre-registered m3, m5; extended).
- 6M: m2-m5 improve 0.06-0.19 BPB; overfitting nearly gone (rise <= 0.006 nats); optimum back to m4 (0.681) ahead of m5 (0.685); dropout beats early stopping (best early-stopped ~0.72); dropout hurts m1 (0.789 -> 0.802).
- 1.5M: every model improves 33% (m1) to 70% (m2); ranking ~monotonic: m1 0.874 < m2 0.921 < m3 1.151 < m5 1.290 < m4 1.355; m1 beats best early-stopped no-dropout model (m4 ~0.94 val-converted).
- H2 under dropout: onset still falls with size at 1.5M (37.9, 17.2, 13.9, 10.1 for m2-m5); val still rises 0.1-1.3 nats.

## 5 What did not work, and limitations (~1/2 page)
- Wall-clock budget: host throughput varied 24% (m5) / 39% (m4); slow-host m4 pilot did ~30% fewer steps; noise rivalled effects (m3 vs m4 differ 0.001) -> token budgets.
- m5 diverged at 3e-3 (val 3.45 vs 1.76 at 2e-3). First LR pilot optimum at grid edge -> second pilot.
- Dropout reruns of m3-m5 differ slightly (m5 1.5M 1.288 vs 1.290), within CIs; GPU non-determinism; report later runs.
- Limits: one simple dataset; tiny scale; device-time budget depends on hardware (FLOP-matched would favour small models); LRs tuned at 100M without dropout; one dropout rate, not at 100M/25M; onset resolution; no CIs on early-stop check; 3 seeds; final-checkpoint metric drives unregularised H1.

## 6 Use of AI tools (~6 lines; keep as written in report.tex after this change)

## 7 Conclusion (~5 sentences)
- H2 holds with and without dropout.
- H1 depends on overfitting control: non-monotonic without regularisation (size/epoch coupling); with dropout unchanged at 6M, smallest model only at 1.5M (best result overall at that U).
- Open: FLOP-matched budget; harder data.
