# Analysis plan (written before the main grid, 27 Oct 2026)

Primary metric: test bits per byte (BPB), final checkpoint, mean over 3 seeds, 95% t-interval.

H1 (optimal model size shrinks as unique data U shrinks)
- Optimal model at each U = lowest mean test BPB.
- Supported if: the optimal model at U=1.5M (or 6M) is smaller than at U=100M, AND
  at that U, the optimal model's CI does not overlap the CI of the U=100M-optimal model.
- Partially supported: optimum shifts but CIs overlap -> report as a trend, not a finding.
- Not supported: same optimum at every U.

H2 (larger models stop benefiting from repetition sooner)
- Overfitting onset per run = epochs at minimum validation loss (from the training log),
  counted only if val loss later rises by > 0.005 nats.
- Supported if: at a fixed small U (1.5M, 6M), onset epochs decrease with model size.
- Secondary evidence: final val - train gap grows with model size at fixed U.

Not used for conclusions: validation set (tuning only); best-val checkpoint (reported as a sensitivity check only).

Divergence rule: a run is flagged diverged if its final val loss exceeds its cell's median by
> 0.5 nats. Diverged runs are reported, excluded from cell means, and rerun once with LR halved
(logged in experiment_log.md). Motivation: m5 diverged at 3e-3 in pilot 1; m4/m5 LRs are near
the stability edge.