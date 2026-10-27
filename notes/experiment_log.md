| 27 Oct | Tim | Smoke test, Modal T4, 8 runs | all pass; device=cuda, Tesla T4; resume-skip works | proceed |
| 27 Oct | Tim | prepare TinyStories | 100M train tok (1M stories), val 2.33M, test 2.56M, 3.92 bytes/tok, vocab 4096 | data fixed for all runs |
| 27 Oct | Tim | LR pilot 1: {3e-4,1e-3,3e-3} x m1,m3,m5 @U=100M | best m1 3e-3, m3 3e-3 (grid edge), m5 1e-3; m5@3e-3 diverged in 1st min (val 3.45) | single LR biases H1 -> per-size LR, pilot 2 |
| 27 Oct | Tim | Throughput T4 | m1 714k, m3 232k, m5 75k tok/s (9.6x vs ~64x params) | wall-clock budget under-serves small models; note in limitations |
| 27 Oct | Tim | m3 (3e-3) beats m5 (1e-3) at U=100M: val 1.685 vs 1.792 | compute-optimal size is inside the model grid | good for H1 |
| 27 Oct | Tim | Decided against U=400M row | m1 does ~4.3 epochs at U=100M | note as limitation |
| 27 Oct | Tim | Throughput variance across Modal T4 containers (pilot 1) | m5: 67k-83k tok/s (24% spread); m1, m3 within 3% | report tokens_seen per run; check spread in main grid; 3 seeds average it; mention in limitations |
| 27 Oct | Tim | LR pilot 2 (8 runs) | best LR: m1-m4 3e-3, m5 2e-3 (3e-3 diverges) | final LRs |
| 27 Oct | Tim | m4@2e-3 worse than both 1e-3 and 3e-3 | its container ran 109k tok/s vs 145-152k -> ~30% fewer steps | throughput confound |
| 27 Oct | Tim | Throughput spread across Modal T4s | m4 39%, m5 24%, m2 10%, m1 2% | switch to fixed per-model token budget = median tps x 600 s |
| 27 Oct | Tim | LR ranking at U=100M | m4 1.674 ≈ m3 1.685 < m5 1.763 < m2 1.783 < m1 2.056 | optimum interior (m3/m4) |