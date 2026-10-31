# DSA4212-Small_LLM

Data-constrained training of small GPTs under a fixed compute budget (DSA4212 project).
Report: `report/report.pdf` (source: `report/report.tex`, bibliography `report/refs.bib`, figures `report/figures/`).

## Research question
Under a fixed per-run compute budget, how do (i) the optimal model size and (ii) the usefulness of repeating
data change as the number of unique training tokens U shrinks?

**Findings (summary).** Larger models start overfitting after fewer epochs, with and without dropout.
Without regularisation the optimal model moves non-monotonically (m4 at 100M/25M, m5 at 6M, m1 at 1.5M), and
early stopping reverses the shift. With dropout 0.1, the optimum at 6M returns to m4, while at 1.5M the
smallest model (m1) is clearly best and achieves the best result at that data level.

## Declared compute budget (fixed 27 Oct 2026, before the main grid)
Per-run budget: a fixed number of training tokens per model, equal to 600 s of median measured training
throughput on one NVIDIA Tesla T4 (Modal).

| Model | Non-emb. params | Median throughput (tok/s) | Token budget | LR |
|---|---|---|---|---|
| m1 | 0.40M | ~716k | 430M | 3e-3 |
| m2 | 1.78M | ~379k | 228M | 3e-3 |
| m3 | 4.74M | ~234k | 141M | 3e-3 |
| m4 | 10.6M | ~145k | 87M  | 3e-3 |
| m5 | 25.2M | ~76k  | 46M  | 2e-3 |

Throughput measured in the LR pilots (`results/tables/lr_sweep.csv`), timing training steps only. A measured
allowance replaced live timing because throughput for the same model varied by up to 24-39% across Modal hosts.
Excluded from the budget: evaluation, tokenizer training, LR pilots. All models trained from random
initialisation; no pretrained weights or embeddings.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # exact pins; Modal image uses Python 3.11
modal setup                             # one-time Modal login
python scripts/check_env.py
```

## Reproduce everything (run from the repo root)
```bash
# 1. Data (Modal volume "small-llm-vol")
modal run modal_app.py::prepare --smoke
modal run modal_app.py --config configs/smoke.json
modal run modal_app.py --config configs/smoke_tokens.json
modal run modal_app.py::prepare                       # TinyStories, 100M train tokens, 4096-token BPE

# 2. Learning-rate pilots (validation set only)
modal run modal_app.py --config configs/lr_sweep.json
modal run modal_app.py --config configs/lr_sweep2.json
modal volume get small-llm-vol results/lr_sweep results/ --force
python code/lr_sweep_summary.py results/lr_sweep

# 3. Main grid: 5 models x 4 U x 3 seeds = 60 runs
modal run modal_app.py --config configs/main_grid.json
modal volume get small-llm-vol results/main results/ --force

# 4. Dropout follow-up: 5 models x U in {6M, 1.5M} x 3 seeds = 30 runs
modal run modal_app.py --config configs/followup_regularisation.json
modal volume get small-llm-vol results/followup_dropout results/ --force

# 5. Analysis (local). early_stop_check needs data/val.bin and data/token_bytes.npy:
modal volume get small-llm-vol data/val.bin data/
modal volume get small-llm-vol data/token_bytes.npy data/
python code/analyze.py results/main --tag main
python code/key_numbers.py results/main
python code/early_stop_check.py results/main
python code/analyze.py results/followup_dropout --tag followup_dropout
python code/key_numbers.py results/followup_dropout
python code/early_stop_check.py results/followup_dropout
```
Finished runs are skipped, so any `modal run` command can be relaunched safely. Validation bytes/token
(3.958, used to convert early-stopped validation loss to BPB) is printed by `code/early_stop_check.py`.
Runs are not bit-exact across re-executions (GPU non-determinism); differences are within the reported intervals.

## Where each report item comes from
| Report item | File |
|---|---|
| Table 1 (models, budgets, epochs) | `configs/main_grid.json`, `results/tables/lr_sweep.csv`, `results/tables/main_summary.csv` |
| Learning-rate pilots (Section 3) | `results/tables/lr_sweep.csv` |
| Fig. 1 left / right | `report/figures/main_bpb_vs_size.pdf`, `report/figures/main_bpb_vs_epochs.pdf` |
| Table 2 (main test BPB) | `results/tables/main_summary.csv` |
| H1/H2 numbers (optimum, onset, validation rise) | `results/tables/main_key_numbers.csv` |
| Early-stopping check (Section 4.3) | `results/tables/early_stop_check.csv`, `results/tables/followup_dropout_early_stop_check.csv` |
| Table 3 and Section 4.4 (dropout) | `results/tables/followup_dropout_summary.csv`, `results/tables/followup_dropout_key_numbers.csv` |

The other figures in `report/figures/` (throughput, train/val gap, validation curves) are produced by
`code/analyze.py` as supporting diagnostics and are not used in the report.

## Repository layout
```
code/        prepare_data, model, train, grid, run_grid, analyze, key_numbers, early_stop_check, lr_sweep_summary
configs/     JSON grids (smoke, smoke_tokens, lr_sweep, lr_sweep2, main_grid, followup_regularisation)
modal_app.py Modal runner (T4 GPUs, persistent volume)
results/     raw run JSONs (lr_sweep, main, followup_dropout) and summary CSVs (tables/)
report/      report.tex, refs.bib, figures/, report.pdf
notes/       analysis_plan.md (pre-registered), experiment_log.md, reading_notes.md, report_outline.md
scripts/     check_env.py
```

## AI assistance
Claude (Anthropic) was used to draft the code, the reading notes and some of the report text, which we
reviewed, edited and checked against the result files. See Section "Use of AI tools" in the report.
