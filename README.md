# DSA4212-Small_LLM

Training small language models from scratch for DSA4212. Experiments run on
[Modal](https://modal.com); each teammate uses their own Modal workspace with the
same code and configuration.

## Environment

| Component | Version | Where it is set |
| --- | --- | --- |
| Python | 3.11 | `PYTHON_VERSION` in `modal_app.py`; use 3.11 locally too |
| PyTorch | 2.11.0 (Linux wheel built for CUDA 13.0) | `requirements.txt` |
| Modal client | 1.6.2 | `requirements-dev.txt` (local only) |

- `requirements.txt`: experiment dependencies. The Modal image installs exactly this file.
- `requirements-dev.txt`: `requirements.txt` plus the Modal client, for your own machine.

On Windows and macOS, pip installs the CPU build of PyTorch 2.11.0, which is enough for
local checks; GPU training happens on Modal. Intel Macs have no PyTorch 2.11 wheel.

## Setup

Clone the team repository (not a fork):

```bash
git clone https://github.com/Teamoreteateo/DSA4212-Small_LLM.git
cd DSA4212-Small_LLM
```

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python scripts/check_env.py
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python scripts\check_env.py
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

`scripts/check_env.py` runs offline and costs nothing. It checks that dependencies are
exact pins and match what is installed, that the local Python matches the Modal image,
that all Python files compile, that non-empty configs parse, and that no credentials,
checkpoints or large files are tracked. Empty config placeholders are skipped.

## Modal (one workspace per teammate)

Nothing in the repository names a workspace or holds a token. Modal uses whichever
profile is active on your machine, so the same commands run in your own workspace.

```bash
modal setup              # once: opens a browser and stores a token in ~/.modal.toml
modal profile current    # confirm this is YOUR workspace before running anything
```

Alternatively set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in your shell. Never commit
tokens; `.modal.toml` and `.env*` are git-ignored.

The app (`dsa4212-small-llm`) and volume (`small-llm-vol`) names are scoped to each
workspace, so the two workspaces do not collide. Results written to one workspace's
volume must be downloaded (`modal volume get small-llm-vol results .`) and committed
to be shared.

Every `modal run` spends credits. The cheapest check of the image is
`modal run modal_app.py::check_image` (CPU only). Run Modal commands from the repository
root, because the image reads `requirements.txt` and `code/` from there.

## Configs

Each experiment is one JSON file in `configs/`. `code/grid.py` runs every combination of
`models` × `unique_tokens` × `seeds` (× `lrs` if present) and passes each key in `common`
to `code/train.py` as a `--flag`. A key on a model entry overrides `common` for that model,
e.g. its own learning rate from the LR pilot:

```json
{"name": "m3", "n_layer": 6, "n_head": 8, "n_embd": 256, "lr": 0.003}
```

`lrs` (used by `lr_sweep.json`) overrides both. Preview the runs without training:
`python code/run_grid.py configs/main_grid.json --dry_run`.

## What is tracked

Tracked: code, `configs/`, notes, the report, and small numerical results (JSON/CSV under
`results/`). Ignored: datasets and token arrays (`data/`, `*.bin`, `*.npy`), checkpoints
(`*.pt`, `*.ckpt`, `*.safetensors`, `checkpoints/`), credentials, caches and temporary
outputs. See `.gitignore`.

## Git workflow

Never commit to `main` directly. Every change goes through a pull request that the
other teammate reviews before merging.

```bash
git checkout main
git pull                          # ALWAYS before starting
git checkout -b <type>/<short-name>   # e.g. feat/data-prep, fix/val-split
# ... work ...
git add <files>
git commit -m "feat: short description"   # keep under ~50 characters
git push -u origin <type>/<short-name>    # -u only on the first push
```

Then open a pull request into `main` on GitHub and ask your teammate to review it.
The author does not merge their own PR before review. After it is merged:

```bash
git checkout main
git pull
git branch -d <type>/<short-name>
```
