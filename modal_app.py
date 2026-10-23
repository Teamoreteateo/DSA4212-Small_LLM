"""Run the project on Modal GPUs. Data and results live on a persistent Modal Volume.
  modal run modal_app.py::prepare --smoke                  # tiny synthetic data -> /vol/data_smoke
  modal run modal_app.py --config configs/smoke.json       # GPU smoke test
  modal run modal_app.py::prepare                          # TinyStories -> /vol/data
  modal run modal_app.py --config configs/main_grid.json   # grid, runs in parallel (<=10 GPUs)
  modal volume get small-llm-vol results .                 # download results/ locally
"""
import json, os, subprocess
import modal

app = modal.App("dsa4212-small-llm")
vol = modal.Volume.from_name("small-llm-vol", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "tokenizers", "datasets", "pandas", "matplotlib")
    .add_local_dir("code", remote_path="/root/code")  # keep add_local_dir last
)

@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=3 * 3600)
def prepare(smoke: bool = False, vocab_size: int = 4096, max_train_tokens: int = 100_000_000):
    if smoke:
        out = f"{VOL}/data_smoke"
        subprocess.run(["python", "/root/code/make_smoke_corpus.py", f"{out}/corpus.txt"], check=True)
        cmd = ["python", "/root/code/prepare_data.py", "--source", "local", "--local_path",
               f"{out}/corpus.txt", "--out_dir", out, "--vocab_size", "512", "--max_train_tokens", "200000"]
    else:
        cmd = ["python", "/root/code/prepare_data.py", "--out_dir", f"{VOL}/data",
               "--vocab_size", str(vocab_size), "--max_train_tokens", str(max_train_tokens)]
    subprocess.run(cmd, check=True)
    vol.commit()

@app.function(image=image, volumes={VOL: vol}, gpu="T4", cpu=4, memory=16384, timeout=1800, retries=1)
def train_one(cell: dict):
    vol.reload()  # see files written by other containers
    out = f"{VOL}/{cell['out']}"
    if os.path.exists(out):
        return f"skip {cell['out']}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["python", "/root/code/train.py", *cell["args"], "--out", out], check=True)
    vol.commit()
    return f"done {cell['out']}"

@app.local_entrypoint()
def main(config: str = "configs/main_grid.json", dry_run: bool = False):
    import sys
    sys.path.insert(0, "code")  # local import only
    from grid import build_cells
    cells = build_cells(json.load(open(config)), data_root=VOL, drop_device=True)
    print(f"{len(cells)} runs")
    if dry_run:
        for c in cells:
            print(c["out"])
        return
    for r in train_one.map(cells, return_exceptions=True):
        print(r)