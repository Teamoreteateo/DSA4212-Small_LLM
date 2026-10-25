"""Run a grid locally (or on Colab). Finished runs are skipped, so it is safe to relaunch."""
import argparse, json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from grid import build_cells

p = argparse.ArgumentParser(); p.add_argument("config"); p.add_argument("--dry_run", action="store_true")
args = p.parse_args()
cells = build_cells(json.load(open(args.config)))
print(f"{len(cells)} runs")
for c in cells:
    if os.path.exists(c["out"]):
        continue
    cmd = [sys.executable, os.path.join(here, "train.py"), *c["args"], "--out", c["out"]]
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)