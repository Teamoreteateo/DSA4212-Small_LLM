"""Sensitivity check: model ranking per U using final vs best (early-stopped) validation loss.
Also converts validation loss to bits per byte using the validation set's bytes/token.
Usage: python code/early_stop_check.py results/main [data_dir]"""
import glob, json, math, os, sys
import numpy as np, pandas as pd

run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/main"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
bpt = None
try:
    v = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    bpt = float(np.load(os.path.join(data_dir, "token_bytes.npy"))[v].mean())
    print(f"validation bytes/token = {bpt:.4f}")
except FileNotFoundError:
    print("data/val.bin not found locally: skipping BPB conversion")

rows = []
for f in glob.glob(os.path.join(run_dir, "*.json")):
    r = json.load(open(f)); c = r["config"]
    rows.append(dict(model=os.path.basename(f).split("_U")[0], U=c["unique_tokens"],
                     final_val=r["final_val_loss"], best_val=r["best_val_loss"]))
d = pd.DataFrame(rows).groupby(["U", "model"])[["final_val", "best_val"]].mean().reset_index()
if bpt:
    d["best_val_bpb"] = d["best_val"] / (math.log(2) * bpt)
tag = os.path.basename(os.path.normpath(run_dir))
prefix = "" if tag == "main" else f"{tag}_"
d.round(4).to_csv(f"results/tables/{prefix}early_stop_check.csv", index=False)
for u, g in d.groupby("U"):
    print(f"U={u/1e6:g}M  final: {g.sort_values('final_val').model.tolist()}  "
          f"early-stopped: {g.sort_values('best_val').model.tolist()}")