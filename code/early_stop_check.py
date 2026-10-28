"""Sensitivity check: model ranking per U using final vs best (early-stopped) validation loss.
Uses validation loss (nats/token), so it compares rankings, not test BPB.
Usage: python code/early_stop_check.py results/main"""
import glob, json, os, sys
import pandas as pd

run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/main"
rows = []
for f in glob.glob(os.path.join(run_dir, "*.json")):
    r = json.load(open(f)); c = r["config"]
    rows.append(dict(model=os.path.basename(f).split("_U")[0], U=c["unique_tokens"],
                     final_val=r["final_val_loss"], best_val=r["best_val_loss"]))
d = pd.DataFrame(rows).groupby(["U", "model"])[["final_val", "best_val"]].mean().reset_index()
os.makedirs("results/tables", exist_ok=True)
d.round(4).to_csv("results/tables/early_stop_check.csv", index=False)
for u, g in d.groupby("U"):
    print(f"U={u / 1e6:g}M  final: {g.sort_values('final_val').model.tolist()}  "
          f"early-stopped: {g.sort_values('best_val').model.tolist()}")
