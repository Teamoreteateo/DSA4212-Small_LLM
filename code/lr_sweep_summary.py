"""Summarise the LR pilot (uses VALIDATION loss only; test set is untouched).
Usage: python code/lr_sweep_summary.py results/lr_sweep"""
import glob, json, os, sys
import pandas as pd

rows = []
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    r = json.load(open(f))
    rows.append(dict(model=os.path.basename(f).split("_U")[0], lr=r["config"]["lr"],
                     n_params_non_emb=r["n_params_non_emb"], final_val=r["final_val_loss"],
                     best_val=r["best_val_loss"], tokens_per_sec=r["tokens_per_sec"]))
df = pd.DataFrame(rows).sort_values(["n_params_non_emb", "lr"])
os.makedirs("results/tables", exist_ok=True)
df.to_csv("results/tables/lr_sweep.csv", index=False)
print(df.pivot_table(index="model", columns="lr", values="final_val").round(4))
print("\nthroughput (tokens/s):\n", df.groupby("model")["tokens_per_sec"].mean().round(0))