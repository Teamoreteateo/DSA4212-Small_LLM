"""Key numbers for the Results section, computed from run JSONs (no plotting).
Usage: python code/key_numbers.py results/main
Writes results/tables/<tag>_key_numbers.csv and prints a summary."""
import glob, json, os, sys
import numpy as np, pandas as pd

OVERFIT_TOL = 0.005  # nats: val loss must rise this much after its minimum to count as overfitting

run_dir = sys.argv[1]
tag = os.path.basename(os.path.normpath(run_dir))
rows = []
for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
    r = json.load(open(f)); c = r["config"]; log = r["log"]
    i_best = int(np.argmin([e["val_loss"] for e in log]))
    rise = log[-1]["val_loss"] - log[i_best]["val_loss"]
    rows.append(dict(model=os.path.basename(f).split("_U")[0], U=c["unique_tokens"], seed=c["seed"],
                     n_params=r["n_params_non_emb"], epochs=r["epochs"], test_bpb=r["test_bpb"],
                     tokens_per_param=r["tokens_seen"] / r["n_params_non_emb"],
                     epochs_at_best_val=log[i_best]["epochs"], overfit=rise > OVERFIT_TOL, val_rise=rise))
df = pd.DataFrame(rows)
s = df.groupby(["U", "model"]).agg(
    n_params=("n_params", "first"), epochs=("epochs", "mean"), tokens_per_param=("tokens_per_param", "mean"),
    bpb=("test_bpb", "mean"), bpb_sd=("test_bpb", "std"), n=("seed", "count"),
    epochs_at_best_val=("epochs_at_best_val", "mean"), frac_overfit=("overfit", "mean"),
    val_rise=("val_rise", "mean")).reset_index()
tcrit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}
s["ci95"] = [sd * tcrit.get(n, 2.0) / np.sqrt(n) if n > 1 else np.nan for sd, n in zip(s.bpb_sd, s.n)]
s = s.sort_values(["U", "n_params"], ascending=[False, True])
os.makedirs("results/tables", exist_ok=True)
s.to_csv(f"results/tables/{tag}_key_numbers.csv", index=False)

print("=== H1: optimal model per U (is it clearly better than the runner-up?) ===")
for u, d in s.groupby("U", sort=False):
    d = d.sort_values("bpb"); b, r2 = d.iloc[0], d.iloc[1]
    sep = (b.bpb + b.ci95) < (r2.bpb - r2.ci95)
    print(f"U={u/1e6:>6g}M  best={b.model} ({b.bpb:.4f} ± {b.ci95:.4f}, {b.tokens_per_param:.1f} tok/param, "
          f"{b.epochs:.2f} ep)  runner-up={r2.model} ({r2.bpb:.4f})  CIs separated: {sep}")
print(f"\n=== H2: overfitting onset (epochs at min val; overfit = val rose > {OVERFIT_TOL} nats after min) ===")
print(s[["U", "model", "epochs", "epochs_at_best_val", "frac_overfit", "val_rise"]].round(3).to_string(index=False))