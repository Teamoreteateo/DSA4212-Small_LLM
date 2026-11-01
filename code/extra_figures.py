"""Overfitting-onset figure for the report (H2), combining the main grid and the dropout follow-up.
Usage: python code/extra_figures.py   ->   report/figures/onset_vs_size.pdf"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

T, F = "results/tables", "report/figures"
os.makedirs(F, exist_ok=True)
main = pd.read_csv(f"{T}/main_key_numbers.csv").assign(dropout=0.0)
drop = pd.read_csv(f"{T}/followup_dropout_key_numbers.csv").assign(dropout=0.1)
both = pd.concat([main[main.U.isin(drop.U.unique())], drop])
colour = {6_000_000: "tab:blue", 1_500_000: "tab:red"}
line = {0.0: "-", 0.1: "--"}
marker = {0.0: "o", 0.1: "s"}

fig, ax = plt.subplots(figsize=(5, 3.6))
for (u, p), d in both.groupby(["U", "dropout"]):
    d = d.sort_values("n_params")
    ov = d.frac_overfit > 0.5                      # runs whose validation loss rose after its minimum
    ax.plot(d.n_params[ov], d.epochs_at_best_val[ov], ls=line[p], marker=marker[p], color=colour[u],
            label=f"U = {u/1e6:g}M, " + ("dropout 0.1" if p else "no dropout"))
    ax.scatter(d.n_params[~ov], d.epochs[~ov], marker=marker[p], facecolors="none", edgecolors=colour[u])
ax.set_xscale("log"); ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("non-embedding parameters"); ax.set_ylabel("overfitting onset (epochs)")
ax.set_title("Overfitting onset vs model size")
ax.legend(fontsize=7, title="open markers: never overfit\n(plotted at total epochs)", title_fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/onset_vs_size.pdf"); plt.close(fig)
print("wrote", f"{F}/onset_vs_size.pdf")