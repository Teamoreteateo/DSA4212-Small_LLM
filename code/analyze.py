"""Aggregate run JSONs into tables and report figures.
Usage: python code/analyze.py results/main --tag main"""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def load(run_dir):
    rows, curves = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        r = json.load(open(f)); c = r["config"]
        name = os.path.basename(f).split("_U")[0]
        rows.append(dict(model=name, U=c["unique_tokens"], seed=c["seed"], dropout=c["dropout"],
                         n_params_non_emb=r["n_params_non_emb"], n_params=r["n_params"],
                         tokens_seen=r["tokens_seen"], epochs=r["epochs"], tokens_per_sec=r["tokens_per_sec"],
                         train_loss=r["final_train_loss"], val_loss=r["final_val_loss"],
                         best_val_loss=r["best_val_loss"], test_bpb=r["test_bpb"], hardware=r["hardware"]))
        for e in r["log"]:
            curves.append(dict(model=name, U=c["unique_tokens"], seed=c["seed"], **e))
    return pd.DataFrame(rows), pd.DataFrame(curves)

def summarise(df):
    df = df.assign(gap=df["val_loss"] - df["train_loss"])
    s = df.groupby(["model", "U"]).agg(
        n_params_non_emb=("n_params_non_emb", "first"), epochs=("epochs", "mean"),
        tokens_seen=("tokens_seen", "mean"), test_bpb_mean=("test_bpb", "mean"),
        test_bpb_std=("test_bpb", "std"), n_seeds=("seed", "count"), gap_mean=("gap", "mean")).reset_index()
    tcrit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}  # 95% t critical values
    s["test_bpb_ci95"] = [st * tcrit.get(n, 2.0) / np.sqrt(n) if n > 1 else np.nan
                          for st, n in zip(s["test_bpb_std"], s["n_seeds"])]
    return s.sort_values(["U", "n_params_non_emb"])

def fmt_tokens(u):
    return f"{u / 1e6:g}M" if u >= 1e6 else f"{u / 1e3:g}K"

def main():
    p = argparse.ArgumentParser(); p.add_argument("run_dir")
    p.add_argument("--fig_dir", default="report/figures"); p.add_argument("--table_dir", default="results/tables")
    p.add_argument("--tag", default="main"); a = p.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True); os.makedirs(a.table_dir, exist_ok=True)
    df, curves = load(a.run_dir)
    if df.empty:
        raise SystemExit(f"no runs in {a.run_dir}")
    s = summarise(df)
    df.to_csv(os.path.join(a.table_dir, f"{a.tag}_runs.csv"), index=False)
    s.to_csv(os.path.join(a.table_dir, f"{a.tag}_summary.csv"), index=False)
    best = s.loc[s.groupby("U")["test_bpb_mean"].idxmin(), ["U", "model", "n_params_non_emb", "epochs", "test_bpb_mean"]]
    best.to_csv(os.path.join(a.table_dir, f"{a.tag}_optimal_model_per_U.csv"), index=False)
    print(s.to_string(index=False)); print("\nOptimal model per U:\n", best.to_string(index=False))

    Us = sorted(s["U"].unique(), reverse=True)
    models = s.sort_values("n_params_non_emb")["model"].unique()
    ucol = lambda i: plt.get_cmap("viridis")(i / max(1, len(Us) - 1))
    mcol = lambda j: plt.get_cmap("plasma")(j / max(1, len(models) - 1))
    def save(fig, n):
        fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, f"{a.tag}_{n}.pdf")); plt.close(fig)

    # Fig: test BPB vs model size, one line per U (H1); circle = optimum
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, u in enumerate(Us):
        d = s[s.U == u]
        ax.errorbar(d.n_params_non_emb, d.test_bpb_mean, yerr=d.test_bpb_ci95, marker="o", capsize=3,
                    color=ucol(i), label=f"U = {fmt_tokens(u)}")
        k = d.test_bpb_mean.idxmin()
        ax.scatter(d.loc[k, "n_params_non_emb"], d.loc[k, "test_bpb_mean"], s=160, facecolors="none", edgecolors=ucol(i))
    ax.set_xscale("log"); ax.set_yscale("log")  # log y: U=1.5M (up to ~3 BPB) would flatten the rest
    ax.set_yticks([0.6, 0.8, 1, 1.5, 2, 3], ["0.6", "0.8", "1", "1.5", "2", "3"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("non-embedding parameters"); ax.set_ylabel("test bits per byte")
    ax.legend(fontsize=8); ax.set_title("Fixed per-model budget: loss vs model size"); save(fig, "bpb_vs_size")

    # Fig: test BPB vs epochs, per model (H2)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for j, m in enumerate(models):
        d = s[s.model == m].sort_values("epochs")
        ax.errorbar(d.epochs, d.test_bpb_mean, yerr=d.test_bpb_ci95, marker="o", capsize=3, color=mcol(j), label=m)
    ax.set_xscale("log"); ax.set_xlabel("epochs over unique data"); ax.set_ylabel("test bits per byte")
    ax.legend(fontsize=8); ax.set_title("Repetition vs final loss"); save(fig, "bpb_vs_epochs")

    # Fig: val - train gap during training, one panel per U (memorisation)
    fig, axes = plt.subplots(1, len(Us), figsize=(3 * len(Us), 3), sharey=True, squeeze=False)
    for ax, u in zip(axes[0], Us):
        for j, m in enumerate(models):
            c = curves[(curves.U == u) & (curves.model == m)].copy()
            if c.empty:
                continue
            c["gap"] = c.val_loss - c.train_loss
            g = c.groupby("train_time")[["epochs", "gap"]].mean()
            ax.plot(g.epochs, g.gap, color=mcol(j), label=m)
        ax.axhline(0, color="grey", lw=0.5); ax.set_xscale("log")
        ax.set_title(f"U = {fmt_tokens(u)}", fontsize=9); ax.set_xlabel("epochs")
    axes[0][0].set_ylabel("val loss - train loss (nats)"); axes[0][-1].legend(fontsize=7); save(fig, "gap_vs_epochs")

    # Fig: validation curves at smallest U (mean +- std over seeds)
    u = Us[-1]; fig, ax = plt.subplots(figsize=(5, 3.6))
    for j, m in enumerate(models):
        c = curves[(curves.U == u) & (curves.model == m)]
        if c.empty:
            continue
        g = c.groupby("train_time")["val_loss"].agg(["mean", "std"]).reset_index()
        ax.plot(g.train_time, g["mean"], color=mcol(j), label=m)
        ax.fill_between(g.train_time, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                        color=mcol(j), alpha=0.2)
    ax.set_xlabel("training time (s)"); ax.set_ylabel("validation loss (nats/token)")
    ax.set_title(f"Validation curves, U = {fmt_tokens(u)}"); ax.legend(fontsize=8); save(fig, "val_curves_smallU")

    # Fig: throughput (why bigger models see fewer tokens)
    t = df.groupby("model").agg(n=("n_params_non_emb", "first"), tps=("tokens_per_sec", "mean")).sort_values("n")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(t.n, t.tps, marker="o"); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("non-embedding parameters"); ax.set_ylabel("training tokens / s")
    ax.set_title(f"Throughput on {df.hardware.iloc[0]}"); save(fig, "throughput")
    print(f"figures written to {a.fig_dir}")

if __name__ == "__main__":
    main()