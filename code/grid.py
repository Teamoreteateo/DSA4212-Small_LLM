"""Expand a grid config into a list of runs. Shared by run_grid.py (local) and modal_app.py.
Per-model "lr" (fixed) or "lrs" (sweep) override the config-level values."""
import itertools

def build_cells(cfg, data_root=None, drop_device=False):
    cells = []
    for m in cfg["models"]:
        lrs = m.get("lrs", cfg.get("lrs", [None]))
        for u, s, lr in itertools.product(cfg["unique_tokens"], cfg["seeds"], lrs):
            common = dict(cfg.get("common", {}))
            if lr is not None:
                common["lr"] = lr
            elif "lr" in m:
                common["lr"] = m["lr"]
            if drop_device:
                common.pop("device", None)
            if data_root is not None and "data_dir" in common:
                common["data_dir"] = f"{data_root}/{common['data_dir']}"
            args = ["--n_layer", str(m["n_layer"]), "--n_head", str(m["n_head"]), "--n_embd", str(m["n_embd"]),
                    "--unique_tokens", str(u), "--seed", str(s)]
            for k, v in common.items():
                args += [f"--{k}", str(v)]
            tag = f"_lr{lr:g}" if lr is not None else ""
            cells.append({"args": args, "out": f"{cfg['out_dir']}/{m['name']}_U{u}_s{s}{tag}.json"})
    return cells