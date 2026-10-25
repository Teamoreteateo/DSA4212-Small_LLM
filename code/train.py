"""Train one GPT under a wall-clock budget on the first `unique_tokens` training tokens.
Budget = training-step time only (fwd+bwd+optimiser), device-synchronised; eval excluded.
LR schedule is defined on the fraction of budget used -> every run completes warmup + cosine.
Repetition is exact: non-overlapping windows over the subset, reshuffled every epoch."""
import argparse, json, math, os, time
from contextlib import nullcontext
import numpy as np, torch
from model import GPT

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256); p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--data_dir", default="data"); p.add_argument("--unique_tokens", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1); p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.1); p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--budget_sec", type=float, default=600); p.add_argument("--eval_every_sec", type=float, default=60)
    p.add_argument("--eval_batches", type=int, default=40); p.add_argument("--test_tokens", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    return p.parse_args()

def lr_at(frac, a):
    if frac < a.warmup_frac:
        return a.lr * frac / a.warmup_frac
    prog = min((frac - a.warmup_frac) / (1 - a.warmup_frac), 1.0)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

class EpochLoader:
    """Non-overlapping windows over the restricted subset, reshuffled every epoch."""
    def __init__(self, data, T, B, rng):
        self.data, self.T, self.B, self.rng = data, T, B, rng
        self.starts = np.arange(0, len(data) - T - 1, T)
        assert len(self.starts) >= B, "unique_tokens too small for one batch"
        self.order, self.pos = rng.permutation(self.starts), 0
    def next(self):
        if self.pos + self.B > len(self.order):
            self.order, self.pos = self.rng.permutation(self.starts), 0
        s = self.order[self.pos:self.pos + self.B]; self.pos += self.B
        x = np.stack([self.data[i:i + self.T] for i in s]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.T] for i in s]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

@torch.no_grad()
def eval_loss(model, data, T, B, n_batches, device, ctx, token_bytes=None, rng=None):
    """Mean nats/token (+ bits per byte if token_bytes given). rng=None -> deterministic sweep."""
    model.eval()
    starts = np.arange(0, len(data) - T - 1, T)
    starts = (rng.choice(starts, size=min(len(starts), n_batches * B), replace=False)
              if rng is not None else starts[:n_batches * B])
    nats, ntok, nbytes = 0.0, 0, 0
    for i in range(0, len(starts), B):
        s = starts[i:i + B]
        x = torch.from_numpy(np.stack([data[j:j + T] for j in s]).astype(np.int64)).to(device)
        y = np.stack([data[j + 1:j + 1 + T] for j in s]).astype(np.int64)
        with ctx:
            _, loss = model(x, torch.from_numpy(y).to(device), reduction="sum")
        nats += loss.item(); ntok += y.size
        if token_bytes is not None:
            nbytes += int(token_bytes[y].sum())
    model.train()
    return nats / ntok, (nats / math.log(2) / nbytes if token_bytes is not None else None)

def pick_device(choice):
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    a = get_args()
    device = pick_device(a.device)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    token_bytes = np.load(os.path.join(a.data_dir, "token_bytes.npy"))
    load = lambda n: np.memmap(os.path.join(a.data_dir, f"{n}.bin"), dtype=np.uint16, mode="r")
    train_all, val, test = load("train"), load("val"), load("test")
    assert a.unique_tokens <= len(train_all), f"only {len(train_all)} train tokens available"
    train = np.array(train_all[:a.unique_tokens])  # the restricted unique subset

    model = GPT(meta["vocab_size_actual"], a.block_size, a.n_layer, a.n_head, a.n_embd, a.dropout).to(device)
    opt = model.configure_optimizer(a.lr, a.weight_decay)
    use_amp = device == "cuda"  # fp16 autocast only on CUDA (T4 has no bf16)
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sync = {"cuda": torch.cuda.synchronize,
            "mps": getattr(torch.mps, "synchronize", lambda: None)}.get(device, lambda: None)
    loader = EpochLoader(train, a.block_size, a.batch_size, rng)

    log, train_time, eval_time, steps, tokens = [], 0.0, 0.0, 0, 0
    next_eval = a.eval_every_sec
    while train_time < a.budget_sec:
        for g in opt.param_groups:
            g["lr"] = lr_at(train_time / a.budget_sec, a)
        t0 = time.perf_counter()
        x, y = loader.next(); x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(opt); scaler.update(); sync()
        train_time += time.perf_counter() - t0; steps += 1; tokens += x.numel()

        if train_time >= next_eval or train_time >= a.budget_sec:
            t1 = time.perf_counter()
            v, _ = eval_loss(model, val, a.block_size, a.batch_size, a.eval_batches, device, ctx,
                             rng=np.random.default_rng(1234))  # same windows for every run
            tr, _ = eval_loss(model, train, a.block_size, a.batch_size, a.eval_batches, device, ctx,
                              rng=np.random.default_rng(1234))
            eval_time += time.perf_counter() - t1
            log.append(dict(train_time=train_time, steps=steps, tokens=tokens, epochs=tokens / a.unique_tokens,
                            lr=opt.param_groups[0]["lr"], batch_loss=loss.item(), train_loss=tr, val_loss=v))
            print(f"[{train_time:5.0f}s] step {steps} ep {tokens / a.unique_tokens:.2f} "
                  f"train {tr:.4f} val {v:.4f}", flush=True)
            next_eval += a.eval_every_sec

    n_test = max(1, a.test_tokens // (a.block_size * a.batch_size))
    test_nats, test_bpb = eval_loss(model, test, a.block_size, a.batch_size, n_test, device, ctx, token_bytes)
    n_ne = model.n_params(True)
    res = dict(config=vars(a), device=device,
               hardware=torch.cuda.get_device_name(0) if device == "cuda" else device,
               n_params=model.n_params(False), n_params_non_emb=n_ne, steps=steps, tokens_seen=tokens,
               epochs=tokens / a.unique_tokens, train_time=train_time, eval_time=eval_time,
               tokens_per_sec=tokens / train_time, approx_train_flops=6 * n_ne * tokens,
               final_train_loss=log[-1]["train_loss"], final_val_loss=log[-1]["val_loss"],
               best_val_loss=min(r["val_loss"] for r in log), test_loss=test_nats, test_bpb=test_bpb,
               torch_version=torch.__version__, log=log)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"done: test_bpb={test_bpb:.4f} epochs={res['epochs']:.2f} params_ne={n_ne:,}")

if __name__ == "__main__":
    main()