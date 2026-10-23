"""Prepare TinyStories: train a byte-level BPE tokenizer, write token arrays.
Outputs in --out_dir: train/val/test.bin (uint16), token_bytes.npy, tokenizer/, meta.json.
Tokenizer training is excluded from the compute budget (allowed by the spec)."""
import argparse, json, os, random
import numpy as np
from tokenizers import ByteLevelBPETokenizer

EOT = "<|endoftext|>"

def load_docs(args):
    if args.source == "local":  # smoke test / offline
        text = open(args.local_path, encoding="utf-8").read()
        docs = [d.strip() for d in text.split("\n\n") if d.strip()]
        random.Random(0).shuffle(docs)
        n_held = max(20, len(docs) // 10)
        return docs[n_held:], docs[:n_held]
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    train = ds["train"].shuffle(seed=0)
    train_docs = train.select(range(min(len(train), args.max_train_docs)))["text"]
    return train_docs, ds["validation"]["text"]  # official val split, never trained on

def encode(tok, docs, eot_id, max_tokens=None, batch=2000):
    ids, total = [], 0
    for i in range(0, len(docs), batch):
        for enc in tok.encode_batch(docs[i:i + batch]):
            ids.extend(enc.ids); ids.append(eot_id); total += len(enc.ids) + 1
        if max_tokens is not None and total >= max_tokens:
            break
    arr = np.array(ids, dtype=np.uint16)
    return arr[:max_tokens] if max_tokens is not None else arr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["tinystories", "local"], default="tinystories")
    p.add_argument("--local_path", default=None)
    p.add_argument("--out_dir", default="data")
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--tokenizer_docs", type=int, default=200_000)
    p.add_argument("--max_train_docs", type=int, default=1_000_000)
    p.add_argument("--max_train_tokens", type=int, default=100_000_000)
    args = p.parse_args()
    assert args.vocab_size <= 65535, "uint16 storage"
    os.makedirs(args.out_dir, exist_ok=True)
    train_docs, heldout_docs = load_docs(args)
    print(f"train docs: {len(train_docs):,}  held-out docs: {len(heldout_docs):,}")

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(train_docs[:args.tokenizer_docs], vocab_size=args.vocab_size,
                            min_frequency=2, special_tokens=[EOT])
    tok_dir = os.path.join(args.out_dir, "tokenizer"); os.makedirs(tok_dir, exist_ok=True)
    tok.save_model(tok_dir)
    eot_id = tok.token_to_id(EOT)
    vocab = tok.get_vocab(); V = max(vocab.values()) + 1

    # byte-level BPE: each char of a token string = 1 byte. EOT counted as 1 byte.
    token_bytes = np.zeros(V, dtype=np.int64)
    for s, i in vocab.items():
        token_bytes[i] = 1 if s == EOT else len(s)
    np.save(os.path.join(args.out_dir, "token_bytes.npy"), token_bytes)

    # held-out split: first half = val (tuning), second half = test (final numbers only)
    train_ids = encode(tok, train_docs, eot_id, args.max_train_tokens)
    half = len(heldout_docs) // 2
    val_ids = encode(tok, heldout_docs[:half], eot_id)
    test_ids = encode(tok, heldout_docs[half:], eot_id)
    for name, arr in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        arr.tofile(os.path.join(args.out_dir, f"{name}.bin")); print(f"{name}: {len(arr):,} tokens")

    meta = dict(vars(args), vocab_size_actual=int(V), eot_id=int(eot_id),
                n_train=int(len(train_ids)), n_val=int(len(val_ids)), n_test=int(len(test_ids)),
                bytes_per_token_train=float(token_bytes[train_ids].mean()))
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()