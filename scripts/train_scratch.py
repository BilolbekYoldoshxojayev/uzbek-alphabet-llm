"""Experiment A: train the same small GPT-2-style model from scratch on old-spelling vs new-alphabet Uzbek.

    python -X utf8 scripts/train_scratch.py --spelling old
    python -X utf8 scripts/train_scratch.py --spelling new
Same documents, same architecture, same hyperparameters, same seed; only the spelling differs.
Each spelling gets its own byte-level BPE tokenizer with the same vocabulary size, trained on its
own text. Measured: tokens needed, training time, GPU energy, peak memory, and validation
bits per WORD (identical word counts in both spellings, so directly comparable), logged over time.
Run twice per --tokenizer mode (separate: own tokenizer per spelling; shared: one tokenizer
trained on old+new together). Writes results/train/scratch/<mode>_<spelling>/: tokenizer.json, metrics.json, curve.csv, model/.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from power import PowerMeter  # noqa: E402
from progress import log  # noqa: E402



def docs_of(path: Path):
    docs, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            cur.append(line)
        elif cur:
            docs.append("\n".join(cur))
            cur = []
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spelling", choices=["old", "new"], required=True)
    ap.add_argument("--tokenizer", choices=["separate", "shared"], default="separate",
                    help="separate: trained on this spelling's text only; shared: one tokenizer trained on old+new "
                         "text together, used by both models")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chars", type=int, default=None, help="use only the first documents, up to this many old-spelling chars")
    ap.add_argument("--outroot", default="results/train/scratch")
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps between validation checks")
    args = ap.parse_args()

    import tokenizers
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(args.seed)
    run = f"{args.tokenizer}_{args.spelling}_s{args.seed}"
    root = ROOT / args.outroot
    out = root / run
    out.mkdir(parents=True, exist_ok=True)
    d = ROOT / "data/train" / args.spelling
    train_docs, valid_docs = docs_of(d / "train.txt"), docs_of(d / "valid.txt")
    if args.chars:  # the old-spelling text decides which documents, so both spellings get the same ones
        total = n_docs = 0
        for x in docs_of(ROOT / "data/train/old/train.txt"):
            total += len(x)
            n_docs += 1
            if total >= args.chars:
                break
        train_docs = train_docs[:n_docs]

    # 1. Tokenizer: byte-level BPE with the same vocab size in every run.
    def new_tokenizer(texts):
        t = Tokenizer(models.BPE())
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder = decoders.ByteLevel()
        t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=["<|endoftext|>"],
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                          show_progress=False))
        return t

    # The same tokenizer is reused across seeds; its training time is measured once and recorded.
    name = "shared" if args.tokenizer == "shared" else f"separate_{args.spelling}"
    cached = root / f"tokenizer_{name}.json"
    if cached.exists():
        tok = Tokenizer.from_file(str(cached))
        tok_seconds = float((root / f"tokenizer_{name}.seconds").read_text())
    else:
        t0 = time.time()
        if args.tokenizer == "separate":
            tok = new_tokenizer(train_docs)
        else:
            other = "new" if args.spelling == "old" else "old"
            tok = new_tokenizer(train_docs + docs_of(ROOT / "data/train" / other / "train.txt")[:len(train_docs)])
        tok_seconds = time.time() - t0
        tok.save(str(cached))
        (root / f"tokenizer_{name}.seconds").write_text(str(tok_seconds))
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")

    def pack(ds):
        ids = []
        for enc in tok.encode_batch(ds):
            ids += enc.ids + [eot]
        n = len(ids) // args.ctx
        return torch.tensor(ids[: n * args.ctx]).view(n, args.ctx), len(ids)

    train, n_tok = pack(train_docs)
    valid, n_vtok = pack(valid_docs)
    chars = sum(map(len, train_docs))
    vwords = sum(len(x.split()) for x in valid_docs)
    log(f"[train-A] {run}: tokenizer {tok_seconds:.0f}s; train {chars:,} chars -> {n_tok:,} tokens "
        f"({n_tok / chars:.3f} tok/char); {len(train):,} sequences")

    # 2. Model
    cfg = GPT2Config(vocab_size=args.vocab, n_positions=args.ctx, n_embd=args.dim, n_layer=args.layers,
                     n_head=args.heads, bos_token_id=eot, eos_token_id=eot)
    model = GPT2LMHeadModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    per_step = args.batch * args.accum
    steps = args.epochs * len(train) // per_step
    warm = max(1, steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    order = torch.cat([torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + e))
                       for e in range(args.epochs)])

    @torch.no_grad()
    def bits_per_word():
        model.eval()
        nll = 0.0
        for i in range(0, len(valid), args.batch):
            x = valid[i:i + args.batch].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
            nll += loss.float().item() * (x.numel() - x.shape[0])
        model.train()
        return nll / math.log(2) / (valid.numel() / n_vtok * vwords)

    torch.cuda.reset_peak_memory_stats()
    curve = []
    eval_seconds = 0.0
    model.train()
    with PowerMeter() as pm:
        for step in range(steps):
            for k in range(args.accum):
                idx = order[(step * args.accum + k) * args.batch:(step * args.accum + k + 1) * args.batch]
                x = train[idx].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x, labels=x).loss / args.accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                t_eval = time.time()
                bpw = bits_per_word()
                eval_seconds += time.time() - t_eval
                curve.append({"step": step + 1, "seconds": round(time.time() - pm.t0 - eval_seconds, 1),
                              "train_docs_fraction": round((step + 1) / steps * args.epochs, 4),
                              "train_loss": round(loss.item() * args.accum, 4), "valid_bits_per_word": round(bpw, 4)})
                log(f"[train-A] {run}: step {step + 1}/{steps} valid bits/word {bpw:.3f} "
                    f"({curve[-1]['seconds'] / 60:.1f} min)")
    model.save_pretrained(out / "model")
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    m = {"spelling": args.spelling, "tokenizer_mode": args.tokenizer, "seed": args.seed, "args": vars(args), "params": n_params,
         "train_docs": len(train_docs), "train_chars": chars, "train_tokens": n_tok, "tokens_per_char": n_tok / chars,
         "valid_words": vwords, "tokenizer_seconds": round(tok_seconds, 1), "steps": steps,
         "train_seconds": round(pm.seconds - eval_seconds, 1), "eval_seconds": round(eval_seconds, 1),
         "tokens_per_second": round(steps * per_step * args.ctx / (pm.seconds - eval_seconds), 1),
         "chars_per_second": round(chars / (pm.seconds - eval_seconds), 1),
         "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
         "gpu_mean_power_w": round(pm.mean_w, 1), "gpu_energy_wh": round(pm.mean_w * (pm.seconds - eval_seconds) / 3600, 1),
         "final_valid_bits_per_word": curve[-1]["valid_bits_per_word"],
         "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
         "gpu": torch.cuda.get_device_name(0)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    log(f"[train-A] {run}: DONE {n_params / 1e6:.0f}M params, {(pm.seconds - eval_seconds) / 60:.1f} min, {m['gpu_energy_wh']:.0f} Wh, "
        f"final valid bits/word {m['final_valid_bits_per_word']:.3f}")


if __name__ == "__main__":
    main()
