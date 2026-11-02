"""Experiment B: adapt an existing model to Uzbek with LoRA, on old-spelling vs new-alphabet text.

    python -X utf8 scripts/train_lora.py --spelling old
    python -X utf8 scripts/train_lora.py --spelling new
Both runs read the SAME documents (the first --chars characters of data/train/<spelling>/train.txt;
old and new are line-aligned), with identical settings and seed. Only the spelling differs.
Writes results/train/lora/<tag>/: adapter/, metrics.json (time, tokens, GPU memory, energy,
validation bits per character), loss.csv.
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



def read_docs(path: Path, max_chars: int | None):
    docs, cur, total = [], [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            cur.append(line)
            continue
        if cur:
            docs.append("\n".join(cur))
            total += len(docs[-1])
            cur = []
            if max_chars and total >= max_chars:
                break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-4b-it")
    ap.add_argument("--spelling", choices=["old", "new"], required=True)
    ap.add_argument("--chars", type=int, default=30_000_000, help="training text, same documents for both spellings")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outroot", default="results/train/lora")
    args = ap.parse_args()

    import peft
    import transformers
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    tag = f"{args.model.replace('/', '__')}__4bit__lora_{args.spelling}_s{args.seed}"
    out = ROOT / args.outroot / tag
    out.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(args.model).sha
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)

    # The old file decides which documents are used, so both spellings get exactly the same ones.
    n_docs = len(read_docs(ROOT / "data/train/old/train.txt", args.chars))
    docs = read_docs(ROOT / f"data/train/{args.spelling}/train.txt", None)[:n_docs]
    vdocs = read_docs(ROOT / f"data/train/{args.spelling}/valid.txt", 200_000)
    chars = sum(map(len, docs))

    def pack(ds):
        ids = []
        for d in ds:
            ids += tok(d, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        n = len(ids) // args.seq
        return torch.tensor(ids[: n * args.seq]).view(n, args.seq), len(ids)

    train, n_tokens = pack(docs)
    valid, n_vtokens = pack(vdocs)
    vchars = sum(map(len, vdocs))
    log(f"[train-B] {tag}: {len(docs):,} docs, {chars:,} chars -> {n_tokens:,} tokens ({n_tokens / chars:.3f} tok/char), "
        f"{len(train)} sequences of {args.seq}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    # Gemma 3 requires token_type_ids in training mode (0 = text, no image tokens here).
    extra = (lambda x: {"token_type_ids": torch.zeros_like(x)}) if "gemma3" in model.config.model_type else (lambda x: {})
    proj = "(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    # Gemma 3 also has a vision tower with q_proj etc.; adapt only the language model.
    has_lm = any("language_model" in n for n, _ in model.named_modules())
    targets = rf".*language_model.*\.{proj}" if has_lm else rf".*\.{proj}"
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=targets))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = len(train) // (args.batch * args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train), generator=g)

    # Words are the same in both spellings (conversion never changes spaces), so bits per WORD is
    # the fair old-vs-new comparison; bits per own character is reported too.
    vwords = sum(len(d.split()) for d in vdocs)

    @torch.no_grad()
    def bits_per_unit():
        model.eval()
        nll = 0.0
        for i in range(0, len(valid), args.batch):
            x = valid[i:i + args.batch].cuda()
            loss = model(input_ids=x, labels=x, **extra(x)).loss  # mean over the (seq-1) predicted tokens per row
            nll += loss.item() * (x.numel() - x.shape[0])
        model.train()
        frac = valid.numel() / n_vtokens  # share of the validation text that was packed and scored
        bits = nll / math.log(2)
        return {"bits_per_word": bits / (frac * vwords), "bits_per_char": bits / (frac * vchars)}

    torch.cuda.reset_peak_memory_stats()
    bpc0 = bits_per_unit()
    log(f"[train-B] {tag}: before training, valid bits/word = {bpc0['bits_per_word']:.3f}; {steps} optimizer steps")
    rows = []
    model.train()
    with PowerMeter() as pm:
        for step in range(steps):
            for k in range(args.accum):
                idx = order[(step * args.accum + k) * args.batch:(step * args.accum + k + 1) * args.batch]
                x = train[idx].cuda()
                loss = model(input_ids=x, labels=x, **extra(x)).loss / args.accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            rows.append({"step": step + 1, "loss": loss.item() * args.accum, "seconds": round(time.time() - pm.t0, 1)})
            if (step + 1) % 25 == 0 or step + 1 == steps:
                log(f"[train-B] {tag}: step {step + 1}/{steps} loss {rows[-1]['loss']:.4f} "
                    f"({rows[-1]['seconds'] / 60:.1f} min)")
    bpc1 = bits_per_unit()
    model.save_pretrained(out / "adapter")
    with open(out / "loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    m = {"model": args.model, "revision": revision, "spelling": args.spelling, "seed": args.seed, "args": vars(args),
         "train_docs": len(docs), "train_chars": chars, "train_tokens": n_tokens, "tokens_per_char": n_tokens / chars,
         "steps": steps, "train_seconds": round(pm.seconds, 1), "tokens_per_second": round(len(train) * args.seq / pm.seconds, 1),
         "chars_per_second": round(chars / pm.seconds, 1), "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
         "gpu_mean_power_w": round(pm.mean_w, 1), "gpu_energy_wh": round(pm.wh, 1), "power_samples": len(pm.samples),
         "valid_before": bpc0, "valid_after": bpc1, "valid_words": vwords, "valid_chars": vchars,
         "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
         "gpu": torch.cuda.get_device_name(0)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    log(f"[train-B] {tag}: DONE in {pm.seconds / 60:.1f} min, {pm.wh:.0f} Wh, valid bits/word {bpc0['bits_per_word']:.3f} -> {bpc1['bits_per_word']:.3f}")


if __name__ == "__main__":
    main()
