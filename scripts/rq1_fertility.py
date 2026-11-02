"""RQ1: tokenizer fertility for each spelling condition.

    python -X utf8 scripts/rq1_fertility.py flores_devtest
Writes results/rq1/<name>_fertility.csv, <name>_tokens_per_word.md, results/logs/rq1_<name>_<time>.json

Words are whitespace-separated words of latin_okina; every condition has the same word
boundaries, so tokens/word is comparable across conditions. Ratio CIs are a sentence-level
bootstrap (1000 resamples, seed 42) of total tokens(condition) / total tokens(latin_okina).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import CONDITIONS, EXTRA_CONDITIONS  # noqa: E402
from progress import log as progress  # noqa: E402

SEED, N_BOOT = 42, 1000

# name -> (kind, source). Mirrors are used where the official repo is gated; the tokenizer
# files are checked to be identical in the log (sha256 of tokenizer.json where present).
TOKENIZERS = {
    "GPT-4o (o200k)":   ("tiktoken", "o200k_base"),
    "GPT-4 (cl100k)":   ("tiktoken", "cl100k_base"),
    "Llama 3.x":        ("hf", "unsloth/Llama-3.2-1B-Instruct"),
    "Qwen 2.5":         ("hf", "Qwen/Qwen2.5-7B-Instruct"),
    "Qwen 3":           ("hf", "Qwen/Qwen3-8B"),
    "Gemma 3":          ("hf", "unsloth/gemma-3-4b-it"),
    "Mistral Nemo":     ("hf", "unsloth/Mistral-Nemo-Instruct-2407"),
    "Aya Expanse":      ("hf", "CohereLabs/aya-expanse-8b"),
    "DeepSeek V3":      ("hf", "deepseek-ai/DeepSeek-V3"),
    "Phi-4":            ("hf", "microsoft/phi-4"),
    "UTF-8 bytes":      ("bytes", None),
}


def load(kind, src):
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(src)
        return (lambda s: len(enc.encode(s))), {"tiktoken": tiktoken.__version__}
    if kind == "bytes":
        return (lambda s: len(s.encode("utf-8"))), {}
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    sha = HfApi().model_info(src).sha
    tok = AutoTokenizer.from_pretrained(src, revision=sha)
    meta = {"repo": src, "revision": sha, "class": type(tok).__name__, "vocab": len(tok)}
    try:
        import hashlib
        p = hf_hub_download(src, "tokenizer.json", revision=sha)
        meta["tokenizer_json_sha256"] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        pass
    return (lambda s: len(tok.encode(s, add_special_tokens=False))), meta


def main(name):
    data = ROOT / "data/parallel" / name
    conds = [c for c in CONDITIONS + EXTRA_CONDITIONS if (data / f"{c}.txt").exists()]
    texts = {c: (data / f"{c}.txt").read_text(encoding="utf-8").splitlines() for c in conds}
    words = np.array([len(l.split()) for l in texts["latin_okina"]])
    n = len(words)
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, n, size=(N_BOOT, n))

    rows, log = [], {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": name, "sentences": n,
                     "words": int(words.sum()), "seed": SEED, "n_boot": N_BOOT, "tokenizers": {}, "failed": {}}
    for tname, (kind, src) in TOKENIZERS.items():
        try:
            count, meta = load(kind, src)
        except Exception as e:  # gated repo, network error, ...: record it, never fill in a number
            log["failed"][tname] = f"{type(e).__name__}: {e}"[:300]
            progress(f"[rq1] {name}: {tname} SKIPPED ({log['failed'][tname][:120]})")
            continue
        log["tokenizers"][tname] = meta
        per = {c: np.array([count(l) for l in texts[c]]) for c in conds}
        base = per["latin_okina"]
        for c in conds:
            t = per[c]
            ratios = t[boot].sum(1) / base[boot].sum(1)
            rows.append({"tokenizer": tname, "condition": c, "tokens": int(t.sum()),
                         "tok_per_word": t.sum() / words.sum(),
                         "tok_per_char": t.sum() / sum(len(l) for l in texts[c]),
                         "ratio_vs_okina": t.sum() / base.sum(),
                         "ci_lo": np.percentile(ratios, 2.5), "ci_hi": np.percentile(ratios, 97.5)})
        progress(f"[rq1] {name}: {tname} done")

    out = ROOT / "results/rq1"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}_fertility.csv", index=False)
    wide = df.pivot(index="tokenizer", columns="condition", values="tok_per_word")[conds]
    wide = wide.reindex([t for t in TOKENIZERS if t in wide.index])
    (out / f"{name}_tokens_per_word.md").write_text(wide.round(3).to_markdown(), encoding="utf-8")
    pct = df.assign(pct=lambda d: (d.ratio_vs_okina - 1) * 100,
                    cell=lambda d: d.apply(lambda r: f"{(r.ratio_vs_okina-1)*100:+.1f} [{(r.ci_lo-1)*100:+.1f}, {(r.ci_hi-1)*100:+.1f}]", axis=1))
    pw = pct.pivot(index="tokenizer", columns="condition", values="cell")[conds[1:]].reindex(wide.index)
    (out / f"{name}_pct_vs_okina.md").write_text(pw.to_markdown(), encoding="utf-8")
    logdir = ROOT / "results/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"rq1_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(wide.round(3).to_markdown())
    print(pw.to_markdown())


if __name__ == "__main__":
    main(sys.argv[1])
