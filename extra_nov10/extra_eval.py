"""Quick check (10 Nov 2026, not in the paper): Belebele accuracy of Gemma 3 4B (bf16) when the new-alphabet text is
(a) decomposed (NFD: o + combining mark) or (b) stripped of its marks (ö->o, ğ->g, ş->s, ç->c).
Same prompt and scoring as scripts/rq2_eval.py. latin_okina and new2026 are re-run to check the setup reproduces.

    python -X utf8 extra_nov10/extra_eval.py
Writes extra_nov10/eval_<condition>.jsonl and extra_eval.md
"""
import json
import sys
import unicodedata
from math import comb
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from rq2_eval import LETTERS, TEMPLATE, letter_ids  # noqa: E402,F401

MODEL = "unsloth/gemma-3-4b-it"
FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]


def strip_marks(s):
    return unicodedata.normalize("NFC", "".join(ch for ch in unicodedata.normalize("NFD", s)
                                                if unicodedata.category(ch) != "Mn"))


CONDS = {"latin_okina": ("latin_okina", lambda s: s),
         "new2026": ("new2026", lambda s: s),
         "new2026_nfd": ("new2026", lambda s: unicodedata.normalize("NFD", s)),
         "new2026_stripped": ("new2026", strip_marks)}


def mcnemar(a, b):
    """Exact two-sided McNemar p-value for two lists of 0/1 correctness."""
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n, k = n01 + n10, min(n01, n10)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = letter_ids(tok, " ")
    correct, tokens = {}, {}
    for name, (src, fn) in CONDS.items():
        rows = [json.loads(l) for l in (ROOT / "data/parallel/belebele_uzn" / f"{src}.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        correct[name], tokens[name] = [], 0
        with open(OUT / f"eval_{name}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                r = {**r, **{k: fn(r[k]) for k in FIELDS}}
                enc = tok(TEMPLATE.format(**r), return_tensors="pt").to("cuda")
                with torch.no_grad():
                    lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)[ids].tolist()
                pred = max(range(4), key=lambda k: lp[k]) + 1
                ok = int(pred == int(r["correct_answer_num"]))
                correct[name].append(ok)
                tokens[name] += enc["input_ids"].shape[1]
                f.write(json.dumps({"i": i, "pred": pred, "ok": ok, "prompt_tokens": enc["input_ids"].shape[1]}) + "\n")
        print(f"{name}: acc={sum(correct[name]) / len(rows):.4f} tokens={tokens[name]}", flush=True)

    n = len(correct["latin_okina"])
    lines = ["| condition | accuracy % | vs latin_okina, points (p) | vs new2026, points (p) | prompt tokens vs new2026 |", "|---|---|---|---|---|"]
    for name in CONDS:
        acc = 100 * sum(correct[name]) / n
        d1 = acc - 100 * sum(correct["latin_okina"]) / n
        d2 = acc - 100 * sum(correct["new2026"]) / n
        lines.append(f"| {name} | {acc:.1f} | {d1:+.1f} (p={mcnemar(correct[name], correct['latin_okina']):.3g}) "
                     f"| {d2:+.1f} (p={mcnemar(correct[name], correct['new2026']):.3g}) | {(tokens[name] / tokens['new2026'] - 1) * 100:+.1f}% |")
    (OUT / "extra_eval.md").write_text("\n".join(lines) + f"\n\nn = {n} questions, Gemma 3 4B bf16, raw English prompt, exact McNemar, no correction for multiple tests.\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
