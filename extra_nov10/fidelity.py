"""Quick check (10 Nov 2026, not in the paper): if you ask Gemma 3 4B a question written in the new alphabet,
does it answer in the new alphabet? (The "script fidelity" idea of Kletz et al., applied inside the Latin script.)

Prompt = passage + question with spelling-neutral labels (Matn / Savol / Javob), chat template, greedy, 48 new tokens.

    python -X utf8 extra_nov10/fidelity.py
Writes extra_nov10/fidelity_<condition>.jsonl and fidelity.md
"""
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MODEL = "unsloth/gemma-3-4b-it"
N = 150
NEW = re.compile(r"[şçöğŞÇÖĞ]")
OLD = re.compile(r"sh|ch|[og]['ʻʼ`’‘]", re.I)


def kind(text):
    n, o = len(NEW.findall(text)), len(OLD.findall(text))
    return "new only" if n and not o else "old only" if o and not n else "mixed" if n and o else "neither"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    lines = ["| question written in | answers: new letters only | old spelling only | mixed | neither |", "|---|---|---|---|---|"]
    for c in ["latin_okina", "new2026"]:
        rows = [json.loads(l) for l in (ROOT / "data/parallel/belebele_uzn" / f"{c}.jsonl").read_text(encoding="utf-8").splitlines()][:N]
        counts = {"new only": 0, "old only": 0, "mixed": 0, "neither": 0}
        done = OUT / f"fidelity_{c}.jsonl"
        if done.exists() and len(done.read_text(encoding="utf-8").splitlines()) == N:   # finished in an earlier run
            for l in done.read_text(encoding="utf-8").splitlines():
                counts[kind(json.loads(l)["answer"])] += 1
            lines.append(f"| {c} | " + " | ".join(f"{counts[k] / N:.0%}" for k in counts) + " |")
            print(lines[-1], "(reused)", flush=True)
            continue
        with open(OUT / f"fidelity_{c}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                msg = [{"role": "user", "content": f"Matn: {r['flores_passage']}\n\nSavol: {r['question']}\n\nJavob:"}]
                enc = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=48, do_sample=False)
                text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                k = kind(text)
                counts[k] += 1
                f.write(json.dumps({"answer": text, "kind": k}, ensure_ascii=False) + "\n")
        lines.append(f"| {c} | " + " | ".join(f"{counts[k] / N:.0%}" for k in counts) + " |")
        print(lines[-1], flush=True)
    (OUT / "fidelity.md").write_text("\n".join(lines) + f"\n\n{N} Belebele questions per spelling, Gemma 3 4B bf16, greedy, 48 new tokens.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
