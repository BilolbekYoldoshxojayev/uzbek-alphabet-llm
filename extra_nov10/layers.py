"""Quick check (10 Nov 2026, not in the paper): at which layers does Gemma 3 4B treat the same Uzbek sentence
in different spellings as the same sentence?

For each layer, every sentence is the mean of its token hidden states. For a sentence in spelling X we look for
the most similar sentence (cosine, after subtracting each spelling's mean vector) among all sentences in the
official Latin spelling. "Retrieval accuracy" = how often that is the same sentence.

    python -X utf8 extra_nov10/layers.py
Writes extra_nov10/layers.md
"""
import unicodedata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/parallel/flores_devtest"
MODEL = "unsloth/gemma-3-4b-it"
N = 500
CONDS = ["latin_okina", "latin_ascii", "new2026", "new2026_nfd", "cyrillic"]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    reps = {}
    for c in CONDS:
        src = "new2026" if c == "new2026_nfd" else c
        lines = (DATA / f"{src}.txt").read_text(encoding="utf-8").splitlines()[:N]
        if c == "new2026_nfd":
            lines = [unicodedata.normalize("NFD", l) for l in lines]
        out = []
        for l in lines:
            enc = tok(l, return_tensors="pt").to("cuda")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, 1:].float().mean(0) for h in hs]).cpu())   # [layers+1, dim], BOS dropped
        reps[c] = torch.stack(out, 1)                                               # [layers+1, N, dim]
        print(c, tuple(reps[c].shape), flush=True)

    def norm(x):
        x = x - x.mean(0, keepdim=True)
        return x / x.norm(dim=-1, keepdim=True)

    L = reps["latin_okina"].shape[0]
    others = CONDS[1:]
    lines = ["| layer | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
    for layer in range(L):
        ref = norm(reps["latin_okina"][layer])
        cells = []
        for c in others:
            sim = norm(reps[c][layer]) @ ref.T
            cells.append(f"{(sim.argmax(1) == torch.arange(N)).float().mean().item() * 100:.0f}")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    note = (f"\n\nRetrieval accuracy (%) of the matching official-Latin sentence, {N} FLORES devtest sentences, Gemma 3 4B bf16. "
            "Layer 0 = token embeddings. Chance = 0.2%.\n")
    (OUT / "layers.md").write_text("\n".join(lines) + note, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
