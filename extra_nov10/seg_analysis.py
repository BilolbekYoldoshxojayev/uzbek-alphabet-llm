"""Quick check (10 Nov 2026, not in the paper): how the new alphabet changes WHERE tokenizers split words,
and what Unicode form (NFC vs NFD) does to token counts.

    python -X utf8 extra_nov10/seg_analysis.py
Writes extra_nov10/seg_analysis.md and seg_examples.md

Words are tokenized one at a time with a leading space (no sentence context). Old = latin_ascii, new = new2026.
"""
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/parallel/flores_devtest"
OUT = Path(__file__).resolve().parent

TOKENIZERS = {
    "GPT-4o (o200k)": ("tiktoken", "o200k_base"),
    "GPT-4 (cl100k)": ("tiktoken", "cl100k_base"),
    "Llama 3.x": ("hf", "unsloth/Llama-3.2-1B-Instruct"),
    "Qwen 2.5": ("hf", "Qwen/Qwen2.5-7B-Instruct"),
    "Gemma 3": ("hf", "unsloth/gemma-3-4b-it"),
    "Mistral Nemo": ("hf", "unsloth/Mistral-Nemo-Instruct-2407"),
    "DeepSeek V3": ("hf", "deepseek-ai/DeepSeek-V3"),
}
DIGRAPH = {"sh": "ş", "ch": "ç", "o'": "ö", "g'": "ğ"}


def load(kind, src):
    if kind == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding(src)
        return enc.encode, (lambda ids: enc.decode(ids, errors="replace"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src)
    return (lambda s: tok.encode(s, add_special_tokens=False)), (lambda ids: tok.decode(ids))


def boundaries(word, encode, decode):
    """Internal token boundaries of a word as character positions; x.5 = boundary inside a character (bytes).
    Returns (n_tokens, list of positions) or None if the tokenizer does not round-trip the word."""
    ids = encode(" " + word)
    if decode(ids).strip() != word:
        return None
    pos = []
    for k in range(1, len(ids)):
        pre = decode(ids[:k]).lstrip()
        if pre.endswith("�"):
            pos.append(len(pre.rstrip("�")) + 0.5)
        else:
            pos.append(len(pre))
    return len(ids), [p for p in pos if p > 0]


def align(old, new):
    """For each new-letter position j, the old span it came from. Returns (old_pos -> new_pos map, list of
    (new index, old start) for converted letters) or None if the words do not line up."""
    m, conv, i, j = {0: 0}, [], 0, 0
    while i < len(old) and j < len(new):
        two = old[i:i + 2].lower()
        if two in DIGRAPH and new[j].lower() == DIGRAPH[two]:
            conv.append((j, i))
            i, j = i + 2, j + 1
        elif old[i] == new[j] or (old[i] == "'" and new[j] in "ʼʻ"):
            i, j = i + 1, j + 1
        else:
            return None
        m[i] = j
    return (m, conv) if i == len(old) and j == len(new) else None


def main():
    old_lines = (DATA / "latin_ascii.txt").read_text(encoding="utf-8").splitlines()
    new_lines = (DATA / "new2026.txt").read_text(encoding="utf-8").splitlines()
    pairs = Counter()
    for o, n in zip(old_lines, new_lines):
        ow, nw = o.split(), n.split()
        assert len(ow) == len(nw)
        pairs.update((a.strip('.,;:!?"()«»“”'), b.strip('.,;:!?"()«»“”')) for a, b in zip(ow, nw) if a != b)
    pairs = {p: c for p, c in pairs.items() if p[0] and p[1] and p[0] != p[1]}
    total_new = "\n".join(new_lines)
    nfd = unicodedata.normalize("NFD", total_new)
    assert unicodedata.normalize("NFC", total_new) == total_new

    rows, examples = [], []
    for name, (kind, src) in TOKENIZERS.items():
        try:
            encode, decode = load(kind, src)
        except Exception as e:
            print(f"{name}: SKIPPED {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
            continue
        n_nfc = sum(len(encode(l)) for l in new_lines)
        n_nfd = sum(len(encode(l)) for l in nfd.split("\n"))
        st = Counter()
        for (o, n), c in pairs.items():
            al = align(o, n)
            bo, bn = boundaries(o, encode, decode), boundaries(n, encode, decode)
            if al is None or bo is None or bn is None:
                st["skipped"] += c
                continue
            m, conv = al
            (to, po), (tn, pn) = bo, bn
            st["words"] += c
            st["old_tokens"] += c * to
            st["new_tokens"] += c * tn
            st["fewer" if tn < to else "more" if tn > to else "same_count"] += c
            mapped = {m.get(p, "in-digraph") for p in po}          # old boundaries in new coordinates
            same = mapped == set(pn)
            st["same_split"] += c * same
            st["same_count_diff_split"] += c * (tn == to and not same)
            for j, i in conv:                                       # each converted letter
                st["letters"] += c
                st["digraph_split"] += c * ((i + 1) in po)          # old: boundary between s and h
                left, right = max([0] + [p for p in pn if p <= j]), min([len(n)] + [p for p in pn if p >= j + 1])
                inbytes = any(j < p < j + 1 for p in pn)
                st["new_bytes"] += c * inbytes
                st["new_alone"] += c * (not inbytes and left == j and right == j + 1)
            if name in ("GPT-4o (o200k)", "Gemma 3") and c >= 8 and tn == to and not same and len(examples) < 40:
                def show(w, ps):
                    cuts = [0] + sorted(int(p) for p in ps) + [len(w)]
                    return " | ".join(w[a:b] for a, b in zip(cuts, cuts[1:]))
                examples.append(f"| {name} | {show(o, po)} | {show(n, pn)} | {c} |")
        w, l = st["words"], st["letters"]
        rows.append(f"| {name} | {(n_nfd / n_nfc - 1) * 100:+.1f}% | {w} | {(st['new_tokens'] / st['old_tokens'] - 1) * 100:+.1f}% "
                    f"| {st['fewer'] / w:.0%} / {st['same_count'] / w:.0%} / {st['more'] / w:.0%} | {1 - st['same_split'] / w:.0%} "
                    f"| {st['same_count_diff_split'] / max(st['same_count'], 1):.0%} | {st['digraph_split'] / l:.0%} "
                    f"| {st['new_alone'] / l:.0%} | {st['new_bytes'] / l:.0%} | {st['skipped']} |")
        print(rows[-1], flush=True)

    head = ("| tokenizer | tokens, NFD vs NFC | changed word tokens | tokens in changed words, new vs old | fewer / same / more tokens "
            "| split differently | of same-count words, split differently | old digraph cut in two | new letter is its own token "
            "| new letter cut into bytes | skipped |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    (OUT / "seg_analysis.md").write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    (OUT / "seg_examples.md").write_text("| tokenizer | old split | new split | count |\n|---|---|---|---|\n"
                                         + "\n".join(examples) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
