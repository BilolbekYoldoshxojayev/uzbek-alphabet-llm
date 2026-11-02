"""Build a parallel old-spelling / new-alphabet training corpus from FineWeb-2 uzn_Latn.

    python -X utf8 scripts/build_train_corpus.py --chars 400000000
Writes data/train/{old,new}/{train,valid}.txt and data/train/build.log.
- Documents are shuffled with seed 42 and taken until --chars characters are collected.
- Any document containing a FLORES-200 dev/devtest Uzbek sentence (Belebele passages are FLORES
  sentences) is dropped, so the test sets never appear in training data.
- "old" = canonical 1995 Latin (oʻ with U+02BB, tutuq U+02BC); "new" = the same text through the
  YangiAlifbo engine. One paragraph per line; documents separated by an empty line.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import canonicalize, to_new2026  # noqa: E402
from progress import log  # noqa: E402

SEED = 42
FLORES = ROOT / "data/raw/flores200/flores200_dataset"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‘’ʻʼ`´']", "'", s.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=400_000_000)
    ap.add_argument("--valid-frac", type=float, default=0.01)
    args = ap.parse_args()

    test = set()
    for f in ("dev/uzn_Latn.dev", "devtest/uzn_Latn.devtest"):
        test |= {norm(l) for l in (FLORES / f).read_text(encoding="utf-8").splitlines() if len(l) > 30}
    log(f"[corpus] {len(test)} FLORES sentences used for decontamination")

    src = ROOT / "data/raw/fineweb2/data/uzn_Latn/train/000_00000.parquet"
    texts = pq.read_table(src, columns=["text"]).column("text").to_pylist()
    log(f"[corpus] read {len(texts):,} FineWeb-2 documents")
    random.Random(SEED).shuffle(texts)

    docs, total, dropped = [], 0, 0
    for t in texts:
        paras = [p.strip() for p in t.split("\n") if p.strip()]
        sents = {norm(s) for p in paras for s in re.split(r"(?<=[.!?])\s+", p) if len(s) > 30}
        if sents & test:
            dropped += 1
            continue
        docs.append(paras)
        total += sum(len(p) for p in paras)
        if total >= args.chars:
            break
    log(f"[corpus] kept {len(docs):,} documents, {total:,} chars; dropped {dropped} overlapping with FLORES")

    n_valid = max(1, int(len(docs) * args.valid_frac))
    splits = {"valid": docs[:n_valid], "train": docs[n_valid:]}
    meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "HuggingFaceFW/fineweb-2 uzn_Latn train @ af9c133",
            "seed": SEED, "target_chars": args.chars, "docs": len(docs), "chars": total, "dropped_flores_overlap": dropped}
    for split, ds in splits.items():
        old = [canonicalize(p) for d in ds for p in d + [""]]
        t0 = time.time()
        nonempty = [l for l in old if l]
        new_nonempty, engine = to_new2026(nonempty)
        it = iter(new_nonempty)
        new = [next(it) if l else "" for l in old]
        for name, lines in (("old", old), ("new", new)):
            d = ROOT / "data/train" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        meta[split] = {"docs": len(ds), "lines": len(old), "chars_old": sum(map(len, old)),
                       "chars_new": sum(map(len, new)), "engine": engine, "convert_seconds": round(time.time() - t0, 1)}
        log(f"[corpus] {split}: {len(ds):,} docs, old {meta[split]['chars_old']:,} chars, "
            f"new {meta[split]['chars_new']:,} chars ({engine})")
    (ROOT / "data/train/build.log").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
