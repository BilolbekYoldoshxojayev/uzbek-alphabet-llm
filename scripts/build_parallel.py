"""Build the 6-condition parallel version of a 1995-Latin Uzbek text file.

    python scripts/build_parallel.py data/raw/flores200/flores200_dataset/devtest/uzn_Latn.devtest flores_devtest
Writes data/parallel/<name>/<condition>.txt, words_new2026.tsv, handcheck_sample.tsv, build.log
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conditions import CONDITIONS, build_all  # noqa: E402

SEED = 42
src, name = Path(sys.argv[1]), sys.argv[2]
out = Path(__file__).resolve().parents[1] / "data/parallel" / name
out.mkdir(parents=True, exist_ok=True)

raw = src.read_text(encoding="utf-8").splitlines()
conds, engine_log = build_all(raw, out / "words_new2026.tsv")
for c in CONDITIONS:
    (out / f"{c}.txt").write_text("\n".join(conds[c]) + "\n", encoding="utf-8")

with open(out / "parallel.jsonl", "w", encoding="utf-8") as f:
    for i in range(len(raw)):
        f.write(json.dumps({"id": i, "raw": raw[i], **{c: conds[c][i] for c in CONDITIONS}}, ensure_ascii=False) + "\n")

# 100-sentence hand-check sample: one row per sentence, columns to fill in by hand.
rng = random.Random(SEED)
idx = sorted(rng.sample(range(len(raw)), min(100, len(raw))))
# Never overwrite a sample that may already hold hand annotations.
sample = out / "handcheck_sample.tsv"
if sample.exists():
    sample = out / "handcheck_sample.new.tsv"
with open(sample, "w", encoding="utf-8") as f:
    f.write("id\tlatin_okina\tnew2026\tcyrillic\tnew2026_errors\tcyrillic_errors\tnotes\n")
    for i in idx:
        f.write(f"{i}\t{conds['latin_okina'][i]}\t{conds['new2026'][i]}\t{conds['cyrillic'][i]}\t\t\t\n")

log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src), "lines": len(raw),
       "seed": SEED, "engine": engine_log, "handcheck_ids": idx}
(out / "build.log").write_text(json.dumps(log, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in log.items() if k != "handcheck_ids"}))
