"""Build the 6-condition version of Belebele uzn_Latn.

    python -X utf8 scripts/build_belebele.py data/raw/belebele/uzn_Latn.jsonl belebele_uzn
Writes data/parallel/<name>/<condition>.jsonl (same rows and answer keys, text fields converted),
words_new2026.tsv, handcheck_sample.tsv (30 questions), build.log
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conditions import CONDITIONS, build_all  # noqa: E402

SEED = 42
FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
src, name = Path(sys.argv[1]), sys.argv[2]
out = Path(__file__).resolve().parents[1] / "data/parallel" / name
out.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
# Convert each distinct string once (passages repeat across questions); no field has a newline.
strings = list(dict.fromkeys(r[f] for r in rows for f in FIELDS))
assert not any("\n" in s for s in strings)
conds, engine_log = build_all(strings, out / "words_new2026.tsv")
lookup = {c: dict(zip(strings, conds[c])) for c in CONDITIONS}

for c in CONDITIONS:
    with open(out / f"{c}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, **{k: lookup[c][r[k]] for k in FIELDS}, "condition": c},
                               ensure_ascii=False) + "\n")

rng = random.Random(SEED)
idx = sorted(rng.sample(range(len(rows)), 30))
sample = out / "handcheck_sample.tsv"
if sample.exists():
    sample = out / "handcheck_sample.new.tsv"
with open(sample, "w", encoding="utf-8") as f:
    f.write("row\tfield\tlatin_okina\tnew2026\tcyrillic\tnew2026_errors\tcyrillic_errors\tnotes\n")
    for i in idx:
        for k in ["question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]:
            s = rows[i][k]
            f.write(f"{i}\t{k}\t{lookup['latin_okina'][s]}\t{lookup['new2026'][s]}\t{lookup['cyrillic'][s]}\t\t\t\n")

log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src), "rows": len(rows),
       "distinct_strings": len(strings), "seed": SEED, "engine": engine_log, "handcheck_rows": idx}
(out / "build.log").write_text(json.dumps(log, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in log.items() if k != "handcheck_rows"}))
