"""Add the RQ4 conditions to built datasets and report how exact back-conversion is.

    python -X utf8 scripts/build_rq4.py
backconv: new2026 converted back by plain rules. mixed_apostrophe: every word uses a random mark.
Writes data/parallel/<name>/{backconv,mixed_apostrophe}.* and results/rq4/backconv_exactness.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import back_convert, mixed_apostrophes  # noqa: E402
from progress import log  # noqa: E402

FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
SEED = 42
report = {}
d = ROOT / "data/parallel/belebele_uzn"
old = [json.loads(l) for l in (d / "latin_okina.jsonl").read_text(encoding="utf-8").splitlines()]
new = [json.loads(l) for l in (d / "new2026.jsonl").read_text(encoding="utf-8").splitlines()]
same = tot = 0
diffs = []
with open(d / "backconv.jsonl", "w", encoding="utf-8") as fb, open(d / "mixed_apostrophe.jsonl", "w", encoding="utf-8") as fm:
    for i, (o, n) in enumerate(zip(old, new)):
        b = {**n, **{k: back_convert(n[k]) for k in FIELDS}, "condition": "backconv"}
        m = {**o, **{k: mixed_apostrophes(o[k], SEED + i * 10 + j) for j, k in enumerate(FIELDS)},
             "condition": "mixed_apostrophe"}
        fb.write(json.dumps(b, ensure_ascii=False) + "\n")
        fm.write(json.dumps(m, ensure_ascii=False) + "\n")
        for k in FIELDS:
            tot += 1
            same += b[k] == o[k]
            if b[k] != o[k] and len(diffs) < 30:
                diffs += [(w1, w2) for w1, w2 in zip(o[k].split(), b[k].split()) if w1 != w2][:2]
report["belebele_fields_identical"] = same / tot
report["belebele_example_differences"] = diffs[:30]
for name in ("flores_devtest", "flores_dev"):
    d = ROOT / "data/parallel" / name
    o = (d / "latin_okina.txt").read_text(encoding="utf-8").splitlines()
    n = (d / "new2026.txt").read_text(encoding="utf-8").splitlines()
    b = [back_convert(x) for x in n]
    (d / "backconv.txt").write_text("\n".join(b) + "\n", encoding="utf-8")
    (d / "mixed_apostrophe.txt").write_text(
        "\n".join(mixed_apostrophes(x, SEED + i) for i, x in enumerate(o)) + "\n", encoding="utf-8")
    report[f"{name}_lines_identical"] = sum(x == y for x, y in zip(o, b)) / len(o)
out = ROOT / "results/rq4"
out.mkdir(parents=True, exist_ok=True)
(out / "backconv_exactness.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"[rq4] back-conversion exact on {report['belebele_fields_identical'] * 100:.2f}% of Belebele fields")
print(json.dumps(report, ensure_ascii=False)[:900])
