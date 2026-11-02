"""Add the letter ablations (new_og: only ö ğ; new_shch: only ş ç) to a built dataset.

    python -X utf8 scripts/build_ablations.py belebele_uzn
Derived from latin_okina + new2026 already in data/parallel/<name>/, so the engine is not re-run
and the 6 main condition files are not touched.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import ABLATIONS, partial_new  # noqa: E402
from progress import log  # noqa: E402

FIELDS = ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
d = ROOT / "data/parallel" / sys.argv[1]

if (d / "latin_okina.jsonl").exists():
    old = [json.loads(l) for l in (d / "latin_okina.jsonl").read_text(encoding="utf-8").splitlines()]
    new = [json.loads(l) for l in (d / "new2026.jsonl").read_text(encoding="utf-8").splitlines()]
    for name, keep in ABLATIONS.items():
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for o, n in zip(old, new):
                f.write(json.dumps({**o, **{k: partial_new(o[k], n[k], keep) for k in FIELDS}, "condition": name},
                                   ensure_ascii=False) + "\n")
else:
    old = (d / "latin_okina.txt").read_text(encoding="utf-8").splitlines()
    new = (d / "new2026.txt").read_text(encoding="utf-8").splitlines()
    for name, keep in ABLATIONS.items():
        (d / f"{name}.txt").write_text("\n".join(partial_new(o, n, keep) for o, n in zip(old, new)) + "\n", encoding="utf-8")
log(f"[ablations] {sys.argv[1]}: wrote {', '.join(ABLATIONS)}")
