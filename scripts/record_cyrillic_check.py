"""Record the Cyrillic review of the hand-check samples and compute its error rates.

    python -X utf8 scripts/record_cyrillic_check.py
The review was done by Claude (an AI, not a native speaker) from its knowledge of Uzbek Cyrillic
spelling, without a dictionary. Fills cyrillic_errors and notes in both handcheck_sample.tsv files and
writes results/analysis/cyrillic_check.md. Words marked uncertain are not counted as errors.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# sample -> row -> ([(wrong, right)], [uncertain words])
ERRORS = {
    "flores_devtest": {
        12: ([("сезий", "цезий"), ("ёд", "йод")], []),
        16: ([("пеститсидларни", "пестицидларни")], []),
        18: ([("Сунами", "Цунами")], []),
        23: ([], ["Масиэй"]),
        24: ([("Броокс", "Брукс"), ("Тҳе", "The")], []),
        25: ([("санксиялар", "санкциялар")], []),
        29: ([], ["бҳажанларни"]),
        32: ([], ["Вегуилла"]),
        33: ([("Автомобил", "Автомобиль")], []),
        35: ([("ўФлиннинг", "О‘Флиннинг")], []),
        37: ([("пҲ", "pH")], []),
        39: ([("сивилизациясида", "цивилизациясида")], []),
        41: ([("Генус", "Genus"), ("Пантҳера", "Panthera")], []),
        42: ([("оселотлар", "оцелотлар")], []),
        54: ([("параллел", "параллель")], []),
        62: ([("италян", "итальян")], []),
        72: ([("обекти", "объекти")], []),
        73: ([("провинсияси", "провинцияси")], []),
        74: ([], ["ФКП", "ГБП"]),
        82: ([("Поинт", "Пойнт"), ("Фаирмонт", "Фэрмонт")], []),
        84: ([("алпинизм", "альпинизм"), ("алпинизм", "альпинизм")], []),
        86: ([("сиклонлар", "циклонлар")], []),
        92: ([("булварлар", "бульварлар")], []),
    },
    "belebele_uzn": {
        10: ([("италян", "итальян")], []),
        23: ([("Платсдарм", "Плацдарм")], []),
        31: ([("мил", "миль")], []),
        32: ([("мил", "миль")], []),
        33: ([("мил", "миль")], []),
        34: ([("мил", "миль")], []),
        35: ([("Реадинг", "Reading")], []),
        37: ([("Эммй", "Эмми")], []),
        45: ([], ["ЖАС"]),
        60: ([("реаксия", "реакция")], []),
        66: ([("Пеститсидлар", "Пестицидлар")], []),
        68: ([("ваксиналар", "вакциналар"), ("Энсефалитга", "Энцефалитга")], []),
        73: ([], ["ОҲА"]),
        91: ([("Жардíн", "Жардин"), ("Униóн", "Унион")], []),
        93: ([("Диэго", "Диего")], []),
        94: ([], ["Callejon"]),
        100: ([("Приднестрове", "Приднестровье")], []),
        143: ([("Мултфилмлар", "Мультфильмлар")], []),
    },
}
out = ["# Cyrillic review of the hand-check samples", "",
       "Done by Claude (an AI, not a native speaker), without a dictionary. Uncertain words are not counted as errors.", ""]
for name, found in ERRORS.items():
    path = ROOT / "data/parallel" / name / "handcheck_sample.tsv"
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        fields, rows = reader.fieldnames, list(reader)
    words = errors = uncertain = 0
    for n, r in enumerate(rows):
        words += sum(bool(re.search(r"\w", w)) for w in r["cyrillic"].split())
        wrong, unsure = found.get(n, ([], []))
        for w, _ in wrong:
            assert w in r["cyrillic"], (name, n, w)
        for w in unsure:
            assert w in r["cyrillic"], (name, n, w)
        errors += len(wrong)
        uncertain += len(unsure)
        if wrong:
            r["cyrillic_errors"] = "; ".join(f"{w} → {c}" for w, c in wrong)
        if unsure:
            r["notes"] = (r.get("notes") or "") + "Cyrillic, uncertain: " + ", ".join(unsure)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    bad = sum(bool(v[0]) for v in found.values())
    line = (f"**{name}:** {errors} errors in {words} words ({100 * errors / words:.1f}%); "
            f"{bad} of {len(rows)} rows have at least one error; {uncertain} more words uncertain.")
    out.append(line)
    print(line)
(ROOT / "results/analysis/cyrillic_check.md").write_text("\n\n".join(out) + "\n", encoding="utf-8")
