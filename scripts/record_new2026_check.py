"""Record the new-alphabet review of the hand-check samples and compute its error rates.

    python -X utf8 scripts/record_new2026_check.py
The review was done by Claude (an AI, not a native speaker), reading every word that the rules could
change. Fills new2026_errors and notes in both handcheck_sample.tsv files and writes
results/analysis/new2026_check.md. Words marked uncertain are not counted as errors.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {"flores_devtest": 522, "belebele_uzn": 187}  # words containing oʻ, gʻ, sh or ch
MISSED = {
    "flores_devtest": {22: [("[b]oʻlmagan", "[b]ölmagan")], 23: [("Polshalik", "Polşalik")], 32: [("Shnayderning", "Şnayderning")],
                       47: [("Machu", "Maçu"), ("Pikchuning", "Pikçuning")], 71: [("afrikanchadan", "afrikançadan")],
                       78: [("kapuchino", "kapuçino")], 90: [("Bellingshauzen", "Bellingşauzen")]},
    "belebele_uzn": {27: [("Shankar", "Şankar"), ("Choudxari", "Çoudxari")], 28: [("Shri", "Şri"), ("Shyam", "Şyam")],
                     60: [("Versache", "Versaçe")]},
}
UNCERTAIN = {"flores_devtest": {35: "oʻFlinning"}, "belebele_uzn": {}}
out = ["# New-alphabet review of the hand-check samples", "",
       "Done by Claude (an AI, not a native speaker). Uncertain words are not counted as errors. No word was converted that should not have been.", ""]
for name, found in MISSED.items():
    path = ROOT / "data/parallel" / name / "handcheck_sample.tsv"
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        fields, rows = reader.fieldnames, list(reader)
    for n, wrong in found.items():
        for w, _ in wrong:
            assert w in rows[n]["new2026"], (name, n, w)
        rows[n]["new2026_errors"] = "; ".join(f"{w} → {c}" for w, c in wrong)
    for n, w in UNCERTAIN[name].items():
        assert w in rows[n]["new2026"], (name, n, w)
        rows[n]["notes"] = ((rows[n].get("notes") or "") + " New alphabet, uncertain: " + w).strip()
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    errors = sum(map(len, found.values()))
    line = (f"**{name}:** {errors} of {CANDIDATES[name]} candidate words missed ({100 * errors / CANDIDATES[name]:.1f}%); "
            f"{len(found)} of {len(rows)} rows have at least one error; 0 false conversions.")
    out.append(line)
    print(line)
(ROOT / "results/analysis/new2026_check.md").write_text("\n\n".join(out) + "\n", encoding="utf-8")
