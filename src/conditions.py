"""Build the six spelling conditions of the same Uzbek text.

Input is Uzbek in 1995 Latin with any apostrophe characters. Steps:
  1. canonicalize: apostrophe after o/g -> U+02BB (okina), any other -> U+02BC (tutuq)
  2. latin_okina / latin_ascii / latin_backtick / latin_rsquo: swap both marks for one char,
     the way a keyboard with a single apostrophe key produces them
  3. new2026: the YangiAlifbo C# engine (lexicon + morphology + foreign filter)
  4. cyrillic: rule-based Latin->Cyrillic transliteration (see to_cyrillic)
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "converter/YangiAlifbo.Cli/bin/Release/net10.0/YangiAlifbo.Cli.dll"

OKINA = "ʻ"   # ʻ official mark for oʻ gʻ
TUTUQ = "ʼ"   # ʼ official tutuq belgisi
APOSTROPHES = "'‘’ʻʼ`´‛′"
_AP = re.compile(f"[{re.escape(APOSTROPHES)}]")

CONDITIONS = ["latin_okina", "latin_ascii", "latin_backtick", "latin_rsquo", "new2026", "cyrillic"]
VARIANT_CHAR = {"latin_okina": None, "latin_ascii": "'", "latin_backtick": "`", "latin_rsquo": "’"}


def canonicalize(text: str) -> str:
    """Map every apostrophe-like char to the official okina or tutuq by context."""
    out = []
    for i, ch in enumerate(text):
        if ch in APOSTROPHES:
            prev = text[i - 1] if i else ""
            is_okina = prev in "oOgG"
            out.append(OKINA if is_okina else TUTUQ)
        else:
            out.append(ch)
    return "".join(out)


def apostrophe_variant(canonical: str, char: str | None) -> str:
    return canonical if char is None else _AP.sub(char, canonical)


def to_new2026(lines: list[str], decisions_path: Path | None = None) -> list[str]:
    """Convert canonical lines with the YangiAlifbo engine; leftover ASCII ' -> tutuq."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        src.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = ["dotnet", str(CLI), str(src), str(dst)]
        if decisions_path:
            cmd += ["--words", str(decisions_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
        out = dst.read_text(encoding="utf-8").splitlines()
    assert len(out) == len(lines), "converter changed the line count"
    # The engine rewrites apostrophes in words it converts to ASCII '; any that remain are tutuq.
    return [l.replace("'", TUTUQ) for l in out], res.stderr.strip()


# ── Latin -> Cyrillic ────────────────────────────────────────────────
_DIGRAPHS = {"o" + OKINA: "ў", "g" + OKINA: "ғ", "sh": "ш", "ch": "ч",
             "yo": "ё", "yu": "ю", "ya": "я", "ye": "е"}
_SINGLE = dict(zip("abdfghijklmnopqrstuvxyz",
                   "абдфгҳижклмнопқрстувхйз"))
_SINGLE[TUTUQ] = "ъ"
_VOWELS = set("aeiouAEIOU" + "аеиоуўэёюяАЕИОУЎЭЁЮЯ")


def _case(src: str, cyr: str, word_upper: bool) -> str:
    if src[0].isupper():
        return cyr.upper() if (word_upper or len(src) == 1) else cyr[0].upper() + cyr[1:]
    return cyr


def to_cyrillic(text: str) -> str:
    """Rule-based transliteration of canonical 1995 Latin to Uzbek Cyrillic.

    Known limits (measured in the hand-check): Russian loans with ц/ь/щ, 'ts', and
    word-internal 'e' after a vowel in loans are not always restored.
    """
    out = []
    for word in re.split(r"(\W+)", text):
        # c (outside ch) and w are not Uzbek letters: such words (USOC, Twitter) stay Latin
        if not word or not any(c.isalpha() for c in word) or re.search(r"[wW]|[cC](?![hH])", word):
            out.append(word)
            continue
        upper = word.isupper() and len(word) > 1
        i, res = 0, []
        while i < len(word):
            two = word[i:i + 2].lower()
            # loanword -tsiya/-tsion is -ция/-цион (operatsiya -> операция)
            if two == "ts" and re.match(r"i[yo]", word[i + 2:i + 4].lower()):
                res.append(_case(word[i:i + 2], "ц", upper)); i += 2; continue
            # yoʻl is й+ў, not ё+ʻ
            if two in _DIGRAPHS and not (two == "yo" and word[i + 2:i + 3] == OKINA):
                res.append(_case(word[i:i + 2], _DIGRAPHS[two], upper)); i += 2; continue
            ch = word[i]
            lo = ch.lower()
            # s'h / c'h: the tutuq only separates the letters, Is'hoq -> Исҳоқ
            if ch == TUTUQ and i and word[i - 1] in "sScC" and word[i + 1:i + 2] in ("h", "H"):
                i += 1; continue
            if lo == "e":
                prev = word[i - 1] if i else ""
                cyr = "э" if (i == 0 or prev in _VOWELS) else "е"
            elif lo in _SINGLE:
                cyr = _SINGLE[lo]
            else:
                cyr = ch
            res.append(_case(ch, cyr, upper) if cyr != ch else ch)
            i += 1
        out.append("".join(res))
    return "".join(out)


def build_all(raw_lines: list[str], decisions_path: Path | None = None):
    canon = [canonicalize(l) for l in raw_lines]
    conds = {name: [apostrophe_variant(l, c) for l in canon] for name, c in VARIANT_CHAR.items()}
    conds["new2026"], log = to_new2026(canon, decisions_path)
    conds["cyrillic"] = [to_cyrillic(l) for l in canon]
    for name, keep in ABLATIONS.items():
        conds[name] = [partial_new(o, n, keep) for o, n in zip(canon, conds["new2026"])]
    return conds, log


# ── Ablations: only some of the new letters ──────────────────────────
# Which letters cause an accuracy change? new_og keeps only ö ğ (sh, ch stay), new_shch only ş ç.
ABLATIONS = {"new_og": set("öÖğĞ"), "new_shch": set("şŞçÇ")}
RQ4_CONDITIONS = ["backconv", "mixed_apostrophe"]
EXTRA_CONDITIONS = list(ABLATIONS) + RQ4_CONDITIONS

# ── RQ4: the fix (back-conversion) and realistic mixed typing ────────
_BACK = {"ö": "o" + OKINA, "ğ": "g" + OKINA, "ş": "sh", "ç": "ch"}


def back_convert(text: str) -> str:
    """Plain rule-based new -> 1995 Latin, the cheap fix a practitioner would apply. It knows nothing
    about the original text: all-caps words get SH/CH, others Sh/Ch."""
    def word(m):
        w = m.group(0)
        caps = len(w) > 1 and w.isupper()
        out = []
        for ch in w:
            lo = ch.lower()
            if lo in _BACK:
                b = _BACK[lo]
                out.append(b if ch == lo else (b.upper() if caps else b[0].upper() + b[1:]))
            else:
                out.append(ch)
        return "".join(out)
    return re.sub(r"\w+", word, text)


def mixed_apostrophes(canonical: str, seed: int) -> str:
    """Each word independently uses one of the four marks (seeded), as in text assembled from
    different keyboards and sources."""
    import random
    rng = random.Random(seed)
    marks = [OKINA, "'", "`", "’"]

    def word(m):
        mark = rng.choice(marks)
        return _AP.sub(mark, m.group(0))
    return re.sub(r"\S+", word, canonical)


def partial_new(old: str, new: str, keep: set[str]) -> str:
    """Walk the canonical and the engine-converted line together and keep only the new letters
    in `keep`; every other conversion is undone exactly as the old text had it. Words the engine
    left unchanged stay unchanged, so the ablations inherit the engine's decisions."""
    out, i, j = [], 0, 0
    while j < len(new):
        c = new[j]
        if c in "öÖğĞşŞçÇ" and (i >= len(old) or old[i] != c):
            two = old[i:i + 2]  # the 2-char spelling this letter replaced: oʻ gʻ sh ch
            out.append(c if c in keep else two)
            i, j = i + 2, j + 1
        else:
            # the engine also turns the tutuq inside converted words into ASCII and back to ʼ,
            # so characters may differ only there; take the old one
            out.append(old[i] if i < len(old) else c)
            i, j = i + 1, j + 1
    return "".join(out)
