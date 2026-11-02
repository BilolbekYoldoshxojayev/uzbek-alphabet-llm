"""Write the tables of the long paper from saved results, so no number is typed by hand.

    python -X utf8 scripts/make_long_tables.py
Run scripts/analyze_all.py first. Writes paper/tables_long.tex.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/analysis"
N = json.loads((A / "numbers.json").read_text(encoding="utf-8"))
OLD, NEW, ONE, TWO = r"\emph{old}", r"\emph{new}", r"\"o\u{g}", r"\c{s}\c{c}"
GEMMA4 = "unsloth__gemma-3-4b-it__4bit"
out = []


def num(x, signed=True, d=1):
    s = f"{abs(x):.{d}f}"
    return f"${'-' if x < 0 else '+' if signed else ''}{s}$"


def bold(s, on):
    return s.replace("$", "$\\mathbf{", 1)[::-1].replace("$", "$}", 1)[::-1] if on else s


def table(star, caption, label, spec, head, rows):
    env = "table*" if star else "table"
    out.append(f"\\begin{{{env}}}[t]\n\\centering\\small\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
               f"\\resizebox{{\\{'textwidth' if star else 'columnwidth'}}}{{!}}{{%\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
               + head + " \\\\\n\\midrule\n" + " \\\\\n".join(rows) + " \\\\\n\\bottomrule\n\\end{tabular}}\n"
               f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


# Accuracy, every condition
df = pd.read_csv(ROOT / "results/rq2/summary_all.csv")
conds = ["latin_ascii", "latin_backtick", "latin_rsquo", "mixed_apostrophe", "new2026", "new_og", "new_shch", "backconv", "cyrillic"]
rows = []
for model in df.model.drop_duplicates():
    d = df[df.model == model].set_index("condition")
    rows.append(model.replace("(bf16)", "bf16").replace("(4-bit)", "4-bit") + f" & {d.okina_acc.iloc[0]:.1f} & "
                + " & ".join(bold(num(d.loc[c, "diff"]), d.loc[c, "p_holm"] < .05) for c in conds))
table(True, "Belebele accuracy (\\%) in official Latin, and change in points for every other spelling. Bold: significant after Holm correction over the nine comparisons per model (exact McNemar, $\\alpha=0.05$).",
      "tab:acc", "lrrrrrrrrrr", f"Model & okina & ascii & backtick & \\textquoteright{{}} & mixed & new & {ONE} & {TWO} & back-conv. & cyr", rows)

# The main gap with intervals
rows = []
for model in df.model.drop_duplicates():
    r = df[(df.model == model) & (df.condition == "new2026")].iloc[0]
    rows.append(f"{model.replace('(bf16)', 'bf16').replace('(4-bit)', '4-bit')} & {r.okina_acc + r['diff']:.1f} & {num(r['diff'])} & [{num(r.lo)}, {num(r.hi)}] & {r.p_holm:.4f}")
table(False, "The new-alphabet gap: accuracy (\\%) on new2026, change from okina in points, 95\\% cluster-bootstrap CI, and Holm-corrected $p$.",
      "tab:gap", "lrrrr", "Model & new2026 & $\\Delta$ & 95\\% CI & $p$", rows)

# Prompt frames
mass = {r["model"]: r for r in N["letter_mass"]}
frames = list(N["robustness"].items())
rows = []
for i, r in enumerate(frames[0][1]):
    m = r["model"]
    cells = []
    for (_, f), key in zip(frames, ("raw", "chat", "uz")):
        x = f[i]
        cells += [f"{x['okina']:.1f}", bold(num(x["diff"]), x["p_holm"] < .05), f"{mass[m][key]:.0f}"]
    rows.append(m + " & " + " & ".join(cells))
table(True, "Robustness to the prompt frame, for okina and new2026. For each frame: okina accuracy (\\%), change on new2026 in points (bold: significant after Holm correction over six models), and mean probability mass on the four answer letters (\\%).",
      "tab:frames", "l" + "rrr" * 3, "Model & \\multicolumn{3}{c}{English, raw} & \\multicolumn{3}{c}{Chat template} & \\multicolumn{3}{c}{Uzbek frame}", rows)

# Dose-response
rows = [f"{r['model']} & {num(r['slope'], d=2)} & [{num(r['lo'], d=2)}, {num(r['hi'], d=2)}] & {num(r['third 1'])} & {num(r['third 2'])} & {num(r['third 3'])}" for r in N["dose"]]
table(False, "Dose--response (exploratory). Slope: points of accuracy gap per additional new letter per 100 letters, with 95\\% cluster-bootstrap CI. Thirds: gap in points, from fewest to most new letters.",
      "tab:dose", "lrrrrr", "Model & Slope & 95\\% CI & Low & Mid & High", rows)

# Context
text = (ROOT / "results/rq3/flores_devtest_context.md").read_text(encoding="utf-8")
lines = [l for l in text.split("\n\n")[1].splitlines() if l.startswith("|")]
head = [c.strip() for c in lines[0].strip("|").split("|")]
idx = [head.index(c) for c in ("latin_okina", "latin_ascii", "new2026", "cyrillic")]
rows = []
for l in lines[2:]:
    c = [x.strip() for x in l.strip("|").split("|")]
    rows.append(re.sub(r"\s*\(.*\)", "", c[0]) + " & " + " & ".join(f"{int(c[i]):,}".replace(",", "{,}") for i in idx))
table(False, "Uzbek words that fit in a 128{,}000-token context, by tokenizer (FLORES devtest).", "tab:context", "lrrrr",
      "Tokenizer & okina & ascii & new & cyr", rows)

# LoRA, per seed
rows = []
for s in (42, 43, 44):
    cells = []
    for sp in ("old", "new"):
        m = json.loads((ROOT / f"results/train/lora/{GEMMA4}__lora_{sp}_s{s}/metrics.json").read_text())
        acc = {r["condition"]: r["acc"] for r in json.loads((ROOT / f"results/rq2/{GEMMA4}__lora_{sp}_s{s}/summary.json").read_text())}
        cells += [f"{m['train_seconds'] / 60:.1f}", f"{m['tokens_per_second']:.0f}", f"{100 * acc['latin_okina']:.1f}", f"{100 * acc['new2026']:.1f}"]
    rows.append(f"{s} & " + " & ".join(cells))
table(False, f"LoRA runs per seed: training minutes, tokens/s, and accuracy (\\%) on old and new text, trained on old or new spelling. Base model: 59.9 on old text, 53.1 on new.",
      "tab:lora", "rrrrrrrrr", "Seed & \\multicolumn{4}{c}{Trained on old} & \\multicolumn{4}{c}{Trained on new}", rows)
out[-1] = out[-1].replace("\\midrule", "& min & tok/s & old & new & min & tok/s & old & new \\\\\n\\midrule", 1)

# From scratch, per seed
rows = []
for mode in ("separate", "shared"):
    for s in (42, 43, 44):
        o, n = (json.loads((ROOT / f"results/train/scratch/{mode}_{sp}_s{s}/metrics.json").read_text()) for sp in ("old", "new"))
        rows.append(f"{mode} & {s} & {o['train_seconds'] / 60:.1f} & {n['train_seconds'] / 60:.1f} & {o['gpu_energy_wh']:.1f} & {n['gpu_energy_wh']:.1f} & {o['final_valid_bits_per_word']:.2f} & {n['final_valid_bits_per_word']:.2f}")
table(False, "From-scratch runs per seed: training minutes, GPU energy (Wh) and final validation bits per word, on old and new spelling.",
      "tab:scratch", "lrrrrrrr", "Tokenizer & Seed & \\multicolumn{2}{c}{Min} & \\multicolumn{2}{c}{Wh} & \\multicolumn{2}{c}{Bits/word}", rows)
out[-1] = out[-1].replace("\\midrule", "& & old & new & old & new & old & new \\\\\n\\midrule", 1)

(ROOT / "paper/tables_long.tex").write_text("% Generated by scripts/make_long_tables.py. Do not edit by hand.\n" + "\n".join(out), encoding="utf-8")
print(len(out), "tables")
print("\n".join(out)[:1400])
