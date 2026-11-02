"""Figure for RQ1: % token change vs official Latin (ʻ), tokenizer × spelling.

    python -X utf8 scripts/plot_rq1.py flores_devtest
Writes results/rq1/<name>_heatmap.png (and .pdf for the paper).
Diverging colors: blue = fewer tokens, orange = more, gray = no change; clipped at ±8% so the
small Latin differences stay visible (Cyrillic cells are labeled with their true value).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
df = pd.read_csv(ROOT / "results/rq1" / f"{name}_fertility.csv")

ROWS = ["GPT-4o (o200k)", "GPT-4 (cl100k)", "Llama 3.x", "Qwen 2.5", "Gemma 3", "Mistral Nemo", "DeepSeek V3"]
COLS = {"latin_ascii": "ASCII '", "latin_backtick": "backtick `", "latin_rsquo": "’ U+2019",
        "new2026": "new 2026\n(all 4)", "new_og": "only ö ğ", "new_shch": "only ş ç", "cyrillic": "Cyrillic"}
pct = (df.pivot(index="tokenizer", columns="condition", values="ratio_vs_okina") - 1) * 100
pct = pct.loc[ROWS, [c for c in COLS if c in pct.columns]]

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
v = pct.to_numpy()
ax.imshow(np.clip(v, -8, 8), cmap=cmap, norm=norm, aspect="auto")
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        strong = abs(v[i, j]) > 5.5
        ax.text(j, i, f"{v[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if strong else INK)
ax.set_xticks(range(v.shape[1]), [COLS[c] for c in pct.columns], fontsize=8, color=INK)
ax.set_yticks(range(v.shape[0]), ROWS, fontsize=8, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
# 2px surface gaps between cells
ax.set_xticks(np.arange(-.5, v.shape[1]), minor=True)
ax.set_yticks(np.arange(-.5, v.shape[0]), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
ax.set_title("Token count change vs official Latin (oʻ with U+02BB), %  —  FLORES-200 " + name.split("_")[-1],
             fontsize=9, color=INK, loc="left")
fig.text(0.01, 0.01, "Blue = fewer tokens, orange = more. Colors clipped at ±8%. 95% bootstrap CIs are within ±0.7 pts (see CSV).",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(ROOT / "results/rq1" / f"{name}_heatmap.{ext}", facecolor=SURFACE)
print(ROOT / "results/rq1" / f"{name}_heatmap.png")
