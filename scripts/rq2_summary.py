"""Cross-model RQ2 summary: Δ accuracy vs latin_okina per model × condition, with Holm correction.

    python -X utf8 scripts/rq2_summary.py
Reads results/rq2/*/summary.json (written by rq2_analyze.py). Writes results/rq2/summary_all.md,
summary_all.csv and summary_delta.png/.pdf.
Holm: within each model, all comparisons against latin_okina are corrected together
(controls the chance of any false positive per model at 5%).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # folder -> display name (order = plot order)
    "unsloth__gemma-3-4b-it": "Gemma 3 4B (bf16)",
    "unsloth__gemma-3-4b-it__4bit": "Gemma 3 4B (4-bit)",
    "Qwen__Qwen2.5-7B-Instruct__4bit": "Qwen 2.5 7B",
    "unsloth__Llama-3.1-8B-Instruct__4bit": "Llama 3.1 8B",
    "Qwen__Qwen3-8B__4bit": "Qwen 3 8B",
    "unsloth__gemma-3-12b-it__4bit": "Gemma 3 12B",
    "unsloth__Mistral-Nemo-Instruct-2407__4bit": "Mistral Nemo 12B",
}
COND = {"latin_ascii": "ASCII '", "latin_backtick": "backtick `", "latin_rsquo": "’", "new2026": "new 2026",
        "new_og": "only ö ğ", "new_shch": "only ş ç", "cyrillic": "Cyrillic",
        "backconv": "back-conv.", "mixed_apostrophe": "mixed '"}


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


rows = []
for folder, name in MODELS.items():
    f = ROOT / "results/rq2" / folder / "summary.json"
    if not f.exists():
        continue
    s = [r for r in json.loads(f.read_text(encoding="utf-8")) if r["condition"] in COND]
    base = next(r for r in json.loads(f.read_text(encoding="utf-8")) if r["condition"] == "latin_okina")
    for r, p in zip(s, holm([r["mcnemar_p"] for r in s])):
        rows.append({"model": name, "condition": r["condition"], "okina_acc": base["acc"] * 100,
                     "diff": r["diff"] * 100, "lo": r["diff_lo"] * 100, "hi": r["diff_hi"] * 100,
                     "p": r["mcnemar_p"], "p_holm": p})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/rq2/summary_all.csv", index=False)
cell = df.assign(c=lambda d: d.apply(lambda r: f"{r['diff']:+.1f}{'*' if r.p_holm < .05 else ''}", axis=1))
tab = cell.pivot(index="model", columns="condition", values="c")[[c for c in COND if c in set(df.condition)]]
tab = tab.reindex([m for m in MODELS.values() if m in tab.index])
tab.insert(0, "okina acc %", df.groupby("model").okina_acc.first().reindex(tab.index).round(1))
tab.columns = ["okina acc %"] + [COND[c] for c in tab.columns[1:]]
md = ("### Belebele uzn: accuracy change vs official Latin (ʻ), percentage points\n\n"
      "`*` = significant after Holm correction (all comparisons per model, exact McNemar, α = 0.05).\n\n"
      + tab.to_markdown())
(ROOT / "results/rq2/summary_all.md").write_text(md, encoding="utf-8")
print(md)

# Figure: dot + 95% CI per model, for the conditions that matter most
show = ["latin_ascii", "new2026", "new_og", "new_shch", "cyrillic"]
colors = {"latin_ascii": "#898781", "new2026": "#eb6834", "new_og": "#2a78d6", "new_shch": "#1baf7a", "cyrillic": "#6250d6"}
models = [m for m in MODELS.values() if m in set(df.model)]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200, facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
w = 0.14
for k, c in enumerate(show):
    d = df[df.condition == c].set_index("model").reindex(models)
    y = [i + (k - 2) * w for i in range(len(models))]
    ax.errorbar(d["diff"], y, xerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="o", ms=4, lw=1.2,
                color=colors[c], label=COND[c], capsize=0)
ax.axvline(0, color="#52514e", lw=0.8)
ax.set_yticks(range(len(models)), models, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Accuracy change vs official Latin ʻ (percentage points, 95% cluster-bootstrap CI)", fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(axis="x", color="#e4e3df", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / "results/rq2" / f"summary_delta.{ext}", facecolor="#fcfcfb")
