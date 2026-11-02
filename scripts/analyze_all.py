"""Validity checks and analysis of the full run. Reads only saved results; no model is run.

    python -X utf8 scripts/analyze_all.py
Writes results/analysis/: validity.md, exp_b.md, exp_a.md, dose_response.md, robustness.md, rq4.md,
numbers.json (every value the paper quotes from these analyses) and exp_a_curves.png/.pdf.
CIs: cluster bootstrap over source articles, 1000 resamples, seed 42, as in rq2_analyze.py.
"""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R2, OUT = ROOT / "results/rq2", ROOT / "results/analysis"
OUT.mkdir(exist_ok=True)
SEEDS, N_BOOT = [42, 43, 44], 1000
GEMMA4 = "unsloth__gemma-3-4b-it__4bit"
MODELS = {GEMMA4: "Gemma 3 4B", "Qwen__Qwen2.5-7B-Instruct__4bit": "Qwen 2.5 7B",
          "unsloth__Llama-3.1-8B-Instruct__4bit": "Llama 3.1 8B", "Qwen__Qwen3-8B__4bit": "Qwen 3 8B",
          "unsloth__gemma-3-12b-it__4bit": "Gemma 3 12B", "unsloth__Mistral-Nemo-Instruct-2407__4bit": "Mistral Nemo 12B"}
NEW_LETTERS = set("öğşçÖĞŞÇ")
numbers = {}


def rows(folder, cond, root=R2):
    return [json.loads(l) for l in (root / folder / f"{cond}.jsonl").read_text(encoding="utf-8").splitlines()]


def correct(folder, cond):
    return np.array([r["pred"] == r["gold"] for r in rows(folder, cond)], dtype=float)


def belebele(cond):
    return [json.loads(l) for l in (ROOT / f"data/parallel/belebele_uzn/{cond}.jsonl").read_text(encoding="utf-8").splitlines()]


LINKS = [r["link"] for r in rows(GEMMA4, "latin_okina")]
CLUSTERS = pd.Series(range(len(LINKS))).groupby(pd.Series(LINKS)).apply(list).tolist()
_rng = np.random.default_rng(42)
BOOTS = [np.concatenate([CLUSTERS[j] for j in _rng.integers(0, len(CLUSTERS), len(CLUSTERS))]) for _ in range(N_BOOT)]


def ci(a, b):
    """Difference a-b in points, with a paired cluster-bootstrap CI."""
    d = np.array([a[i].mean() - b[i].mean() for i in BOOTS]) * 100
    return (a.mean() - b.mean()) * 100, np.percentile(d, 2.5), np.percentile(d, 97.5)


def mcnemar(a, b):
    w, l = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return binomtest(w, w + l).pvalue if w + l else 1.0


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def fmt(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def save(name, title, parts):
    text = f"# {title}\n\n" + "\n\n".join(parts) + "\n"
    (OUT / name).write_text(text, encoding="utf-8")
    print(text)


def validity():
    parts = []
    # 1. Every result file is complete and in the same question order.
    bad, n = [], 0
    for d in sorted(p for p in R2.iterdir() if p.is_dir() and "limit" not in p.name):
        for f in d.glob("*.jsonl"):
            n += 1
            r = rows(d.name, f.stem)
            if len(r) != 900 or [x["link"] for x in r] != LINKS:
                bad.append(f"{d.name}/{f.name}")
    parts.append(f"**Files:** {n} result files checked; incomplete or out of order: {bad or 'none'}.")
    numbers["result_files"] = n

    # 2. Code and data did not change between restarts, or since.
    ms = sorted((ROOT / "results/logs").glob("manifest_2026110*.json"))
    h = [json.loads(m.read_text(encoding="utf-8"))["sha256"] for m in ms[-3:]]
    changed_between = sorted(k for k in h[-1] if any(x.get(k) != h[-1][k] for x in h))
    now = []
    for k, v in h[-1].items():
        p = ROOT / k
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != v:
            now.append(k)
    parts.append(f"**Manifests:** {len(h)} from this run. Files differing between them: {changed_between or 'none'}. "
                 f"Files changed since the last manifest: {now or 'none'}.")

    # 3. Determinism: same model and prompt, separate process.
    a, b = "unsloth__gemma-3-4b-it", "unsloth__gemma-3-4b-it__limit50"
    same, gap = 0, 0.0
    for c in ("latin_okina", "new2026"):
        x, y = rows(a, c)[:50], rows(b, c)
        same += sum(p["pred"] == q["pred"] for p, q in zip(x, y))
        gap = max(gap, max(abs(u - v) for p, q in zip(x, y) for u, v in zip(p["logprobs"], q["logprobs"])))
    parts.append(f"**Determinism (Gemma 3 4B bf16):** {same}/100 predictions identical on repeat; largest log-prob difference {gap:.2e}.")
    numbers["determinism"] = {"same_of_100": same, "max_logprob_diff": gap}

    # 4. Llama before and after the corrupted weight file was repaired.
    q = ROOT / "results/rq2_quarantine/Llama-3.1-8B__corrupted_shard2"
    t = []
    for c in sorted(f.stem for f in q.glob("*.jsonl")):
        x, y = rows("", c, q), rows("unsloth__Llama-3.1-8B-Instruct__4bit", c)
        t.append({"condition": c, "corrupted acc": round(100 * np.mean([r["pred"] == r["gold"] for r in x]), 1),
                  "repaired acc": round(100 * np.mean([r["pred"] == r["gold"] for r in y]), 1),
                  "same predictions %": round(100 * np.mean([p["pred"] == r["pred"] for p, r in zip(x, y)]), 1)})
    parts.append("**Llama, corrupted file vs repaired** (the paper uses repaired only):\n\n" + pd.DataFrame(t).to_markdown(index=False))

    # 5. Probability mass on the four answer letters.
    t = []
    for folder, name in MODELS.items():
        row = {"model": name}
        for suffix, label in (("", "raw"), ("__chatprompt", "chat"), ("__uzprompt", "uz")):
            v = [np.exp(r["logprobs"]).sum() for c in ("latin_okina", "new2026") for r in rows(folder + suffix, c)]
            row[label] = round(100 * float(np.mean(v)), 1)
        t.append(row)
    numbers["letter_mass"] = t
    parts.append("**Mean probability mass on A-D, % (okina and new2026 pooled):**\n\n" + pd.DataFrame(t).to_markdown(index=False))

    # 6. Heat, per training run.
    g = pd.read_csv(ROOT / "results/logs/gpu_monitor.csv", on_bad_lines="skip")
    g["t"] = pd.to_datetime(g["time"], errors="coerce")
    g["slow"] = (g.iloc[:, 6] == "Active") | (g.iloc[:, 7] == "Active")
    g["temp"] = pd.to_numeric(g["temperature.gpu"], errors="coerce")
    t = []
    for line in (ROOT / "results/logs/commands.log").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if not p[1].startswith(("lora_", "scratch_")) or not p[0].startswith("2026-11-0"):
            continue
        end = pd.Timestamp(p[0])
        w = g[(g.t > end - pd.Timedelta(seconds=int(p[3].split("=")[1]))) & (g.t <= end)]
        t.append({"run": p[1], "readings": len(w), "mean C": round(w.temp.mean(), 1), "max C": int(w.temp.max()),
                  "slowdown %": round(100 * w.slow.mean(), 2)})
    t = pd.DataFrame(t)
    numbers["heat"] = {"max_c": int(t["max C"].max()), "max_slowdown_pct": float(t["slowdown %"].max()),
                       "before_fans_slowdown_pct": round(100 * g[g.t < "2026-11-01 13:38"].slow.mean(), 1)}
    parts.append("**Heat during training** (monitor reads every 10 s):\n\n" + t.to_markdown(index=False))
    save("validity.md", "Validity checks", parts)


def exp_b():
    root = ROOT / "results/train/lora"
    m = {(sp, s): json.loads((root / f"{GEMMA4}__lora_{sp}_s{s}/metrics.json").read_text()) for sp in ("old", "new") for s in SEEDS}
    fields = {"train_tokens": "tokens", "train_seconds": "train s", "tokens_per_second": "tok/s", "gpu_energy_wh": "energy Wh",
              "peak_gpu_memory_gb": "peak GB"}
    t = [{"spelling": sp, "seed": s, **{v: m[sp, s][k] for k, v in fields.items()},
          "bits/word before": round(m[sp, s]["valid_before"]["bits_per_word"], 3),
          "bits/word after": round(m[sp, s]["valid_after"]["bits_per_word"], 3)} for sp in ("old", "new") for s in SEEDS]
    t = pd.DataFrame(t)
    g = t.groupby("spelling").agg(["mean", "std"]).drop(columns="seed", level=0).round(2)
    summary = {sp: {c: {"mean": float(t[t.spelling == sp][c].mean()), "sd": float(t[t.spelling == sp][c].std())}
                    for c in t.columns if c not in ("spelling", "seed")} for sp in ("old", "new")}
    numbers["exp_b_training"] = summary
    parts = ["**Per run:**\n\n" + t.to_markdown(index=False), "**Mean and SD:**\n\n" + g.to_markdown()]

    base = {c: correct(GEMMA4, c) for c in ("latin_okina", "new2026")}
    trained = {(sp, c, s): correct(f"{GEMMA4}__lora_{sp}_s{s}", c) for sp in ("old", "new") for c in base for s in SEEDS}
    avg = {(sp, c): np.mean([trained[sp, c, s] for s in SEEDS], axis=0) for sp in ("old", "new") for c in base}

    t = []
    for s in SEEDS:
        row = {"seed": s}
        for sp in ("old", "new"):
            for c in base:
                row[f"{sp}-trained, {c}"] = round(100 * trained[sp, c, s].mean(), 1)
        t.append(row)
    parts.append(f"**Accuracy %.** Base model: latin_okina {100 * base['latin_okina'].mean():.1f}, new2026 {100 * base['new2026'].mean():.1f}.\n\n"
                 + pd.DataFrame(t).to_markdown(index=False))

    pairs = {"old-trained on old text vs base": (avg["old", "latin_okina"], base["latin_okina"]),
             "new-trained on old text vs base": (avg["new", "latin_okina"], base["latin_okina"]),
             "old-trained on new text vs base": (avg["old", "new2026"], base["new2026"]),
             "new-trained on new text vs base": (avg["new", "new2026"], base["new2026"]),
             "gap new-old text, base": (base["new2026"], base["latin_okina"]),
             "gap new-old text, old-trained": (avg["old", "new2026"], avg["old", "latin_okina"]),
             "gap new-old text, new-trained": (avg["new", "new2026"], avg["new", "latin_okina"]),
             "on new text: new-trained vs old-trained": (avg["new", "new2026"], avg["old", "new2026"]),
             "on old text: new-trained vs old-trained": (avg["new", "latin_okina"], avg["old", "latin_okina"]),
             "own spelling: new-trained on new vs old-trained on old": (avg["new", "new2026"], avg["old", "latin_okina"])}
    t, nb = [], {}
    for k, (a, b) in pairs.items():
        c = ci(a, b)
        nb[k] = [round(float(x), 2) for x in c]
        t.append({"comparison (mean of 3 seeds)": k, "points [95% CI]": fmt(c)})
    numbers["exp_b_accuracy"] = nb
    parts.append("**Comparisons:**\n\n" + pd.DataFrame(t).to_markdown(index=False))

    # Per seed, the two comparisons that matter, each with McNemar and Holm.
    t = []
    for s in SEEDS:
        t.append(("new-trained vs base, new text", s, trained["new", "new2026", s], base["new2026"]))
        t.append(("new-trained vs old-trained, new text", s, trained["new", "new2026", s], trained["old", "new2026", s]))
        t.append(("own spelling, new vs old", s, trained["new", "new2026", s], trained["old", "latin_okina", s]))
    p = holm([mcnemar(a, b) for _, _, a, b in t])
    parts.append("**Per seed:**\n\n" + pd.DataFrame(
        [{"comparison": n, "seed": s, "points": round(100 * (a.mean() - b.mean()), 1), "p Holm": round(q, 4)}
         for (n, s, a, b), q in zip(t, p)]).to_markdown(index=False))
    numbers["exp_b_per_seed"] = [{"comparison": n, "seed": s, "points": round(100 * (a.mean() - b.mean()), 2), "p_holm": q}
                                 for (n, s, a, b), q in zip(t, p)]
    save("exp_b.md", "Experiment B: LoRA on Gemma 3 4B, old vs new spelling", parts)


def exp_a():
    root = ROOT / "results/train/scratch"
    t = []
    for mode in ("separate", "shared"):
        for sp in ("old", "new"):
            for s in SEEDS:
                m = json.loads((root / f"{mode}_{sp}_s{s}/metrics.json").read_text())
                t.append({"tokenizer": mode, "spelling": sp, "seed": s, "tokens": m["train_tokens"], "tok/char": round(m["tokens_per_char"], 4),
                          "steps": m["steps"], "train s": m["train_seconds"], "tok/s": m["tokens_per_second"], "energy Wh": m["gpu_energy_wh"],
                          "peak GB": m["peak_gpu_memory_gb"], "bits/word": m["final_valid_bits_per_word"], "valid words": m["valid_words"],
                          "tokenizer s": m["tokenizer_seconds"]})
    t = pd.DataFrame(t)
    cols = ["tokens", "tok/char", "steps", "train s", "tok/s", "energy Wh", "peak GB", "bits/word"]
    g = t.groupby(["tokenizer", "spelling"])[cols].agg(["mean", "std"]).round(3)
    d = []
    for mode in ("separate", "shared"):
        x = t[t.tokenizer == mode].pivot(index="seed", columns="spelling")
        for c in cols:
            diff = x[c]["new"] - x[c]["old"]
            d.append({"tokenizer": mode, "metric": c, "old mean": round(x[c]["old"].mean(), 3), "new mean": round(x[c]["new"].mean(), 3),
                      "new-old per seed": ", ".join(f"{v:+.3f}" for v in diff), "mean %": round(100 * diff.mean() / x[c]["old"].mean(), 2)})
    numbers["exp_a"] = d
    numbers["exp_a_valid_words"] = sorted({int(v) for v in t["valid words"]})
    numbers["exp_a_tokenizer_seconds"] = {f"{a}_{b}": float(v) for (a, b), v in t.groupby(["tokenizer", "spelling"])["tokenizer s"].first().items()}
    save("exp_a.md", "Experiment A: from-scratch model, old vs new spelling",
         ["**Per run:**\n\n" + t.to_markdown(index=False), "**Mean and SD:**\n\n" + g.to_markdown(),
          "**New minus old, paired by seed:**\n\n" + pd.DataFrame(d).to_markdown(index=False)])

    # Learning curves: mean of 3 seeds, in the two panels of a small multiple. Identity is also in the line style.
    colors, styles = {"old": "#2a78d6", "new": "#eb6834"}, {"old": "-", "new": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200, sharey=True, facecolor="#fcfcfb")
    for ax, mode in zip(axes, ("separate", "shared")):
        ax.set_facecolor("#fcfcfb")
        for sp in ("old", "new"):
            c = [list(csv.DictReader(open(root / f"{mode}_{sp}_s{s}/curve.csv"))) for s in SEEDS]
            x = [100 * float(r["train_docs_fraction"]) for r in c[0]]
            y = np.array([[float(r["valid_bits_per_word"]) for r in k] for k in c])
            ax.plot(x, y.mean(0), color=colors[sp], ls=styles[sp], lw=2, label=f"{sp} spelling")
        ax.set_title(f"{mode} tokenizer" + ("s" if mode == "separate" else ""), fontsize=9)
        ax.set_xlabel("Training data seen (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Validation bits per word", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"exp_a_curves.{ext}", facecolor="#fcfcfb")


def dose_response():
    fields = ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4")
    dens, length = [], []
    for r in belebele("new2026"):
        text = " ".join(r[f] for f in fields)
        letters = sum(ch.isalpha() for ch in text)
        dens.append(100 * sum(ch in NEW_LETTERS for ch in text) / letters)
        length.append(np.log(letters))
    dens, length = np.array(dens), np.array(length)

    def slope(y, i, control=False):
        X = np.column_stack([np.ones(len(i)), dens[i]] + ([length[i]] if control else []))
        return np.linalg.lstsq(X, y[i], rcond=None)[0][1] * 100

    diffs = {n: correct(f, "new2026") - correct(f, "latin_okina") for f, n in MODELS.items()}
    every = np.arange(len(dens))
    edges = np.percentile(dens, [100 / 3, 200 / 3])
    third = np.digitize(dens, edges)
    t = []
    for n, y in list(diffs.items()) + [("Pooled, 6 models", np.mean(list(diffs.values()), axis=0))]:
        b = [slope(y, i) for i in BOOTS]
        c = [slope(y, i, True) for i in BOOTS]
        t.append({"model": n, "slope": round(slope(y, every), 2), "lo": round(np.percentile(b, 2.5), 2), "hi": round(np.percentile(b, 97.5), 2),
                  "slope, length controlled": round(slope(y, every, True), 2), "lo2": round(np.percentile(c, 2.5), 2), "hi2": round(np.percentile(c, 97.5), 2),
                  **{f"third {k + 1}": round(100 * y[third == k].mean(), 1) for k in range(3)}})
    numbers["dose"] = t
    numbers["dose_density"] = {"mean": float(dens.mean()), "sd": float(dens.std()), "edges": [float(e) for e in edges]}
    save("dose_response.md", "Dose-response (exploratory)",
         [f"New letters per 100 letters: mean {dens.mean():.2f}, SD {dens.std():.2f}, tertile edges {edges[0]:.2f} and {edges[1]:.2f}. "
          "Slope: change in the new2026-minus-okina accuracy gap, in points, per one more new letter per 100 letters. Thirds: the gap in points, from fewest to most new letters.",
          pd.DataFrame(t).to_markdown(index=False)])


def robustness():
    parts, nb = [], {}
    for suffix, label in (("", "English frame, raw"), ("__chatprompt", "Chat template"), ("__uzprompt", "Uzbek frame")):
        t = []
        for f, n in MODELS.items():
            a, b = correct(f + suffix, "new2026"), correct(f + suffix, "latin_okina")
            t.append((n, 100 * b.mean(), ci(a, b), mcnemar(a, b)))
        ps = holm([x[3] for x in t])
        parts.append(f"**{label}:**\n\n" + pd.DataFrame([{"model": x[0], "okina %": round(x[1], 1), "new-okina [95% CI]": fmt(x[2]), "p Holm": round(q, 4)}
                                                        for x, q in zip(t, ps)]).to_markdown(index=False))
        nb[label] = [{"model": x[0], "okina": round(x[1], 1), "diff": round(float(x[2][0]), 1), "p_holm": q} for x, q in zip(t, ps)]
    numbers["robustness"] = nb
    save("robustness.md", "Robustness of the new-alphabet gap to the prompt", parts)


def rq4():
    t = []
    for f, n in MODELS.items():
        s = {r["condition"]: r for r in json.loads((R2 / f / "summary.json").read_text(encoding="utf-8"))}
        base = rows(f, "latin_okina")
        row = {"model": n}
        for c in ("new2026", "backconv", "mixed_apostrophe"):
            row[c] = f"{s[c]['diff'] * 100:+.1f} [{s[c]['diff_lo'] * 100:+.1f}, {s[c]['diff_hi'] * 100:+.1f}]"
        row["backconv same answers %"] = round(100 * np.mean([p["pred"] == q["pred"] for p, q in zip(base, rows(f, "backconv"))]), 1)
        t.append(row)
    numbers["rq4"] = t
    save("rq4.md", "RQ4: normalization", ["Change in accuracy vs okina, points [95% CI]:", pd.DataFrame(t).to_markdown(index=False)])


validity()
exp_b()
exp_a()
dose_response()
robustness()
rq4()
(OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float), encoding="utf-8")
