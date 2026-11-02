"""RQ2 analysis: accuracy per condition and paired differences vs latin_okina.

    python -X utf8 scripts/rq2_analyze.py results/rq2/<model_dir> [<model_dir> ...]
Writes <model_dir>/summary.md and summary.json.

CIs: cluster bootstrap over source articles (`link`; questions on the same passage are not
independent), 1000 resamples, seed 42, the same resamples for every condition (paired).
p-value: exact McNemar test on the questions where the two conditions disagree.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import CONDITIONS, EXTRA_CONDITIONS  # noqa: E402

SEED, N_BOOT, BASE = 42, 1000, "latin_okina"


def analyze(d: Path):
    res = {c: [json.loads(l) for l in (d / f"{c}.jsonl").read_text(encoding="utf-8").splitlines()]
           for c in CONDITIONS + EXTRA_CONDITIONS if (d / f"{c}.jsonl").exists()}
    n = {len(v) for v in res.values()}
    assert len(n) == 1, f"conditions have different question counts: {n}"
    ok = {c: np.array([r["pred"] == r["gold"] for r in v], dtype=float) for c, v in res.items()}
    links = [r["link"] for r in res[next(iter(res))]]
    clusters = pd.Series(range(len(links))).groupby(pd.Series(links)).apply(list).tolist()
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(N_BOOT)]

    rows = []
    for c, a in ok.items():
        accs = np.array([a[b].mean() for b in boots])
        row = {"condition": c, "n": len(a), "acc": a.mean(),
               "acc_lo": np.percentile(accs, 2.5), "acc_hi": np.percentile(accs, 97.5)}
        if c != BASE and BASE in ok:
            b0 = ok[BASE]
            diffs = np.array([a[b].mean() - b0[b].mean() for b in boots])
            win, lose = int(((a == 1) & (b0 == 0)).sum()), int(((a == 0) & (b0 == 1)).sum())
            row.update({"diff": a.mean() - b0.mean(), "diff_lo": np.percentile(diffs, 2.5),
                        "diff_hi": np.percentile(diffs, 97.5), "gained": win, "lost": lose,
                        "mcnemar_p": binomtest(win, win + lose).pvalue if win + lose else 1.0})
        rows.append(row)
    df = pd.DataFrame(rows)
    (d / "summary.json").write_text(df.to_json(orient="records", indent=1), encoding="utf-8")
    show = pd.DataFrame({
        "condition": df.condition, "n": df.n,
        "accuracy [95% CI]": df.apply(lambda r: f"{r.acc*100:.1f} [{r.acc_lo*100:.1f}, {r.acc_hi*100:.1f}]", axis=1),
        "Δ vs okina, pts [95% CI]": df.apply(lambda r: "—" if r.condition == BASE else
                                             f"{r['diff']*100:+.1f} [{r.diff_lo*100:+.1f}, {r.diff_hi*100:+.1f}]", axis=1),
        "gained/lost": df.apply(lambda r: "—" if r.condition == BASE else f"{int(r.gained)}/{int(r.lost)}", axis=1),
        "McNemar p": df.apply(lambda r: "—" if r.condition == BASE else f"{r.mcnemar_p:.3f}", axis=1)})
    md = f"### {d.name}\n\n" + show.to_markdown(index=False)
    (d / "summary.md").write_text(md, encoding="utf-8")
    print(md + "\n")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(Path(p))
