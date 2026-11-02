"""RQ3: what the token differences mean for API cost and usable context.

    python -X utf8 scripts/rq3_cost.py flores_devtest
Reads results/rq1/<name>_fertility.csv, writes results/rq3/<name>_context.md, <name>_cost.md.

Prices are only attached where the tokenizer is known to be the one the API bills with:
tiktoken 0.12.0 maps gpt-4o, gpt-4.1, gpt-5, gpt-5-mini and o3 to o200k_base. Prices are the
Standard tier input price per 1M tokens from https://developers.openai.com/api/docs/pricing,
read on 2026-10-30. Relative cost changes do not depend on the price at all (cost ∝ tokens).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import CONDITIONS  # noqa: E402
from progress import log  # noqa: E402

PRICES_CHECKED = "2026-10-30"
OPENAI_O200K_INPUT_USD_PER_M = {"gpt-4o": 2.50, "gpt-4.1": 2.00, "gpt-5": 1.25, "gpt-5-mini": 0.25, "o3": 2.00}
CONTEXT = 128_000
# Qwen 2.5 = Qwen 3 and Phi-4 = GPT-4 (cl100k) have identical tokenizers (checked in RQ1); keep one of each.
DROP = {"Qwen 3", "Phi-4", "UTF-8 bytes"}


def main(name):
    df = pd.read_csv(ROOT / "results/rq1" / f"{name}_fertility.csv")
    df = df[~df.tokenizer.isin(DROP)]
    tpw = df.pivot(index="tokenizer", columns="condition", values="tok_per_word")[CONDITIONS]
    out = ROOT / "results/rq3"
    out.mkdir(parents=True, exist_ok=True)

    words = (CONTEXT / tpw).round(-2).astype(int)
    vs_ascii = ((tpw.div(tpw["latin_ascii"], axis=0) - 1) * 100).round(1)
    md = [f"### Uzbek words that fit in a {CONTEXT:,}-token context ({name})\n", words.to_markdown(),
          "\n\n### Token (= cost) change vs `latin_ascii`, the most common way Uzbek is typed (%)\n",
          vs_ascii.to_markdown()]
    (out / f"{name}_context.md").write_text("\n".join(md), encoding="utf-8")

    o200k = tpw.loc["GPT-4o (o200k)"]
    cost = pd.DataFrame({m: (o200k * p).round(2) for m, p in OPENAI_O200K_INPUT_USD_PER_M.items()}).T
    cost.index.name = "model (o200k_base)"
    md = [f"### Input cost in USD per 1 million Uzbek words ({name}; prices checked {PRICES_CHECKED})\n",
          cost[CONDITIONS].to_markdown()]
    (out / f"{name}_cost.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[rq3] {name}: wrote results/rq3/{name}_context.md and {name}_cost.md")
    print("\n".join(md))
    print((out / f"{name}_context.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1])
