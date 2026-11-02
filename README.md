# How Uzbekistan's 2026 alphabet affects LLMs

**Research question:** how does the change to Uzbek's Latin alphabet affect the tokenization, accuracy, cost and training of large language models?

On 21 October 2026 Uzbekistan adopted a revised Latin alphabet (oʻ→ö, gʻ→ğ, sh→ş, ch→ç). Uzbek is also typed today with several different oʻ/gʻ marks and is still widely written in Cyrillic. We measure what these spellings of the *same text* do to LLMs.

## Main results

- **Tokens:** compared with ASCII-apostrophe Uzbek, the new alphabet changes token counts by −3.0% to +1.3%.
- **Accuracy:** it lowers Belebele accuracy in all 7 runs of 6 models, by 3.1–9.6 points, all significant. Apostrophe variants have no significant effect.
- **Fix:** converting back to the old spelling removes the loss on our converted text.
- **Fine-tuning:** training Gemma 3 4B on 4M characters of new-alphabet text changes its new-minus-old gap from −6.8 to +0.3 points. This is one model, and the training text came from the same converter as the test text.
- **Training from scratch:** no consistent old-vs-new difference in tokens, time, energy or quality, with 3 seeds.

Full tables: [results/rq2/summary_all.md](results/rq2/summary_all.md) and [results/analysis/](results/analysis/).

## Conditions

| Condition | Example | How it is made |
|---|---|---|
| `latin_okina` | oʻzbekcha | official 1995 Latin: ʻ U+02BB for oʻ/gʻ, ʼ U+02BC for tutuq |
| `latin_ascii` | o'zbekcha | every apostrophe → `'` U+0027 |
| `latin_backtick` | o`zbekcha | every apostrophe → `` ` `` U+0060 |
| `latin_rsquo` | o’zbekcha | every apostrophe → `’` U+2019 |
| `new2026` | özbekça | [YangiAlifbo](https://github.com/BilolbekYoldoshxojayev/yangicha-alifbo) engine, strict mode, output used as-is |
| `cyrillic` | ўзбекча | rule-based transliteration ([src/conditions.py](src/conditions.py)) |
| `new_og`, `new_shch` | | only ö/ğ, or only ş/ç |
| `backconv` | | new2026 converted back to 1995 spelling |
| `mixed_apostrophe` | | a seeded random apostrophe per word |

## Layout

```
converter/YangiAlifbo.Cli/   batch CLI over the YangiAlifbo C# engine
src/conditions.py            builds the conditions
scripts/                     data building, experiments and analysis
data/parallel/               the parallel data in every spelling, plus hand-check samples
results/                     every per-question result, summary, training metric and log
paper/                       short (main.tex) and long (main_long.tex) papers
docs/WALKTHROUGH.md          plain-language explanation of everything that was done
```

Not included: model weights, the source datasets, and the training corpus. `scripts/build_train_corpus.py` rebuilds the corpus from FineWeb-2 (ODC-By 1.0).

## Reproduce

Requirements: Windows or Linux, .NET 10 SDK, Python 3.12, and a 12 GB GPU for the models. The results here came from an RTX 4080 Laptop GPU.

```bash
git clone https://github.com/BilolbekYoldoshxojayev/yangicha-alifbo external/yangicha-alifbo
git -C external/yangicha-alifbo checkout fdaafae2249fbde3505a9166ff207adfc5cd6b5d
dotnet build -c Release converter/YangiAlifbo.Cli
python -m venv .venv && .venv/Scripts/pip install -r requirements.lock.txt
python -X utf8 scripts/build_parallel.py data/raw/flores200/flores200_dataset/devtest/uzn_Latn.devtest flores_devtest
python -X utf8 scripts/rq1_fertility.py flores_devtest
python -X utf8 scripts/run_all_experiments.py
python -X utf8 scripts/analyze_all.py
python -X utf8 scripts/make_long_tables.py
```

The full experiment run takes about 17 hours. Finished steps are skipped on restart. The analysis needs no GPU.

## Known limits

- New-alphabet and Cyrillic text are machine-converted, not written by people.
- The conversion error rates come from one annotator, the author of the converter, who hand-checked a sample after a first AI review. They are not independent.
- We had limited GPU power (one 12 GB laptop GPU), so models stop at 12B, and training runs are small.
- `results/rq2/` contains the re-run Llama results only. An early run was discarded because of a corrupted weight file.

## Use of AI

Claude (an AI assistant) wrote the evaluation and analysis code, ran the experiments, did a first review of the converted text samples, contributed to the detailed study design and drafted the paper. The author conceived the study, defined its research questions and conditions, proposed the training comparison, built the YangiAlifbo converter, and hand-checked the converted text samples. See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## Data and licenses

| Item | License |
|---|---|
| Code | MIT ([LICENSE](LICENSE)) |
| `data/parallel/` (adapted from FLORES-200 and Belebele @ `7899cdf`) | CC BY-SA 4.0 ([details](data/parallel/LICENSE.md)) |
| Training corpus (not included): FineWeb-2 @ `af9c133` | ODC-By 1.0 |

Gold answers in Belebele are not balanced, so always answering the most common option scores 27.9%, not 25%.
