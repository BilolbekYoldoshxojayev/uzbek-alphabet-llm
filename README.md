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

### Accuracy per model

Belebele (900 questions), change in accuracy against the official Latin spelling (ʻ), in percentage points. `*` = significant after Holm correction (exact McNemar, α = 0.05).

| model | official Latin, % | ASCII ' | new 2026 | only ö ğ | only ş ç | Cyrillic | converted back |
|:--|--:|--:|:--|:--|:--|:--|--:|
| Gemma 3 4B (bf16) | 64.0 | −0.3 | −7.4* | −3.1* | −5.9* | −4.7* | −0.1 |
| Gemma 3 4B (4-bit) | 59.9 | +0.4 | −6.8* | −1.3 | −3.9* | −2.6 | 0.0 |
| Qwen 2.5 7B | 60.2 | −0.6 | −4.7* | −1.2 | −0.4 | −4.7* | 0.0 |
| Llama 3.1 8B | 69.1 | +0.6 | −6.9* | −2.4 | −3.0 | −8.7* | +0.1 |
| Qwen 3 8B | 69.7 | +0.9 | −4.3* | −1.8 | −1.1 | −3.3 | −0.1 |
| Gemma 3 12B | 84.9 | +1.0 | −3.1* | −1.0 | −1.4 | −0.7 | 0.0 |
| Mistral Nemo 12B | 74.7 | +0.7 | −9.6* | −3.8* | −6.9* | −18.6* | 0.0 |

### Tokens per tokenizer

FLORES devtest (1,012 sentences), change in token count against the official Latin spelling (ʻ), in percent. Intervals are in [results/rq1/](results/rq1/).

| tokenizer | ASCII ' | new 2026 | only ö ğ | only ş ç | Cyrillic |
|:--|--:|--:|--:|--:|--:|
| GPT-4o (o200k) | −0.2 | +0.9 | −2.4 | +3.4 | +10.2 |
| GPT-4 (cl100k) | −4.8 | −5.0 | −6.5 | +1.6 | +55.6 |
| Llama 3.x | −4.8 | −7.7 | −7.7 | +0.1 | +24.4 |
| Qwen 2.5 / Qwen 3 | −1.0 | −3.9 | −3.7 | −0.1 | +30.0 |
| Gemma 3 | 0.0 | −2.0 | −4.4 | +2.4 | +10.1 |
| Mistral Nemo | +4.0 | +4.6 | +0.7 | +4.2 | +38.1 |
| DeepSeek V3 | −0.4 | +0.9 | −2.0 | +2.9 | +22.1 |

### Training from scratch

A 98M-parameter GPT-2-style model, 3 seeds, same documents in each spelling ([results/analysis/exp_a.md](results/analysis/exp_a.md)).

| tokenizer setup | tokens, new vs old | bits per word, old | bits per word, new |
|:--|--:|--:|--:|
| separate vocabulary per spelling | +0.27% | 12.11 | 12.01 |
| one shared vocabulary | +0.08% | 11.96 | 12.09 |

The bits-per-word difference changes sign between setups and between seeds.

## Follow-up checks (10–11 November 2026, exploratory)

These five checks were run after the paper and are **not in it**. Each ran once, on tokenizers only or on Gemma 3 4B (bf16) only. They are first looks, not results. Scripts, per-question outputs and tables are in [extra_nov10/](extra_nov10/).

**1. Where tokenizers cut the words** ([seg_analysis.md](extra_nov10/seg_analysis.md)). For every FLORES word the reform changes (5,396 word tokens), we compared the token boundaries in the ASCII-apostrophe spelling and the new alphabet. Words were tokenized one at a time, without sentence context.

| tokenizer | changed words cut at different places | same token count but different cuts | new letter is a one-letter token |
|:--|--:|--:|--:|
| GPT-4o (o200k) | 80% | 55% | 38% |
| GPT-4 (cl100k) | 81% | 65% | 65% |
| Llama 3.x | 78% | 52% | 42% |
| Qwen 2.5 | 80% | 57% | 45% |
| Gemma 3 | 75% | 44% | 41% |
| Mistral Nemo | 82% | 54% | 47% |
| DeepSeek V3 | 86% | 73% | 79% |

Examples (GPT-4o): `bo|'lgan` → `böl|gan`, `chun|ki` → `ç|unki`, `tom|osha` → `tomo|şa`. More in [seg_examples.md](extra_nov10/seg_examples.md).

**2 and 3. Stripped marks and Unicode form** ([extra_eval.md](extra_nov10/extra_eval.md)). Belebele accuracy of Gemma 3 4B when the new-alphabet text has its marks removed (ö→o, ğ→g, ş→s, ç→c) or is decomposed (NFD: letter + combining mark). The released data is NFC.

| spelling | accuracy, % | vs new 2026 (exact McNemar p) | prompt tokens vs new 2026 |
|:--|--:|:--|--:|
| official Latin | 64.0 | +7.4 (p < 0.001) | +1.5% |
| new 2026 (NFC) | 56.6 | | |
| new 2026, decomposed (NFD) | 57.9 | +1.3 (p = 0.35) | +9.7% |
| new 2026, marks removed | 55.9 | −0.7 (p = 0.69) | −4.0% |

The first two rows reproduce the paper's numbers exactly. Removing the marks does not restore accuracy; note that it also removes information (ş and s become one letter). On FLORES, NFD raises token counts by 11–19% on six of seven tokenizers and by 0.0% on Qwen 2.5.

**4. Layers** ([layers.md](extra_nov10/layers.md)). For 500 FLORES sentences we mean-pooled Gemma 3 4B's hidden states at each layer and asked how often a sentence in spelling X is closest (cosine, after centering) to the same sentence in official Latin. Chance is 0.2%.

| layers (0 = embeddings, 34 = last) | ASCII ' | new 2026 | Cyrillic |
|:--|--:|--:|--:|
| 0–4 | 100% | 98–100% | 85–94% |
| 12–15 | 97–98% | 70–73% | 60–64% |
| 23–32 | 100% | 88–91% | 90–94% |

Early layers are helped by the many words that are spelled identically in both versions.

**5. Does the model answer in the new alphabet?** ([fidelity.md](extra_nov10/fidelity.md)). 150 Belebele passages and questions per spelling, chat template, no instruction about spelling, greedy decoding, 48 new tokens. Each answer is labelled by whether it contains new letters (ş ç ö ğ), old forms (sh, ch, oʻ, gʻ), both or neither.

| question written in | new letters only | old spelling only | mixed | neither |
|:--|--:|--:|--:|--:|
| official Latin | 0% | 81% | 0% | 19% |
| new 2026 | 22% | 32% | 29% | 17% |

Many "new letters only" answers copy words from the passage.

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
extra_nov10/                 follow-up checks run after the paper (exploratory)
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

Claude (an AI assistant) wrote the evaluation and analysis code, ran the experiments, did a first review of the converted text samples, contributed to the detailed study design and drafted the paper. The author conceived the study, defined its research questions and conditions, proposed the training comparison, built the YangiAlifbo converter, and hand-checked the converted text samples. See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md). The follow-up checks in `extra_nov10/` were also written and run by Claude at the author's request.

## Data and licenses

| Item | License |
|---|---|
| Code | MIT ([LICENSE](LICENSE)) |
| `data/parallel/` (adapted from FLORES-200 and Belebele @ `7899cdf`) | CC BY-SA 4.0 ([details](data/parallel/LICENSE.md)) |
| Training corpus (not included): FineWeb-2 @ `af9c133` | ODC-By 1.0 |

Gold answers in Belebele are not balanced, so always answering the most common option scores 27.9%, not 25%.
