# Walkthrough: everything done so far, and how to run it yourself

This file explains every step taken in the first session (Oct 29–30, 2026), why each one was done, and the exact commands to repeat it. It is honest about who did what (see §0).

Shell: **PowerShell**. Every command starts from the project folder:

```powershell
cd C:\Users\Bilolbek\Desktop\Research
```

---

## 0. Who did what (important for the paper)

| Part | Author |
|---|---|
| YangiAlifbo engine (lexicon, morphology, foreign filter, CNNs) | **You** |
| Research question, the 6 conditions, choice of RQ1–RQ4 | **You** (brief) |
| Decisions: keep engine output as-is; use Belebele | **You** |
| Literature search, converter CLI, Python pipeline, Cyrillic transliterator, RQ1 script, RQ2 harness and analysis script, first-pass hand-check | **Claude (AI assistant)**, at your request |

What this means in practice:
- Most venues (ACL-family workshops such as SIGTURK and LoResLM) ask authors to **disclose substantial use of AI assistants**; the ACL Responsible NLP checklist has a question on it. Check the current policy of the venue you submit to, and say in the paper (Acknowledgements or a short "Use of AI assistants" section) that code and analysis scripts were written with an AI coding assistant.
- Git commits made with Claude carry a `Co-Authored-By: Claude` line, so the history shows it.
- The best protection is understanding: after this walkthrough you should be able to explain and re-run every step. If you want, rewrite parts yourself (the Cyrillic transliterator in §5.4 is a good C++/Python exercise) and compare outputs.
- The **hand-check must be yours**. You are the native speaker; my pass was only a first look.

---

## 1. Checking your PC

Command I ran (PowerShell, read-only):
```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version; dotnet --list-sdks; git --version
```

Result:
| | |
|---|---|
| CPU | Intel Core Ultra 9 185H, 16 cores / 22 threads |
| RAM | 31.4 GB |
| GPU | NVIDIA RTX 4080 Laptop, **12 GB** VRAM (driver 617.14) + Intel Arc (not used) |
| Disk C: | 246 GB free |
| Tools | Python 3.12.10, .NET SDK 10.0.402, git 2.56 |

What fits on the GPU: a model's weights take about *(parameters × 2 bytes)* in bf16 (16-bit). 4B model ≈ 8.6 GB → fits. 8B ≈ 16 GB → does **not** fit in bf16; needs 4-bit quantization (≈ 5 GB, small accuracy loss) or Kaggle. 12B → Kaggle.

---

## 2. Literature check

Searches (web search tool): "Uzbek new Latin alphabet 2026 LLM tokenization", "arXiv Uzbek apostrophe oʻ gʻ variants language model robustness", "Uzbek orthography script variation large language models benchmark 2026 ACL Anthology", "Uzbek tokenizer fertility apostrophe U+02BB U+2019 normalization LLM accuracy". I then opened the most relevant pages.

Found:
1. **Roʻzimatov, "The Tokenization Premium… Evaluating the 2026 Alphabet Reform"**, Zenodo record 21872526, Aug 10 2026, CC BY 4.0, listed in *BG Pulse Journal USA* 2(08). Six tokenizers (not named on the page), token counts only. Finds the reform gives "modest and tokenizer-dependent" savings. **No accuracy, no apostrophe variants, conversion method not stated, peer review not stated.** → Overlaps with our RQ1; must be cited. Our RQ2–RQ4 and apostrophe conditions are new.
2. github.com/fuzayil-baxtiyoriy/uzbek-tokenizer-analysis: 20 sentences, 4 tokenizers, Latin vs Cyrillic vs English, no 2026 alphabet, no accuracy.
3. arXiv 2608.09356 (Zhang, Aug 2026): script unification for 11 Turkic languages with fastText/CANINE, not LLMs, nothing on the 2026 alphabet.
4. The ones you already knew (TUMLU, UzbekTagger, etc.).

Limits: only web search was used. I did **not** search Google Scholar or the ACL Anthology site directly. Do that yourself before submitting (search "Uzbek" in ACL Anthology, sort by year).

### 2.1 Second, deeper check (Oct 30, 2026)
Where I searched: arXiv (all papers mentioning "Uzbek", newest first), OpenAlex (the open index of papers, Zenodo, journals; query "Uzbek tokenization", newest 50), GitHub (repos for "uzbek alphabet", "uzbek tokenizer", "uzbek orthography", "yangi alifbo", "uzbek apostrophe"), Hugging Face datasets ("uzbek", newest 100), ACL Anthology (via site search), and the web in English, Uzbek and Russian. Semantic Scholar refused (rate limit). Google Scholar can't be searched by a tool, so **you must check it yourself** (see below).

**New, and the closest work found:**
- **Ibadullaev, Hayitova, Gafurov, Salmanova, Izrailbekova, "Uzbek orthography reform and subword tokenization: code and results"**, Zenodo DOI 10.5281/zenodo.23665432, **Oct 26, 2026**, CC BY 4.0. It is the code and data for a paper **submitted to PeerJ Computer Science**, titled "Orthography or tokenizer?", about apostrophes, digraphs and diacritics in Uzbek subword tokenization after the 2026 reform. From its file list: an 80,000-sentence corpus (CC-100 + Wikipedia); **subword tokenizers trained by them** at several vocabulary sizes and random seeds (`fig_fertility_vs_vocab`, `tok_by_seed`); **character-level language models** (`char_lm_*`, bits per word); equivalence tests (TOST); round-trip conversion mismatches. **Not visible:** named commercial tokenizers, LLM task accuracy, Cyrillic. I only saw the file list. Reading its README needs the 720 kB zip downloaded.
  → This overlaps with our **RQ1 apostrophe + new-alphabet** part, in a different setting: tokenizers they trained themselves vs. the tokenizers of real production LLMs in ours. It does **not** seem to cover LLM accuracy (RQ2), API cost (RQ3) or the fix (RQ4). **Cite it**, and say clearly how ours differs.

  **Read in full (Oct 30).** I downloaded the code zip (MD5 `f677aba7…` matches Zenodo) into the scratchpad and read it without running anything:
  - **Tokenizers:** only ones they **trained themselves** on 80,000 sentences (CC-100 + Wikipedia), vocab 1k–16k, 5 seeds: SentencePiece BPE/Unigram (default, and with `split_by_unicode_script=False`) and byte-level BPE (with the GPT-2 regex, and without). **No GPT/Llama/Qwen/Gemma tokenizer, no LLM, no accuracy, no Belebele, no Cyrillic condition** (Cyrillic sentences are filtered out of the corpus).
  - **Apostrophes:** every variant is first normalized to ASCII `'` (`"apostrophe": "ascii"` in the run). So **they do not compare ʻ vs ' vs ` vs ’**, which is our conditions 1–4.
  - **Converter:** plain regex (o'→ö, g'→ğ, every sh→ş, every ch→ç, except an optional word list). No lexicon, morphology or foreign-word filter, so "show" → "şow". Round-trip old→new→old matches 99.19% of sentences.
  - **Main finding:** with default settings, the new spelling lowers fertility by **4.6% to 11.1%** (BPE, vocab 1k→16k), because default SentencePiece and the GPT-2 regex **always split the apostrophe off the letter** (oʻ/gʻ split 100% of the time). With script splitting turned off (`bpe_noscript`), the difference is **~0%** (e.g. 2.9732 vs 2.9732 at 1k). Their conclusion: the gain comes from the **tokenizer's settings, not the orthography**, as their title "Orthography or tokenizer?" says. Character LMs: new spelling −1% to −2.7% bits per word, +1.2% to +3% bits per character.
  - Their toolkit also mentions OCR and speech-recognition experiments, but **no OCR/ASR results** are in the release.
  - **For us:** their result explains ours. Tokenizers whose pre-tokenizer splits `'` (cl100k, Llama 3) gain from the new alphabet; Gemma 3, which doesn't split it, barely changes (−2%). Cite them for this mechanism. Our contribution stays distinct: **real production tokenizers, the four apostrophe characters compared, Cyrillic, a lexicon-aware converter, and above all LLM accuracy (RQ2) and the fix (RQ4).**
- Roʻzimatov (Zenodo, Aug 2026). Already in §2: token counts only.

**Checked, not overlapping:**
- Uzakova, "The problem of multiple writing systems and text normalization in MT of Uzbek and Karakalpak" (in Russian), *Buxoro Xalqaro Universiteti Ilmiy Axborotnomasi*, Oct 25, 2026. A discussion of how mixed scripts and apostrophes hurt machine translation. No 2026 alphabet, no LLM measurements.
- Dobrovolskyi, "Beyond Two Bytes per Letter" (arXiv 2608.21384): Cyrillic tokenization overhead, mainly Ukrainian. Uzbek not named.
- `alifbe` (PyPI, Sep 2026): a rule-based old→new Latin converter and normalizer (placeholder author). A tool, not a study. It can serve as a **baseline converter** to compare YangiAlifbo against.
- `uzlm/alloma-8B-Instruct` (Hugging Face): an Uzbek model that replaces apostrophes with a special token to cut fertility (1.7 vs ~3.5 tokens/word for Llama). No comparison between spellings.
- fuzayil-baxtiyoriy/uzbek-tokenizer-analysis (updated Oct 29): still no 2026 alphabet, no accuracy.
- ~20 new GitHub repos (keyboards, Telegram bots, converters for the new alphabet). None measure LLMs.

**Also learned:** the President **signed the alphabet law on Oct 21, 2026** ([gazeta.uz](https://www.gazeta.uz/ru/2026/10/21/uzbek-alphabet/)). Use this date in the paper. Debate in Uzbek media (Anhor.uz, 23 Oct): Maqsud Asadov (director, Institute of Uzbek Language, Literature and Folklore) says two-character letters cause technical obstacles; philologist Komil Jalilov says it makes no difference to AI. **Correction:** an earlier version of this file reversed Asadov's position, based on a search summary. Our study measures this disagreement directly.

**Conclusion.** Nobody can guarantee "100%": papers under review, local journals and Telegram posts are not all indexed. What the search supports:
- **RQ1 is no longer new by itself** (Roʻzimatov; Ibadullaev et al.).
- **Still unclaimed:**
  - LLM **accuracy** under the 2026 alphabet and the apostrophe variants (RQ2);
  - the **normalization fix** measured on accuracy (RQ4);
  - cost on real production tokenizers (RQ3);
  - a lexicon-aware converter with a measured error rate;
  - a released 6-condition parallel benchmark.
- **Lead the paper with RQ2/RQ4**, and present RQ1 as confirming and extending prior token-count work.

**▶ Your check (10 min):** on [scholar.google.com](https://scholar.google.com), search each of these, with "Since 2026" selected: `Uzbek alphabet language model`, `Uzbek apostrophe tokenization`, `"Uzbek" "Belebele" script`, `oʻzbek alifbo til modeli`. Note any new hit here.

---

## 3. Project layout

```
Research/
  README.md                 research question, conditions, how to reproduce
  PLAN.md                   4-week plan
  docs/WALKTHROUGH.md       this file
  .gitignore                keeps .venv, external/, data/raw/, build output out of git
  requirements.lock.txt     exact versions of every Python package
  converter/YangiAlifbo.Cli/   C# command-line wrapper around your engine
  external/yangicha-alifbo/    a copy of your repo (cloned, not modified)
  src/conditions.py         builds the 6 spellings
  scripts/
    build_parallel.py       text file → 6 conditions (used for FLORES)
    build_belebele.py       Belebele questions → 6 conditions
    rq1_fertility.py        token counting (RQ1)
    rq2_eval.py             runs a model on Belebele (RQ2)
    rq2_analyze.py          accuracy, confidence intervals, significance (RQ2)
  data/raw/                 downloaded datasets, untouched
  data/parallel/<name>/     our 6-condition versions
  results/rq1/, results/rq2/, results/logs/
```

Nothing is in git yet (`git init` has not been run).

---

## 4. The converter CLI (C#)

### 4.1 Why a CLI was needed
Your repo has no mode that converts a file. The engine lives in `src/YangiAlifbo.Core` (a library). I wrote a small console program that references that library. **Your repo was not changed**; the CLI is in this project.

Your repo was cloned at commit `fdaafae2249fbde3505a9166ff207adfc5cd6b5d` (Oct 27 2026). That commit is the version all results use.

### 4.2 What I read in your engine
- `WordConverter.Convert(detector, word, mode, slang)` → returns the converted word, or `null` if it should stay as is. Steps: (1) `IsForeignOrReject` → keep; (2) `HasConvertiblePatterns` (sh/ch/oʻ/gʻ present?) → if not, keep; (3) in Strict mode, `IsUzbekOrLikely` (lexicon + morphology + CNN fallback) → if yes, `UzbekPhonetics.Convert`.
- `UzbekPhonetics.Convert`: first turns 9 apostrophe look-alikes into ASCII `'`, protects `s'h`/`c'h` (Is'hoq), then sh→ş, ch→ç, o'→ö, g'→ğ, keeping case.
- `LanguageDetector()` (default constructor) loads the embedded dictionaries **and** your personal whitelist/blacklist from `%AppData%`.

### 4.3 Problems found in the engine (worth fixing in your repo)
1. **Hyphenated words are not converted.** `boʻlsa-da`, `video-oʻyin`, `kecha-yu` are looked up as one word, which is not in the lexicon. In FLORES, `boʻlsa-da` alone appeared 11 times.
2. **`TextConverter` skips words that start with punctuation**, e.g. `“Shahar”` or `«Shahar»`, because it checks only if the first character is a letter.
3. **The default detector reads personal lists from AppData**, so two computers could give different output.
4. **Misses**: Uzbek-spelled foreign names (Polsha, Vashington, Bishkek, Charlz) and some common words (chop, tinchlikparvar, chakralar, kapuchino, afrikancha, mashhurligi).
5. **Apostrophes inside converted words become ASCII `'`** (the tutuq in a converted word loses its official form).

### 4.4 What the CLI does (`converter/YangiAlifbo.Cli/Program.cs`)
1. Refuses to run (exit code 3) if your AppData whitelist/blacklist has any entries → output depends only on the engine (fix for problem 3).
2. Reads the input file line by line (UTF-8), writes one output line per input line.
3. Splits each line on whitespace. For each token:
   - peels punctuation off the front (`“ « ( ‘`) and the back (`. , ” ! ?`), **but** keeps a final apostrophe after o/g (`tog‘`, `bo‘`) because there it is part of the letter (fix for problem 2);
   - splits the rest on `-`, `"` and `/` and converts each part separately (fix for problem 1: `boʻlsa-da` → `bölsa-da`);
   - for each part with a convertible pattern, calls `WordConverter.Convert(..., Strict, slang: false)`.
4. With `--words file.tsv`, writes every word it considered: line number, word, result, 1/0 changed. This is what the hand-check uses.
5. Prints a summary to the terminal: `lines=… convertible_words=… converted=… kept=…`.

### ▶ Try it yourself
```powershell
dotnet build -c Release converter/YangiAlifbo.Cli
"Toshkent shahridagi o‘quvchilar “Shaxmat” to‘garagiga borishdi.`nU show va push haqida gapirdi, bo'lsa-da." | Set-Content -Encoding utf8 test_in.txt
dotnet converter\YangiAlifbo.Cli\bin\Release\net10.0\YangiAlifbo.Cli.dll test_in.txt test_out.txt --words test_words.tsv
Get-Content -Encoding utf8 test_out.txt
Get-Content -Encoding utf8 test_words.tsv
```
Expected: `Toşkent şahridagi öquvçilar “Şaxmat” tögaragiga borişdi.` and `show`, `push` unchanged. Delete the three `test_*` files afterwards.

---

## 5. Building the 6 conditions (`src/conditions.py`)

### 5.1 Step 1: canonicalize (one official form)
Real text mixes `'`, `ʻ`, `’`, `` ` `` etc. First every apostrophe-like character (9 of them) is mapped by context:
- right after `o`, `O`, `g`, `G` → **ʻ U+02BB** (the oʻ/gʻ mark);
- anywhere else → **ʼ U+02BC** (tutuq belgisi, as in `maʼno`).

Bug I made and fixed: I first blocked ʻ when two apostrophes came close together, which broke `Erdoʻgʻan` (became `Erdoʻgʼan`). Now the rule is only "after o/g".

Known edge case: an English possessive like `Twitter'da` becomes `Twitterʼda` (tutuq). That is fine for our purpose.

### 5.2 Conditions 1–4: apostrophe variants
From the canonical text: `latin_okina` = canonical as is. `latin_ascii`, `latin_backtick`, `latin_rsquo` = **both** ʻ and ʼ replaced by `'`, `` ` ``, `’`.
Why both: a person typing on a keyboard uses the same key for oʻ and for the tutuq. This mirrors real typing. (Alternative design: change only the oʻ/gʻ mark. Mention this choice in the paper.)

### 5.3 Condition 5: new 2026 alphabet
The canonical text goes through the CLI (§4). Afterwards, any ASCII `'` left (the engine's own output inside converted words) becomes ʼ tutuq. "ng" is unchanged, because the reform only stops counting it as one letter; spelling stays the same. **Per your decision, no manual fixes**: misses count as errors.

### 5.4 Condition 6: Cyrillic (rule-based, written by me)
Two-letter rules first: oʻ→ў, gʻ→ғ, sh→ш, ch→ч, yo→ё, yu→ю, ya→я, ye→е.
Single letters: a→а b→б d→д f→ф g→г h→ҳ i→и j→ж k→к l→л m→м n→н o→о p→п q→қ r→р s→с t→т u→у v→в x→х y→й z→з, tutuq ʼ→ъ.
Special rules:
- `e` → **э** at the start of a word or after a vowel (`ekran`→экран, `aeroport`→аэропорт), otherwise **е**.
- `sʼh` → сҳ (the tutuq disappears: Isʼhoq → Исҳоқ).
- `ts` before `iy`/`io` → **ц** (operatsiya → операция, -tsion → -цион). Fix added after the hand-check.
- `yoʻ` → **йў**, not ё+ʻ (`yoʻl` → йўл). Bug found in the hand-check (`ёʻл`), fixed.
- Words containing `w` or a `c` not followed by `h` (USOC, Twitter, WiFi) stay **in Latin**, as they usually do in Cyrillic Uzbek text. Without this rule they became mixed, e.g. `УСОC`.
- Upper/lower case is preserved (`SH`→Ш, `Sh`→Ш, `Yangi`→Янги).

Errors that **cannot** be fixed by rules, because the 1995 Latin spelling lost the information: ь (`sentyabr` should be сентябрь, `italyan` итальян), ц written as s (`provinsiya` провинция, `sanksiya` санкция, `aksiz` акциз). So our Cyrillic is **transliterated, not native**. Say so in the paper's limitations.

### 5.5 `scripts/build_parallel.py` (for a plain text file)
Input: one sentence per line. Output in `data/parallel/<name>/`:
- `latin_okina.txt … cyrillic.txt`: same line count, line *i* = same sentence;
- `parallel.jsonl`: all 6 versions + original per sentence;
- `words_new2026.tsv`: engine decisions (from `--words`);
- `handcheck_sample.tsv`: 100 random sentences (`random.Random(42)` → always the same 100), with empty columns for your error notes;
- `build.log`: time, source, seed, engine summary, the 100 ids.
If `handcheck_sample.tsv` already exists, it is **not overwritten** (a `.new.tsv` is written instead), so your annotations are safe.

---

## 6. Data

### 6.1 FLORES-200 (for RQ1)
- What: 1,012 English Wikipedia/news sentences professionally translated into 200 languages (Meta, NLLB project). We use `uzn_Latn` devtest.
- Download: `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (25.6 MB), sha256 `b8b0b767…44011f6`. I extracted only `uzn_Latn` and `eng_Latn` (dev + devtest) and the README into `data/raw/flores200/`.
- License: CC BY-SA 4.0.
- Spelling found: ASCII `'` 2,358 times, ʻ only 8, ʼ 1. So FLORES is "keyboard" Uzbek, and our official-ʻ condition is **generated**. 997 of 1,012 lines contain sh/ch/oʻ/gʻ.

### 6.2 Belebele (for RQ2)
- What: 900 multiple-choice reading questions (Meta, 2023), each with a FLORES passage, a question and 4 options, one correct. Passages are FLORES sentences, so **RQ1 and RQ2 texts overlap**.
- Source: Hugging Face `facebook/belebele`, file `data/uzn_Latn.jsonl`, pinned at commit `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, sha256 `4dfc231d…6519202`. Not gated. CC BY-SA 4.0 → we may release our converted version under the same license.
- Fields: `link, question_number, flores_passage, question, mc_answer1–4, correct_answer_num, dialect, ds`.
- 488 distinct passages; 4,788 distinct strings to convert. Spelling: ASCII `'` 10,221, ʻ 28, ʼ 4.
- Correct-answer counts: 1→206, 2→251, 3→246, 4→197. **Always answering "2" scores 27.9%**, which is the real "dumb baseline", not 25%.
- 2 questions already have two identical options in the original data (they stay as they are).
- `scripts/build_belebele.py` converts each distinct string once, puts the converted text back into the same rows (answer keys untouched), writes `data/parallel/belebele_uzn/<condition>.jsonl`, and a 30-question hand-check sample (each question + 4 options as rows).
- Engine stats: 14,987 convertible words, 14,719 converted, 268 kept.

---

## 7. Hand-check (my first pass; yours is the real one)

FLORES sample of 100 sentences, new2026 column:
- 523 convertible words in the sample.
- Wrongly converted foreign words: **0** found (e.g. "show", "Bush", "Richard" correctly kept).
- Missed Uzbek words: **6** (Polshalik, Shnayderning, Machu, Pikchuning, afrikanchadan, kapuchino) → **1.1%** of convertible words; **5 of 100 sentences** have ≥1 error.
- Not counted (errors already in FLORES itself): `b]oʻlmagan`, `oʻFlinning`. `Bellingshauzen` correctly kept (it is s+h, not sh).

Cyrillic: I only scanned words matching suspicious patterns (тс, э inside a word, ъ, ё/ю/я, Latin letters). That scan found the two bugs fixed in §5.4. Remaining errors are the ь/ц type. **No error rate yet. That needs your full check.**

### ▶ Your task
Open `data\parallel\flores_devtest\handcheck_sample.tsv` in Excel (Data → From Text/CSV → UTF-8, tab-separated) or VS Code. For each row, write the wrong words in `new2026_errors` and `cyrillic_errors` (e.g. `Polshalik→Polşalik`), leave empty if perfect. Then do the same for `data\parallel\belebele_uzn\handcheck_sample.tsv`. Save as UTF-8, tab-separated.

---

## 8. Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install transformers==4.57.1 tokenizers==0.22.1 tiktoken==0.12.0 sentencepiece==0.2.1 protobuf==6.33.0 huggingface_hub==0.36.0 datasets==4.3.0 pandas==2.3.3 numpy==2.3.4 matplotlib==3.10.7 tabulate==0.9.0
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install accelerate==1.11.0 scipy==1.16.3
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```
Already done; you do not need to repeat it. Notes:
- A **venv** is a private Python folder for this project, so packages don't clash with other projects.
- **Pinning** (`==version`) means anyone gets the same versions later = reproducible.
- `torch …+cu128` is the GPU build (CUDA 12.8). On a new machine, install torch with the `--index-url` line above.
- `-X utf8` (used in every command below) makes Python print Uzbek/Cyrillic characters correctly in the Windows terminal.

---

## 9. RQ1: tokenizer fertility (`scripts/rq1_fertility.py`)

### 9.1 Concepts
- **Token**: the unit a model reads. A tokenizer splits text into pieces from a fixed vocabulary (e.g. `toʻgʻri` → `to`, `ʻ`, `g`, `ʻ`, `ri`). Pieces cost money (APIs charge per token) and take space in the context window.
- **Fertility** = tokens per word. Lower = more efficient. English is usually ~1.3; Uzbek ~2.6–3.4 here.
- **Tokens per character**: same idea per character. Note that ş is 1 character while sh is 2.

### 9.2 Tokenizers used
| Name in table | Source | Note |
|---|---|---|
| GPT-4o (o200k) | `tiktoken` `o200k_base` | also used by GPT-4.1/GPT-5 family |
| GPT-4 (cl100k) | `tiktoken` `cl100k_base` | |
| Llama 3.x | `unsloth/Llama-3.2-1B-Instruct` | mirror; Meta's repo is gated |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct` | |
| Qwen 3 | `Qwen/Qwen3-8B` | turned out identical to Qwen 2.5 |
| Gemma 3 | `unsloth/gemma-3-4b-it` | mirror; Google's repo is gated |
| Mistral Nemo | `unsloth/Mistral-Nemo-Instruct-2407` | mirror |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` | |
| Phi-4 | `microsoft/phi-4` | turned out identical to cl100k |
| UTF-8 bytes | — | baseline: raw bytes (ʻ ş ö are 2 bytes, ' is 1) |
| Aya Expanse | `CohereLabs/aya-expanse-8b` | **skipped: gated**; needs your HF token |

Only the tokenizer files are downloaded (a few MB each), not the models. For each Hugging Face tokenizer, the exact repo commit and the sha256 of `tokenizer.json` are saved in the log. That makes it reproducible, and lets you prove a mirror equals the original once you have access. Special tokens (like `<bos>`) are **not** counted (`add_special_tokens=False`).

### 9.3 Metrics and statistics
- Words are counted once, from `latin_okina` (whitespace split), and used for all conditions. This is valid because conversion never adds or removes spaces.
- `ratio_vs_okina` = total tokens in a condition ÷ total tokens in `latin_okina`. Shown as % change.
- **Bootstrap confidence interval** (explained simply): we have 1,012 sentences. Draw 1,012 sentences *with replacement* (some twice, some not at all), recompute the ratio, and repeat 1,000 times (random seed 42). The middle 95% of those 1,000 ratios is the 95% CI. It answers: "if we had picked a different set of similar sentences, how much would the number move?" These CIs are narrow because the same sentences are compared across conditions (paired).

### 9.4 Results (FLORES devtest)
Tokens per word:
| tokenizer | okina ʻ | ascii ' | backtick | ’ | new2026 | cyrillic |
|---|---|---|---|---|---|---|
| GPT-4o | 2.626 | 2.620 | 2.695 | 2.647 | 2.650 | 2.894 |
| GPT-4 / Phi-4 | 3.398 | 3.236 | 3.263 | 3.250 | 3.228 | 5.286 |
| Llama 3.x | 3.324 | 3.163 | 3.189 | 3.167 | 3.068 | 4.135 |
| Qwen 2.5 / 3 | 3.287 | 3.255 | 3.282 | 3.269 | 3.160 | 4.274 |
| Gemma 3 | 2.840 | 2.839 | 2.839 | 2.839 | 2.784 | 3.125 |
| Mistral Nemo | 2.672 | 2.780 | 2.825 | 2.810 | 2.795 | 3.689 |
| DeepSeek V3 | 3.175 | 3.162 | 3.199 | 3.201 | 3.202 | 3.876 |

What it says:
1. **The new alphabet saves little**: vs official ʻ, −2% to −8% on Gemma/Qwen/GPT-4/Llama, but **+0.9%** on GPT-4o and DeepSeek and **+4.6%** on Mistral Nemo. Vs ASCII `'` (how most people really type), the saving is at most ~3%.
2. **Official ʻ is often the most expensive Latin form** (GPT-4, Llama: ~5% more than ASCII). A *hypothesis* (not tested yet): tokenizers learned from web text where `'` is common and ʻ is rare, so ʻ gets split off as its own token.
3. **Mistral Nemo is the exception**: ʻ is its cheapest form.
4. **Cyrillic costs +10% to +56%.**
5. Gemma 3 does not care which apostrophe is used (0.0%).

Files: `results/rq1/flores_devtest_tokens_per_word.md`, `…_pct_vs_okina.md` (with CIs), `…_fertility.csv` (everything), log in `results/logs/rq1_flores_devtest_*.json`.

---

## 10. RQ2: accuracy harness

### 10.1 `scripts/rq2_eval.py`: how a question is asked
Prompt (English frame, identical in all conditions; only the Uzbek text inside changes):
```
Read the passage and answer the question with the letter of the correct option.

Passage: {passage}

Question: {question}
A. {option 1}
B. {option 2}
C. {option 3}
D. {option 4}

Answer:
```
Why the frame is in English: if it were Uzbek, it would also have to be converted, and the frame itself would differ between conditions. English keeps it byte-identical (its sha256 is logged).

**Log-likelihood scoring** (explained simply): we don't let the model write text. We run the prompt once and look at the probabilities the model gives to the next token being " A", " B", " C" or " D". The highest one is its answer. Advantages: no randomness, no parsing of free text, fast (one forward pass per question). Each probability is saved, so we can analyze confidence later.

Other details:
- Model loaded in **bf16** on the GPU, or 4-bit with `--load-4bit` (for 7–12B models).
- **One question at a time, no padding**: batching would add padding, which can change numbers slightly.
- TF32 math off and seed 42 for determinism.
- The model's exact Hugging Face commit is fetched and used, then logged.
- Output: `results/rq2/<model>/<condition>.jsonl`, one line per question: `gold`, `pred`, the 4 log-probabilities, prompt length in tokens. Log: `results/logs/rq2_<model>_<time>.json` (versions, GPU, accuracy, seconds per condition).
- `--limit N`: first N questions only (for testing).

### 10.2 `scripts/rq2_analyze.py`: statistics
- Accuracy per condition with a 95% CI.
- **Paired difference** vs `latin_okina`, with a 95% CI.
- **Cluster bootstrap**: several questions come from the same article, so they are not independent. Instead of resampling questions, we resample *articles* (`link` field) and take all their questions. This gives honest (wider) intervals. The same resamples are used for every condition (paired).
- **McNemar test** (explained simply): look only at questions where the two spellings disagree, i.e. one right and one wrong. If spelling didn't matter, "gained" and "lost" would be about 50/50. The p-value says how surprising the actual split is. Small p (< 0.05) = unlikely to be chance.
- Writes `summary.md` and `summary.json` in the model's results folder.

### 10.3 What has been run
- Smoke test: Gemma 3 4B (`unsloth/gemma-3-4b-it`, a mirror of Google's gated repo), 20 questions, 2 conditions. Both got 65%, which is meaningless with 20 questions, but it proved the pipeline works. ~0.3 s per question. The letters A–D get almost all probability (e.g. −0.06 log-prob ≈ 94% for the chosen letter).
- Full pilot: finished 5 conditions × 900. **Cyrillic stopped at 848/900** when the session ended. **I have not looked at any accuracy numbers from it.** You will see them first, below.

---

## 11. ▶ Your commands: finish the pilot (run these in order)

```powershell
cd C:\Users\Bilolbek\Desktop\Research
```

**Step 1: re-run the unfinished Cyrillic condition** (~15 s model load from cache + ~5 min). You will see the accuracy line printed at the end:
```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\rq2_eval.py --model unsloth/gemma-3-4b-it --conditions cyrillic
```

**Step 2: check all 6 files have 900 lines:**
```powershell
Get-ChildItem results\rq2\unsloth__gemma-3-4b-it\*.jsonl | ForEach-Object { "$($_.Name) $((Get-Content $_).Count)" }
```

**Step 3: compute accuracy, CIs and McNemar:**
```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\rq2_analyze.py results\rq2\unsloth__gemma-3-4b-it
```
How to read the table: `accuracy [95% CI]`; `Δ vs okina` = difference in percentage points (if the CI contains 0, there is no clear difference); `gained/lost` = questions that flipped right/wrong; `McNemar p`.

**Step 4 (optional, to check determinism):** run 20 questions again and compare with the first 20 lines of the full run. The predictions should be identical:
```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\rq2_eval.py --model unsloth/gemma-3-4b-it --conditions latin_ascii --limit 20
```
⚠ This overwrites `latin_ascii.jsonl` with 20 rows. Re-run it without `--limit` afterwards (~5 min), or skip this step for now.

### Optional: rebuild everything from scratch to see it work
```powershell
dotnet build -c Release converter/YangiAlifbo.Cli
.\.venv\Scripts\python.exe -X utf8 scripts\build_parallel.py data\raw\flores200\flores200_dataset\devtest\uzn_Latn.devtest flores_devtest
.\.venv\Scripts\python.exe -X utf8 scripts\build_belebele.py data\raw\belebele\uzn_Latn.jsonl belebele_uzn
.\.venv\Scripts\python.exe -X utf8 scripts\rq1_fertility.py flores_devtest
```
Expected engine summaries: FLORES `convertible_words=5288 converted=5183 kept=105`; Belebele `14987 / 14719 / 268`. The RQ1 tables should match §9.4 exactly. Hand-check samples are not overwritten (`.new.tsv` files appear; ignore or delete them).

---

## 12a. Session 2 (Oct 30): what was added

**Watch everything live** (one window, leave it open):
```powershell
cd C:\Users\Bilolbek\Desktop\Research; Get-Content results\logs\progress.log -Wait -Tail 30
```
`src/progress.py` gives every long script one shared log. Lines are tagged `[rq1]`, `[rq2]`, `[rq3]`, `[download]`, `[queue]`, `[ablations]`. RQ2 prints a line every 100 questions and a `DONE acc=…` line per condition.

1. **Gemma 3 4B pilot (bf16)**, analyzed with `rq2_analyze.py`:

   | condition | accuracy [95% CI] | Δ vs okina (pts) | McNemar p |
   |---|---|---|---|
   | latin_okina | 64.0 [60.9, 67.3] | — | — |
   | latin_ascii | 63.7 | −0.3 [−1.8, +1.0] | 0.74 |
   | latin_backtick | 63.7 | −0.3 [−1.9, +1.1] | 0.76 |
   | latin_rsquo | 63.8 | −0.2 [−1.7, +1.3] | 0.88 |
   | **new2026** | **56.6** | **−7.4 [−10.3, −4.5]** | **<0.001** |
   | cyrillic | 59.3 | −4.7 [−7.7, −1.4] | 0.003 |

   Checks done before believing it: no combining characters, all text is NFC; the model still puts ~94–96% of probability on A–D in every condition (it follows the format); prompts are about the same length. So the drop is not a bug. It is one model so far; more models are running.
2. **Model queue** (`scripts/run_rq2_queue.py`), running on your GPU: Gemma 3 4B again in 4-bit (to see if quantization changes the gaps), Qwen 2.5 7B, Llama 3.1 8B, Qwen 3 8B, Gemma 3 12B, Mistral Nemo 12B. All in 4-bit NF4, because 7B+ in bf16 doesn't fit in 12 GB. Each runs in its own process, and the next model downloads while the current one runs. Console output per model: `results/logs/<model>.console.txt`. The background downloader fails on the already-cached Gemma with a Windows symlink-permission error. That's harmless, because each model also downloads itself when it starts.
3. **RQ1 replicated on FLORES dev** (997 other sentences): every number within ~0.3 pts of devtest.
4. **Letter ablation** (original idea): two extra conditions, `new_og` (only ö ğ, sh/ch kept) and `new_shch` (only ş ç, oʻ/gʻ kept), built by `scripts/build_ablations.py` from the engine output (so the engine's word decisions are kept). Checked: keeping no new letters reproduces `latin_okina` exactly, keeping all reproduces `new2026` exactly (1012/1012 lines). **Result:** ö/ğ *save* tokens (−2.0% to −7.7%), but ş/ç *cost* tokens on most tokenizers (up to +4.2%), because "sh"/"ch" are frequent English letter pairs that tokenizers already merge. The two halves of the reform pull in opposite directions.
5. **RQ3 cost** (`scripts/rq3_cost.py`): context capacity per tokenizer, % change vs ASCII, and $ per 1M Uzbek words for the OpenAI models that tiktoken maps to o200k (gpt-4o, gpt-4.1, gpt-5, gpt-5-mini, o3), using prices read on 2026-10-30 from developers.openai.com/api/docs/pricing. Example: on GPT-5, 1M Uzbek words cost $3.28 in ASCII Latin and $3.31 in the new alphabet. **Not priced:** Claude and Gemini (tokenizers not public; counting needs API keys), DeepSeek (its API now serves V4; we measured the V3 tokenizer).
6. **Figure:** `results/rq1/flores_devtest_heatmap.png/.pdf` (`scripts/plot_rq1.py`).
2b. **Gemma 3 4B in 4-bit** (first model of the queue): okina 59.9%, ascii +0.4, backtick −0.2, rsquo +0.3, **new2026 −6.8 [−9.7, −4.0], p < 0.001**, cyrillic −2.6 [−5.5, +0.3], p = 0.11. 4-bit lowers overall accuracy by ~4 pts but **the new-alphabet gap survives quantization**.
   The first queue's parent process died at 09:47 (its shell exited). The queue is now **resumable**: it skips finished models, and `--skip-done` skips finished conditions. A second copy that I didn't start was running from 10:04:35, so I stopped my own copy to avoid two processes writing the same files. The running copy does the 6 main conditions; the ablations (`new_og`, `new_shch`) run afterwards with `run_rq2_queue.py`, which now includes them.
7. **Exploratory dose–response (Gemma 3 4B bf16):** questions split into thirds by new letters per word (0.22 / 0.32 / 0.43). The accuracy drop (okina → new2026) is +5.0, +6.0, +11.3 pts. More new letters, bigger drop. One model, no CI yet; to be re-run on all models with a proper test (logistic regression) before it goes in the paper.

## 12b. Old vs new spelling for TRAINING (started Oct 30)

Question: if Uzbek models are trained or adapted on new-alphabet text instead of old-spelling text, what changes in time, cost, energy and quality?

**Data** (`scripts/build_train_corpus.py`):
- Source: FineWeb-2 `uzn_Latn` (HF `HuggingFaceFW/fineweb-2` @ `af9c133`, ODC-By, 1.23M web documents). Shuffled with seed 42; the first ~400M characters are kept.
- Every document containing one of the 2,009 FLORES dev/devtest sentences is dropped, so Belebele/FLORES never appear in training.
- `data/train/old/` holds the canonical 1995 Latin. `data/train/new/` holds the same lines converted by YangiAlifbo, line-aligned with the old ones.
- Uzbek Wikipedia was not used: much of it is bot-written stubs.

**Experiment B: adapting an existing model** (`scripts/train_lora.py`):
- Gemma 3 4B (4-bit) is the model with the biggest drop. LoRA rank 16 on the language model's attention and MLP layers, lr 2e-4, 1024-token sequences, 16 sequences per step.
- Trained on the same 30M characters of documents, once in old spelling and once in new.
- Logged: tokens, time, tokens/s, chars/s, peak GPU memory, GPU energy (nvidia-smi power every 2 s × time), and validation **bits per word**. The word count is identical in both spellings, so this is the fair quality measure; bits per own character is also reported.
- Then both adapted models are re-tested on Belebele in all 8 spellings (`rq2_eval.py --adapter`).

**Experiment A: from scratch** (`scripts/train_scratch.py`):
- A GPT-2-style model (12 layers, 768 dim, 16k vocab, ~100M params), one per spelling, in **two tokenizer setups** (decided by Bilolbek, Oct 30):
  - **separate**: each spelling trains its own byte-level BPE (16k) on its own text. This is the realistic case and answers "is a new-alphabet model better overall?"
  - **shared**: one BPE (16k) trained on old+new text together, used by both models. This is the controlled case: the vocabulary is identical and only the spelling differs.
  - 4 models in total: `results/train/scratch/{separate,shared}_{old,new}/`.
- Same documents, same hyperparameters, same seed, 1 epoch.
- Logged: tokens needed, training time, energy, memory, and validation bits per word every 200 steps (a learning curve).

**Runner:** `scripts/run_training_queue.py`. **Not started: waiting for Bilolbek's go.** When started, it waits until Mistral is done, the corpus is built and the GPU is free, then runs B old → B new → evaluate both → A old → A new. Finished steps are skipped on restart. Progress goes to `progress.log` with tags `[corpus]`, `[train-B]`, `[train-A]`, `[queue-train]`.

**Cost:** measured as GPU-hours and Wh on your laptop. No cloud-price conversion yet (prices not verified).

## 12c. The final run and what it showed (Nov 1–2)

**What ran:** `scripts/run_all_experiments.py`: the Llama re-run, back-converted and mixed-apostrophe conditions, chat and Uzbek prompts, 6 LoRA runs, 12 from-scratch runs. It replaced `run_training_queue.py`. `scripts/analyze_all.py` produces everything in `results/analysis/`; every number below comes from there.

**Validity checks:**
- All 106 result files are complete and in the same question order.
- Code and data hashes stayed the same across all restarts.
- A repeated run gave 100/100 identical answers.
- Llama was re-run after its corrupted weight file was repaired. Its gap changed from −7.8 to −6.9 and its base accuracy from 67.7 to 69.1. About 10% of individual answers changed, so discarding the old run was the right call.
- No training run had a heat slowdown. Before the fan change, 7.5% of readings did, but no training had started by then.

**Corrections to things said earlier:**
- "94–96% of probability on the letters" was only true for some models. The real range is 61–94% by model on the raw prompt, which is why the chat-prompt test matters (95–100%).
- "Back-conversion is exact" was wrong: 99.69% of fields match.
- Asadov's position had been reversed; see §2.
- The early LoRA "10% faster" on new spelling did not hold up. Speed changed with the machine's state, and the one pair with equal speed differs by 1.5%, the same as the tokens.

**What each result means, simply:**
- **Accuracy:** models lose 3–10 points on new-alphabet text. It held under three different prompts, with two non-significant exceptions.
- **Dose-response:** the more new letters a question has, the bigger the loss. This is exploratory: we chose it after seeing the data.
- **LoRA (a small add-on trained on a frozen model):** training on 4M characters of new text closes Gemma 4B's gap from −6.8 to +0.3. That suggests the loss comes from models not having seen this spelling, not from the spelling being worse. It's one model, and the training text came from the same converter as the test text, so the model may partly be learning the converter's habits.
- **From scratch:** tokens, time, energy and quality show no consistent difference. "No difference detected" with 3 seeds does not prove there is none.
- **An unexplained run:** `separate_old_s42` was 27% slower, with lower GPU power and no heat flag. We don't know why. It's reported, not hidden, and the paper shows the comparison without it too.

**You should be able to explain all of this yourself before submitting.** Reviewers and the researchers you contact will ask.

## 12. Open issues and limitations (to remember for the paper)
- Cyrillic is transliterated, not native (ь/ц errors). Error rate still to be measured by you.
- new2026 has ~1% missed words (mostly names); reported as is.
- FLORES/Belebele are translated news/Wikipedia, not everyday Uzbek.
- English prompt frame: a real Uzbek user would write the instruction in Uzbek. A possible extra experiment.
- Gated models (Aya, official Llama/Gemma repos) need your HF account. Mirrors were used for tokenizers and for Gemma 3 4B.
- Literature check did not cover Google Scholar / ACL Anthology directly.
- Nothing is committed to git yet.
