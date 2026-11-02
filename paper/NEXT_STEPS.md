# What is left before submitting

All experiments and analyses are done. The two converted-text samples were reviewed by Claude, not by a native speaker, and the paper says so.

## Two versions
- **Short, for the workshop:** `main.tex` + `content.tex`. Anonymous in review mode, 4 pages.
- **Long, for arXiv:** `main_long.tex` + `content_long.tex` + `tables_long.tex`. Shows your name and email. Compile it with `..\tools\tectonic\tectonic.exe -X compile main_long.tex`.
- If a result changes, update both texts. Re-create the long tables with `scripts/make_long_tables.py`.
- Check the workshop's preprint rules before posting the long version.

## Only you can do these
1. **Compile.** `cd paper; ..\tools\tectonic\tectonic.exe -X compile main.tex`. Claude is blocked from running it. Check that Table 2 no longer overlaps the text and that the body still ends on page 4.
2. **Read both papers: done by you.** Be ready to explain every result if asked.
3. **Google Scholar check: done by you.** You found only Ro'zimatov, which is cited.
4. **Choose a venue.** No 2027 call for papers was found for SIGTURK or LoResLM. Check their AI-use policy, page limit and whether review is anonymous when the calls appear. LoResLM 2026 allowed 4-page short papers, with references not counted, and required a limitations section.
5. **Submit**, using your own accounts. arXiv needs an endorser for a first-time cs.CL submitter; ask a mentor.

## Strongly recommended
- **Hand-check: done by you.** If you found any correction, put it in the `handcheck_sample.tsv` files so the counts can be updated. A second native speaker would make the error rates independent.

## If the paper is too long
Cut in this order: the context sentence in RQ3, the dose-response paragraph, the UzLiB and Kletz sentences in Related Work.

## Optional, stronger follow-ups
- API models (Claude, GPT, Gemini) on Belebele; needs keys and a small budget.
- Text written natively in the new alphabet, to test back-conversion and to test fine-tuning without the converter.
- Fine-tuning on more models than Gemma 3 4B, and more seeds for from-scratch training.
- A second task or benchmark (TUMLU, UzLiB), after checking licenses.
- Naive find-and-replace converter as a baseline for YangiAlifbo.
- Upstream fixes to YangiAlifbo (`docs/WALKTHROUGH.md` §4.3).

## Release checklist
- GitHub repo: created but **private**, https://github.com/BilolbekYoldoshxojayev/uzbek-alphabet-llm. Make it public before posting the long paper, or its link will not work: `gh repo edit BilolbekYoldoshxojayev/uzbek-alphabet-llm --visibility public --accept-visibility-change-consequences`. The short paper omits the link so it stays anonymous for review; add it to the final version.
- Hugging Face dataset of the parallel FLORES + Belebele under CC BY-SA 4.0. Not done: it needs your Hugging Face login. The data is in the GitHub repo for now.
- One-page summary for Kuriyozov, Sharipov, Isbarov, Wegmann, Nguyen and Kletz.
