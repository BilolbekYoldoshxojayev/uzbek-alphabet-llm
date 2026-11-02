# What is left before submitting

All experiments and analyses are done, and the author hand-checked the converted text samples with no corrections.

## Two versions
- **Short, for the workshop:** `main.tex` + `content.tex`. Anonymous in review mode, including the converter's name and link. For the final version, follow the comment in `main.tex`.
- **Long, for arXiv:** `main_long.tex` + `content_long.tex` + `tables_long.tex`. Shows your name and email. `arxiv_upload/` is a copy ready for arXiv; rebuild it if the long paper changes.
- If a result changes, update both texts. Re-create the long tables with `scripts/make_long_tables.py`.

## Only you can do these
1. **Compile the short paper:** `cd paper; ..\tools\tectonic\tectonic.exe -X compile main.tex`. Check that the converter footnote says "withheld" and that your name appears nowhere.
2. **Test the arXiv folder on pdfLaTeX.** It has never been compiled with pdfLaTeX, which arXiv uses. Upload `arxiv_upload.zip` to Overleaf, set the compiler to pdfLaTeX and the main file to `main_long.tex`. Look at the Cyrillic text and the o' letters.
3. **Create an arXiv account** and start a `cs.CL` submission. arXiv emails you an endorsement link and a 6-character code.
4. **Send one endorser email.** Two drafts are in your Gmail. Put in the code, read them, and send one. Wait about a week before trying the other. arXiv says not to email many people at once or repeat requests to the same person.
5. **Submit to arXiv** once endorsed: upload the zip, choose a license (CC BY 4.0 suggested), and check arXiv's PDF before confirming.
6. **Submit to a workshop** when a call opens. LoResLM 2027 has been announced with no dates yet; no SIGTURK 2027 call was found. Check the AI-use, preprint and anonymity rules, and whether an author must register and pay.

## Optional
- Hugging Face dataset of the parallel FLORES + Belebele under CC BY-SA 4.0; needs your login.
- A second native speaker to check the samples, which would make the error rates independent.

## If the paper is too long
Cut in this order: the context sentence in RQ3, the dose-response paragraph, the UzLiB and Kletz sentences in Related Work.

## Stronger follow-ups
- API models (Claude, GPT, Gemini) on Belebele; needs keys and a small budget.
- Text written natively in the new alphabet, to test back-conversion and to test fine-tuning without the converter.
- Fine-tuning on more models than Gemma 3 4B, and more seeds for from-scratch training.
- A second task or benchmark (TUMLU, UzLiB), after checking licenses.
- Naive find-and-replace converter as a baseline for YangiAlifbo.
- Upstream fixes to YangiAlifbo (`docs/WALKTHROUGH.md` §4.3).
