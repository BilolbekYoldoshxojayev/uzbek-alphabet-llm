# 4-week plan (Oct 30 – Nov 26, 2026)

Compute key: **CPU** = home PC · **GPU-local** = RTX 4080 Laptop 12 GB · **Kaggle** = free T4×2 / P100 (30 h/week) · **$** = paid API.

## Week 1 (Oct 30 – Nov 5): finish RQ1 + RQ3, lock the dataset
- [ ] Bilolbek: full hand-check of `data/parallel/flores_devtest/handcheck_sample.tsv` (new2026 and Cyrillic columns); report word- and sentence-level error rates. *CPU*
- [x] Proper-noun policy for new2026 (decided Oct 29): keep the engine output as-is; Uzbek-spelled names it misses (Polsha, Vashington, Bishkek) are counted as conversion errors in the hand-check and reported.
- [ ] Add Aya Expanse (needs your HF token + accepting Cohere's license), Claude and Gemini token counts (free count-tokens endpoints, needs API keys). *CPU*
- [ ] Deduplicate tokenizer families (Qwen 2.5 = Qwen 3, Phi-4 = cl100k) in the paper table; add one Uzbek-trained tokenizer for contrast.
- [ ] Run RQ1 on FLORES dev + Belebele passages to check the result holds across text. *CPU*
- [ ] RQ3: cost and context tables from measured token counts × prices checked on the day. *CPU*
- [ ] Plots: % token change vs official Latin, per tokenizer.
- [ ] Check license + script of Belebele (`uzn_Latn`), TUMLU Uzbek, UzLiB. Pick RQ2 data. **(your decision)**

## Week 2 (Nov 6 – 12): RQ2 on open models
- [ ] Build the 6-condition Belebele set (900 questions) with the same pipeline; hand-check 30 questions.
- [ ] Evaluation harness: identical prompt, 0-shot, answer chosen by log-likelihood of A/B/C/D (deterministic, no sampling), fixed seed, pinned versions, every run logged.
- [ ] Pilot on GPU-local with a 4B model (Gemma 3 4B, bf16) to validate the harness. *GPU-local*
- [ ] Full runs, 6 models × 6 conditions: Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B, Aya Expanse 8B, Mistral Nemo 12B (+ optionally an Uzbek fine-tune). 8–12B in 4-bit or on Kaggle. **(your decision: model list)** *Kaggle*
- [ ] Paired statistics: bootstrap CIs on accuracy differences + McNemar test per model.

## Week 3 (Nov 13 – 19): API models, second benchmark, RQ4
- [ ] 2–3 API models on Belebele × 6 conditions (≈5,400 calls/model; measure tokens on a 50-question pilot, then compute exact cost before the full run). **(your decision: which APIs, budget cap)** *$*
- [ ] Second benchmark (TUMLU Uzbek or UzLiB) on the 2–3 best open models. *Kaggle*
- [ ] RQ4: normalize every Latin variant → one form (official ʻ, and separately ASCII ') before the model; re-run; does the gap close? *Kaggle + $ (small)*
- [ ] Error analysis: which questions flip between conditions; are they the ones where tokenization changed most?

## Week 4 (Nov 20 – 26): write and release
- [ ] 4-page paper (SIGTURK / LoResLM format): intro, conditions + converter, RQ1–RQ4, limitations (conversion errors, Cyrillic is transliterated not native, FLORES is translated news).
- [ ] Release: GitHub repo + Hugging Face dataset (CC BY-SA 4.0, inherited from FLORES/Belebele).
- [ ] One-page result summary for Kuriyozov, Sharipov, Isbarov, Wegmann, Nguyen, Kletz.
- [ ] arXiv submission (needs an endorser for a first-time cs.CL author — ask a mentor).

## Budget sketch
| Item | Where | Cost |
|---|---|---|
| RQ1, RQ3, data building | CPU | free |
| 6 open models × 6 conditions × 900 Qs | Kaggle (~15–25 GPU-h) | free |
| 2–3 API models × 6 conditions × 900 Qs (+ RQ4) | API | to be computed from the pilot; aim ≤ $100 |
