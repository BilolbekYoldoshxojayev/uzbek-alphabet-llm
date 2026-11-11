| condition | accuracy % | vs latin_okina, points (p) | vs new2026, points (p) | prompt tokens vs new2026 |
|---|---|---|---|---|
| latin_okina | 64.0 | +0.0 (p=1) | +7.4 (p=1.59e-07) | +1.5% |
| new2026 | 56.6 | -7.4 (p=1.59e-07) | +0.0 (p=1) | +0.0% |
| new2026_nfd | 57.9 | -6.1 (p=1.98e-05) | +1.3 (p=0.349) | +9.7% |
| new2026_stripped | 55.9 | -8.1 (p=8.24e-08) | -0.7 (p=0.687) | -4.0% |

n = 900 questions, Gemma 3 4B bf16, raw English prompt, exact McNemar, no correction for multiple tests.
