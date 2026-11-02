# Dose-response (exploratory)

New letters per 100 letters: mean 4.85, SD 1.41, tertile edges 4.24 and 5.36. Slope: change in the new2026-minus-okina accuracy gap, in points, per one more new letter per 100 letters. Thirds: the gap in points, from fewest to most new letters.

| model            |   slope |    lo |    hi |   slope, length controlled |   lo2 |   hi2 |   third 1 |   third 2 |   third 3 |
|:-----------------|--------:|------:|------:|---------------------------:|------:|------:|----------:|----------:|----------:|
| Gemma 3 4B       |   -1.05 | -3.23 |  0.74 |                      -1.03 | -3.26 |  0.86 |      -4   |      -7   |      -9.3 |
| Qwen 2.5 7B      |   -1.24 | -3.21 |  0.64 |                      -1.3  | -3.25 |  0.6  |      -4   |      -2   |      -8   |
| Llama 3.1 8B     |   -1.22 | -3.23 |  0.74 |                      -1.09 | -3.08 |  0.89 |      -2.3 |      -8.7 |      -9.7 |
| Qwen 3 8B        |    0.28 | -1.17 |  1.78 |                       0.39 | -1.08 |  1.88 |      -6   |      -3.3 |      -3.7 |
| Gemma 3 12B      |   -1.15 | -2.38 | -0.11 |                      -1.22 | -2.38 | -0.19 |      -1.7 |      -3   |      -4.7 |
| Mistral Nemo 12B |   -1.49 | -3.2  |  0.23 |                      -1.29 | -3    |  0.42 |      -5.7 |     -12.3 |     -10.7 |
| Pooled, 6 models |   -0.98 | -1.76 | -0.26 |                      -0.92 | -1.72 | -0.18 |      -3.9 |      -6.1 |      -7.7 |
