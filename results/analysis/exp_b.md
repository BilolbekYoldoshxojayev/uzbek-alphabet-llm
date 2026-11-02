# Experiment B: LoRA on Gemma 3 4B, old vs new spelling

**Per run:**

| spelling   |   seed |   tokens |   train s |   tok/s |   energy Wh |   peak GB |   bits/word before |   bits/word after |
|:-----------|-------:|---------:|----------:|--------:|------------:|----------:|-------------------:|------------------:|
| old        |     42 |  1429899 |    3778.8 |   378.3 |        78.5 |      6.58 |              22.97 |             9.476 |
| old        |     43 |  1429899 |    3778.5 |   378.3 |        76.6 |      6.58 |              22.97 |             9.46  |
| old        |     44 |  1429899 |    3356.3 |   425.9 |        72.3 |      6.58 |              22.97 |             9.475 |
| new        |     42 |  1407916 |    3386.9 |   415.6 |        74   |      6.58 |              28.38 |            10.012 |
| new        |     43 |  1407916 |    3459.9 |   406.8 |        75.1 |      6.58 |              28.38 |            10.008 |
| new        |     44 |  1407916 |    3304.7 |   425.9 |        71   |      6.58 |              28.38 |            10.008 |

**Mean and SD:**

| spelling   |   ('tokens', 'mean') |   ('tokens', 'std') |   ('train s', 'mean') |   ('train s', 'std') |   ('tok/s', 'mean') |   ('tok/s', 'std') |   ('energy Wh', 'mean') |   ('energy Wh', 'std') |   ('peak GB', 'mean') |   ('peak GB', 'std') |   ('bits/word before', 'mean') |   ('bits/word before', 'std') |   ('bits/word after', 'mean') |   ('bits/word after', 'std') |
|:-----------|---------------------:|--------------------:|----------------------:|---------------------:|--------------------:|-------------------:|------------------------:|-----------------------:|----------------------:|---------------------:|-------------------------------:|------------------------------:|------------------------------:|-----------------------------:|
| new        |          1.40792e+06 |                   0 |               3383.83 |                77.65 |              416.1  |               9.56 |                   73.37 |                   2.12 |                  6.58 |                    0 |                          28.38 |                             0 |                         10.01 |                         0    |
| old        |          1.4299e+06  |                   0 |               3637.87 |               243.84 |              394.17 |              27.48 |                   75.8  |                   3.18 |                  6.58 |                    0 |                          22.97 |                             0 |                          9.47 |                         0.01 |

**Accuracy %.** Base model: latin_okina 59.9, new2026 53.1.

|   seed |   old-trained, latin_okina |   old-trained, new2026 |   new-trained, latin_okina |   new-trained, new2026 |
|-------:|---------------------------:|-----------------------:|---------------------------:|-----------------------:|
|     42 |                       57.9 |                   54.7 |                       58.6 |                   58.6 |
|     43 |                       58.8 |                   54.3 |                       57.6 |                   57.8 |
|     44 |                       59.4 |                   55.1 |                       57.6 |                   58.3 |

**Comparisons:**

| comparison (mean of 3 seeds)                           | points [95% CI]   |
|:-------------------------------------------------------|:------------------|
| old-trained on old text vs base                        | -1.2 [-3.5, +0.9] |
| new-trained on old text vs base                        | -2.0 [-4.3, +0.4] |
| old-trained on new text vs base                        | +1.6 [-1.0, +4.3] |
| new-trained on new text vs base                        | +5.1 [+2.6, +7.9] |
| gap new-old text, base                                 | -6.8 [-9.7, -4.0] |
| gap new-old text, old-trained                          | -4.0 [-6.1, -1.9] |
| gap new-old text, new-trained                          | +0.3 [-1.1, +1.8] |
| on new text: new-trained vs old-trained                | +3.5 [+1.5, +5.6] |
| on old text: new-trained vs old-trained                | -0.8 [-2.0, +0.4] |
| own spelling: new-trained on new vs old-trained on old | -0.5 [-2.1, +1.2] |

**Per seed:**

| comparison                           |   seed |   points |   p Holm |
|:-------------------------------------|-------:|---------:|---------:|
| new-trained vs base, new text        |     42 |      5.4 |   0.0035 |
| new-trained vs old-trained, new text |     42 |      3.9 |   0.0131 |
| own spelling, new vs old             |     42 |      0.7 |   1      |
| new-trained vs base, new text        |     43 |      4.7 |   0.0161 |
| new-trained vs old-trained, new text |     43 |      3.4 |   0.0407 |
| own spelling, new vs old             |     43 |     -1   |   1      |
| new-trained vs base, new text        |     44 |      5.2 |   0.0047 |
| new-trained vs old-trained, new text |     44 |      3.2 |   0.0407 |
| own spelling, new vs old             |     44 |     -1.1 |   1      |
