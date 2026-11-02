# Validity checks

**Files:** 106 result files checked; incomplete or out of order: none.

**Manifests:** 3 from this run. Files differing between them: none. Files changed since the last manifest: none.

**Determinism (Gemma 3 4B bf16):** 100/100 predictions identical on repeat; largest log-prob difference 0.00e+00.

**Llama, corrupted file vs repaired** (the paper uses repaired only):

| condition      |   corrupted acc |   repaired acc |   same predictions % |
|:---------------|----------------:|---------------:|---------------------:|
| cyrillic       |            58.2 |           60.4 |                 85.1 |
| latin_ascii    |            67.4 |           69.7 |                 89.4 |
| latin_backtick |            67.8 |           69.2 |                 88.4 |
| latin_okina    |            67.7 |           69.1 |                 91.1 |
| latin_rsquo    |            68.9 |           69   |                 89.6 |
| new2026        |            59.9 |           62.2 |                 86   |
| new_og         |            65.8 |           66.7 |                 87.9 |
| new_shch       |            63.6 |           66.1 |                 88.6 |

**Mean probability mass on A-D, % (okina and new2026 pooled):**

| model            |   raw |   chat |   uz |
|:-----------------|------:|-------:|-----:|
| Gemma 3 4B       |  93.8 |  100   | 70.4 |
| Qwen 2.5 7B      |  91.4 |  100   | 90.8 |
| Llama 3.1 8B     |  92.6 |   99.9 | 99.1 |
| Qwen 3 8B        |  71.7 |   99.9 | 26.7 |
| Gemma 3 12B      |  85   |  100   | 96.1 |
| Mistral Nemo 12B |  60.7 |   94.9 | 73.9 |

**Heat during training** (monitor reads every 10 s):

| run                      |   readings |   mean C |   max C |   slowdown % |
|:-------------------------|-----------:|---------:|--------:|-------------:|
| lora_old_s42             |        391 |     54.8 |      65 |            0 |
| lora_new_s42             |        353 |     56.4 |      63 |            0 |
| lora_new_s43             |        361 |     56.3 |      63 |            0 |
| lora_old_s43             |        392 |     50.3 |      66 |            0 |
| lora_old_s44             |        351 |     50.1 |      57 |            0 |
| lora_new_s44             |        345 |     50   |      57 |            0 |
| scratch_separate_old_s42 |        254 |     53   |      59 |            0 |
| scratch_separate_new_s42 |        197 |     57.2 |      62 |            0 |
| scratch_shared_new_s42   |        200 |     57.7 |      62 |            0 |
| scratch_shared_old_s42   |        193 |     59   |      63 |            0 |
| scratch_separate_old_s43 |        190 |     58.6 |      63 |            0 |
| scratch_separate_new_s43 |        191 |     58.6 |      62 |            0 |
| scratch_shared_new_s43   |        194 |     58.9 |      62 |            0 |
| scratch_shared_old_s43   |        193 |     59   |      63 |            0 |
| scratch_separate_old_s44 |        189 |     58.9 |      63 |            0 |
| scratch_separate_new_s44 |        192 |     58.4 |      62 |            0 |
| scratch_shared_new_s44   |        194 |     58.8 |      63 |            0 |
| scratch_shared_old_s44   |        194 |     58.8 |      62 |            0 |
