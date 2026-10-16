# Table 1: Main Evaluation Results on SWE-bench Verified Subset

| Method | Base Model | Resolve Rate (%) | Localization Acc. (%) | Mean Tokens | Mean Turns |
|:---|:---|:---:|:---:|:---:|:---:|
| Zero-Shot Baseline | Gemma 4 4B | 14.0% | 31.5% | 19,420 | 11.4 |
| SFT Warmup (Traces) | Gemma 4 4B | 28.5% | 62.0% | 14,850 | 8.7 |
| **GraphRL-SWE (Ours)** | **Gemma 4 4B** | **42.0%** | **87.5%** | **6,420** | **4.6** |
| *Gain vs SFT* | - | *+13.5%* | *+25.5%* | *-56.8%* | *-47.1%* |
| *Gain vs Zero-Shot* | - | *+28.0%* | *+56.0%* | *-66.9%* | *-59.6%* |
