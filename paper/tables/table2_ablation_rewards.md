# Table 2: Reward Component Ablation ($\\alpha, \\beta, \\gamma$)

| Configuration | $\\alpha$ (Test) | $\\beta$ (Graph Loc.) | $\\gamma$ (Efficiency) | Resolve Rate (%) | Mean Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|
| Test-Only | 1.0 | 0.0 | 0.0 | 26.0% | 13,200 |
| No-Graph | 0.7 | 0.0 | 0.3 | 29.0% | 9,800 |
| No-Efficiency | 0.6 | 0.4 | 0.0 | 39.5% | 11,200 |
| Balanced (Ours) | 0.5 | 0.3 | 0.2 | 42.0% | 6,420 |
| Loc-Heavy | 0.3 | 0.5 | 0.2 | 37.0% | 7,100 |
