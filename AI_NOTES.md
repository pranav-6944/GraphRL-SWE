# AI Notes — GraphRL-SWE Paper Track Project

## Project Overview
- **Competition:** Kaggle Gemma 4 Developer Agent Paper Track
- **Deadline:** Nov 12, 2026, 11:59 PM UTC
- **Target Prize:** Overall Best Paper ($15K)
- **Paper Title:** "GraphRL-SWE: Graph-Guided Reinforcement Learning for Efficient Code Agent Navigation on Consumer Hardware"

## Key File Locations
- **Config:** `configs/default.yaml`
- **AST Parser:** `graphrl_swe/graph/ast_parser.py` (L1-350) — tree-sitter based parsing
- **Graph Builder:** `graphrl_swe/graph/graph_builder.py` (L1-380) — NetworkX DiGraph construction
- **Graph Utils:** `graphrl_swe/graph/graph_utils.py` (L1-400) — distances, localization reward, PageRank
- **Reward Shaping:** `graphrl_swe/rewards/graph_reward.py` (L1-300) — multi-signal reward (NOVEL)
- **Visualizer:** `graphrl_swe/graph/visualizer.py` — paper figures
- **Tests:** `tests/test_graph.py` — unit tests for graph pipeline

## Architecture Notes
- **Code Graph:** Heterogeneous DiGraph with node types: module, class, function, method, variable
- **Edge Types:** calls, imports, inherits_from, contains, has_method, references
- **Reward Function:** R = α·R_test + β·R_localization + γ·R_efficiency
  - R_test = binary (1 if tests pass, 0 otherwise)
  - R_localization = 1 - (avg_min_graph_distance / max_distance) ← NOVEL
  - R_efficiency = token/step efficiency penalty

## Week 1 Progress (Oct 16-22)
- [x] Day 1: Project structure created
- [x] Day 1: Core graph pipeline implemented (parser, builder, utils)
- [x] Day 1: Reward shaping module implemented
- [x] Day 1: Visualizer and tests created
- [ ] Day 2: Run tests, fix issues, parse real SWE-bench repos
- [ ] Day 3: Implement trajectory data loader
- [ ] Day 4-5: SFT warmup training setup
- [ ] Day 6-7: SFT training on Kaggle T4

## Important Keywords / Search Terms
- tree-sitter, AST, NetworkX DiGraph
- GRPO, QLoRA, reward shaping
- SWE-bench, SWE-Smith, Open-SWE-Traces
- localization_reward (graph_utils.py)
- GraphGuidedReward (graph_reward.py)

## Dependencies Installed
- tree-sitter, tree-sitter-python
- networkx, matplotlib, loguru, pytest, numpy, pyyaml, tqdm, rich, pyvis
