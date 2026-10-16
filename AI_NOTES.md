# AI Notes — GraphRL-SWE Paper Track Project

## Project Overview
- **Competition:** Kaggle Gemma 4 Developer Agent Paper Track
- **Deadline:** Nov 12, 2026, 11:59 PM UTC
- **Target Prize:** Overall Best Paper ($15K)
- **Paper Title:** "GraphRL-SWE: Graph-Guided Reinforcement Learning for Efficient Code Agent Navigation on Consumer Hardware"

## Key File Locations
- **Config:** `configs/default.yaml` (L1-113) — model, lora, graph, rewards, sft, grpo, evaluation params
- **AST Parser:** `graphrl_swe/graph/ast_parser.py` (L1-569) — tree-sitter based parsing + fallback
- **Graph Builder:** `graphrl_swe/graph/graph_builder.py` (L1-450) — NetworkX DiGraph construction
- **Graph Utils:** `graphrl_swe/graph/graph_utils.py` (L1-430) — distances, localization reward, PageRank
- **Reward Shaping:** `graphrl_swe/rewards/graph_reward.py` (L1-352) — multi-signal reward (NOVEL)
  - `AgentStep` (L37-45)
  - `TrajectoryInfo` (L48-81)
  - `GraphGuidedReward` (L85-300)
  - `compute` (L144-180), `compute_per_step` (L181-240)
  - `BinaryReward` (L303-315), `FileMatchReward` (L316-352)
- **Agent Environment:** `graphrl_swe/agent/environment.py` (L1-355) — sandbox & graph-guided actions
- **Prompt Templates:** `graphrl_swe/agent/prompt_templates.py` (L1-230) — Gemma 4 chat formatting, multi-line action parsing
- **Graph Agent:** `graphrl_swe/agent/graph_agent.py` (L1-180) — multi-turn autonomous episode loop
- **Visualizer:** `graphrl_swe/graph/visualizer.py` (L1-330) — paper figures & reward curves
- **Tests:** `tests/test_graph.py` (23 tests), `tests/test_reward.py` (7 tests), `tests/test_agent.py` (8 tests) — 38/38 passing (100%)

## Architecture Notes
- **Code Graph:** Heterogeneous DiGraph with node types: module, class, function, method, variable
- **Edge Types:** calls, imports, inherits_from, contains, has_method, references
- **Reward Function:** R = α·R_test + β·R_localization + γ·R_efficiency
  - R_test = binary (1 if tests pass, 0 otherwise)
  - R_localization = 1 - (avg_min_graph_distance / max_distance) [NOVEL]
  - R_efficiency = 0.6 * (1 - log(tokens+1)/log(max+1)) + 0.4 * (1 - turns/max_turns)
- **Agent Action Space:** open_file, edit_file, search_code, run_command, graph_neighbors, graph_callers, graph_callees, graph_subgraph, graph_path, finish

## Week 1 Progress (Oct 16-22)
- [x] Day 1: Project structure created
- [x] Day 1: Core graph pipeline implemented (parser, builder, utils)
- [x] Day 1: Reward shaping module implemented & exported
- [x] Day 1: Visualizer and tests created
- [x] Day 2: Agent harness & graph navigation actions (`graphrl_swe/agent`)
- [x] Day 2: 38/38 unit tests passing (100% pass)
- [ ] Day 3-4: Trajectory dataset loader & SFT formatter (`graphrl_swe/training/dataset.py`)
- [ ] Day 4-5: SFT warmup training loop (`graphrl_swe/training/sft_trainer.py`)
- [ ] Day 5-6: GRPO RL training pipeline (`graphrl_swe/training/grpo_trainer.py`)
- [ ] Day 7: Evaluation suite & Kaggle reproduction notebook

## Important Keywords / Search Terms
- tree-sitter, AST, NetworkX DiGraph
- GRPO, QLoRA, reward shaping, TrajectoryInfo, AgentStep
- SWE-bench, SWE-Smith, Open-SWE-Traces
- localization_reward (graph_utils.py:L130-185)
- GraphGuidedReward (graph_reward.py:L85-300)

## Dependencies Installed
- tree-sitter, tree-sitter-python
- networkx, matplotlib, loguru, pytest, numpy, pyyaml, tqdm, rich, pyvis
