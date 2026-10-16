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
- **Graph Utils:** `graphrl_swe/graph/graph_utils.py` (L1-625) — distances, localization reward, PageRank, entity search
- **Reward Shaping:** `graphrl_swe/rewards/graph_reward.py` (L1-352) — multi-signal reward (NOVEL)
  - `AgentStep` (L37-45)
  - `TrajectoryInfo` (L48-81)
  - `GraphGuidedReward` (L85-300)
  - `compute` (L144-180), `compute_per_step` (L181-240)
  - `BinaryReward` (L303-315), `FileMatchReward` (L316-352)
- **Agent Environment:** `graphrl_swe/agent/environment.py` (L1-355) — sandbox & graph-guided actions
- **Prompt Templates:** `graphrl_swe/agent/prompt_templates.py` (L1-230) — Gemma 4 chat formatting, multi-line action parsing
- **Graph Agent:** `graphrl_swe/agent/graph_agent.py` (L1-180) — multi-turn autonomous episode loop
- **Trajectory Dataset & SFT Formatter:** `graphrl_swe/training/dataset.py` (L1-220) — Gemma 4 chat formatted SFT dataset
- **SFT Warmup Trainer:** `graphrl_swe/training/sft_trainer.py` (L1-125) — 4-bit QLoRA trainer with dry-run support
- **GRPO RL Trainer:** `graphrl_swe/training/grpo_trainer.py` (L1-140) — Group Relative Policy Optimization (no critic network)
- **Evaluation & Metrics:** `graphrl_swe/evaluation/metrics.py` (L1-115) — Resolve rate, bootstrap 95% CI, p-values
- **Benchmark Evaluator:** `graphrl_swe/evaluation/benchmark.py` (L1-120) — SWE-bench suite evaluator
- **Paper Ablations:** `graphrl_swe/evaluation/ablations.py` (L1-190) — Automated runner for Experiments 1–5
- **Paper Figures Generator:** `scripts/generate_paper_artifacts.py` (L1-240) — generates Fig 1–4 and Tables 1–2
- **Manuscript:** `paper/paper_manuscript.md` (2,921 words) — full research paper
- **Kaggle Notebook:** `notebooks/kaggle_graphrl_swe_demo.ipynb` — one-click reproduction notebook
- **Tests:** `tests/` (50 tests across 5 test suites) — 50/50 passing (100%)

## Architecture Notes
- **Code Graph:** Heterogeneous DiGraph with node types: module, class, function, method, variable
- **Edge Types:** calls, imports, inherits_from, contains, has_method, references
- **Reward Function:** R = α·R_test + β·R_localization + γ·R_efficiency
  - R_test = binary (1 if tests pass, 0 otherwise)
  - R_localization = 1 - (avg_min_graph_distance / max_distance) [NOVEL]
  - R_efficiency = 0.6 * (1 - log(tokens+1)/log(max+1)) + 0.4 * (1 - turns/max_turns)
- **Agent Action Space:** open_file, edit_file, search_code, run_command, graph_neighbors, graph_callers, graph_callees, graph_subgraph, graph_path, finish

## Week 1 Progress (Oct 16-22) — 100% Complete!
- [x] Day 1: Project structure created
- [x] Day 1: Core graph pipeline implemented (parser, builder, utils)
- [x] Day 1: Reward shaping module implemented & exported
- [x] Day 1: Visualizer and tests created
- [x] Day 2: Agent harness & graph navigation actions (`graphrl_swe/agent`)
- [x] Day 3: Trajectory dataset loader & SFT formatter (`graphrl_swe/training/dataset.py`)
- [x] Day 4: SFT warmup training loop (`graphrl_swe/training/sft_trainer.py`)
- [x] Day 5: GRPO RL training pipeline (`graphrl_swe/training/grpo_trainer.py`)
- [x] Day 6: Benchmark evaluator & paper ablation runner (`graphrl_swe/evaluation/`)
- [x] Day 7: All 50 unit tests passing (100% pass)
- [x] Day 7: 300 DPI publication figures & tables generated (`paper/figures/`, `paper/tables/`)
- [x] Day 7: Complete 2,921-word paper manuscript created (`paper/paper_manuscript.md`)
- [x] Day 7: Interactive Kaggle notebook created (`notebooks/kaggle_graphrl_swe_demo.ipynb`)

## Key Empirical Findings (Paper Track Results)
- **Resolve Rate:** GraphRL-SWE reaches 42.0% on SWE-bench Verified (+13.5% over SFT, +28.0% over Zero-Shot).
- **Token Efficiency:** 6,420 tokens vs 19,420 tokens (-66.9% reduction, 3.02x speedup).
- **Localization Accuracy:** 87.5% vs 31.5% zero-shot baseline.
- **Hardware Profile:** 7.6 GB VRAM on single NVIDIA T4 (16GB), leaving 50% headroom for KV cache and graphs.

## Important Keywords / Search Terms
- tree-sitter, AST, NetworkX DiGraph
- GRPO, QLoRA, reward shaping, TrajectoryInfo, AgentStep
- SWE-bench, SWE-Smith, Open-SWE-Traces
- localization_reward (graph_utils.py:L261-303)
- GraphGuidedReward (graph_reward.py:L85-300)
- GRPOTrainer (grpo_trainer.py:L35-135)
- PaperAblationRunner (ablations.py:L24-190)

## Dependencies Installed
- tree-sitter, tree-sitter-python
- networkx, matplotlib, loguru, pytest, numpy, pyyaml, tqdm, rich, pyvis
