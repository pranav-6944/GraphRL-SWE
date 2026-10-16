# GraphRL-SWE: Graph-Guided Reinforcement Learning for Efficient Code Agent Navigation on Consumer Hardware

[![Paper Track](https://img.shields.io/badge/Kaggle-Gemma%204%20Paper%20Track-blue)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Abstract

Current software engineering agents rely on massive cloud-hosted models and consume
excessive tokens navigating unfamiliar codebases. We introduce **GraphRL-SWE**, a
method that combines **code-graph-guided fault localization** with **reinforcement
learning reward shaping** to train efficient coding agents on consumer hardware.

Our approach constructs lightweight heterogeneous code graphs from repository ASTs
and uses graph-structural distances as dense intermediate rewards during GRPO training.
This enables a QLoRA-tuned **Gemma 4 4B** model running on a single T4 GPU to navigate
repositories with significantly fewer tokens and higher localization accuracy than
flat-context baselines.

## Key Contributions

1. **Graph-Guided Reward Shaping** — A novel RL reward function that uses code-graph
   structural distances to provide dense navigation signals beyond binary test pass/fail.
2. **Efficient Consumer-Hardware Pipeline** — Full training and inference pipeline
   that runs on a single T4 GPU (16GB VRAM) using QLoRA + GRPO.
3. **Comprehensive Efficiency Analysis** — Pareto frontier analysis of model size vs.
   resolve rate vs. token efficiency on SWE-bench subsets.

## Project Structure

```
graphrl_swe/
├── graph/          # Code graph construction (tree-sitter + NetworkX)
├── rewards/        # Graph-guided reward shaping modules
├── agent/          # SWE agent harness and tools
├── training/       # SFT warmup + GRPO RL training
└── evaluation/     # Metrics and SWE-bench evaluation
notebooks/          # Kaggle-compatible notebooks
paper/              # LaTeX paper and figures
configs/            # Training configurations
scripts/            # Data preparation and utility scripts
tests/              # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Parse a repository into a code graph
python scripts/parse_repos.py --repo_path /path/to/repo --output graphs/

# Run SFT warmup training
python -m graphrl_swe.training.sft_trainer --config configs/default.yaml

# Run GRPO RL training with graph rewards
python -m graphrl_swe.training.grpo_trainer --config configs/default.yaml

# Evaluate on SWE-bench
python -m graphrl_swe.evaluation.benchmark --model_path checkpoints/best
```

## Citation

```bibtex
@misc{lamkhade2026graphrlswe,
  title={GraphRL-SWE: Graph-Guided Reinforcement Learning for Efficient Code Agent Navigation on Consumer Hardware},
  author={Pranav Lamkhade},
  year={2026},
  note={Kaggle Gemma 4 Developer Agent Paper Track}
}
```
