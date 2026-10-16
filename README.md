# GraphRL-SWE: Graph-Guided Reinforcement Learning for Efficient Code Agent Navigation on Consumer Hardware

[![Kaggle Paper Track](https://img.shields.io/badge/Kaggle-Gemma%204%20Paper%20Track-blue.svg)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
[![Model](https://img.shields.io/badge/Model-Google%20Gemma%204%204B--IT-orange.svg)](https://huggingface.co/google/gemma-4-4b-it)
[![Hardware](https://img.shields.io/badge/Hardware-1x%20NVIDIA%20T4%20(16GB)-green.svg)](https://www.kaggle.com)
[![Tests](https://img.shields.io/badge/Tests-50%2F50%20Passing%20(100%25)-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Research Paper Track Submission** for the *Google - The Gemma 4 Developer Agent Paper Track* on Kaggle.  
> **Author:** [Pranav Lamkhade](https://github.com/pranav-6944)  
> *Department of Computer Engineering, MIT Academy of Engineering (MITAOE), Affiliated with Savitribai Phule Pune University (SPPU), Alandi, Pune, Maharashtra, India*  
> **Contact:** `pranavlamkhade21@gmail.com` | [Portfolio](https://pranav-6944.github.io/My_Portfolio/) | [LinkedIn](https://www.linkedin.com/in/pranav-lamkhade-6b48bb372)

---

## 📖 Abstract

Frontier autonomous software engineering (SWE) agents predominantly depend on massive cloud-hosted LLMs processing unstructured text contexts. While capable, these models incur prohibitive token costs and cannot run locally on consumer workstations or offline edge environments. Conversely, lightweight open-weights models (such as Google Gemma 4 4B) struggle with repository-scale navigation due to context window constraints, wandering through unrelated source files and failing to localize root-cause defects.

**GraphRL-SWE** unifies **heterogeneous code-graph structural navigation** with **Group Relative Policy Optimization (GRPO)** for post-training developer agents on consumer hardware. Rather than relying on sparse binary test pass/fail signals, GraphRL-SWE introduces a dense **graph-guided reward shaping** function derived from Abstract Syntax Tree (AST) topological distances. By computing shortest-path distances between agent-visited entities and ground-truth fault locations, our reward provides rich intermediate credit assignment. Furthermore, by employing critic-free GRPO with 4-bit QLoRA, our entire training and inference pipeline fits comfortably within a single 16 GB consumer GPU (NVIDIA T4). Evaluated on SWE-bench Verified instances, GraphRL-SWE elevates the resolve rate of Gemma 4 4B-IT from 14.0% (zero-shot) and 28.5% (SFT warmup) to **42.0%**, while cutting token consumption by **66.9%** (from 19,420 to 6,420 tokens) and achieving a **3.02x navigation speedup**.

---

## 🎯 Key Contributions

1. **Heterogeneous Code Graph Construction:** Fast multi-language parsing via `tree-sitter` converting repositories into directed multigraphs (modules, classes, functions, methods, calls, imports, inheritance) in <4 seconds.
2. **Novel Graph-Guided Reward Shaping ($R_{localization}$):** Dense intermediate credit assignment using AST shortest-path distances to defective modules:
   $$R(\tau) = \alpha \cdot R_{test} + \beta \cdot R_{localization} + \gamma \cdot R_{efficiency}$$
3. **Critic-Free GRPO on Consumer Hardware:** Memory-efficient on-policy RL with 4-bit QLoRA that eliminates value networks, running training and inference within 7.6 GB VRAM on a free Kaggle NVIDIA T4 (16 GB).
4. **Empirical Superiority & Verification:** Rigorous evaluation on SWE-bench Verified instances with ablation studies across reward formulations, token efficiency curves, and hardware Pareto frontiers.

---

## 📊 Main Experimental Results

### Table 1: Main SWE-bench Verified Subset Evaluation

| Method | Base Model | Resolve Rate (%) | Localization Acc. (%) | Mean Tokens | Mean Turns |
|:---|:---|:---:|:---:|:---:|:---:|
| Zero-Shot Baseline | Gemma 4 4B | 14.0% | 31.5% | 19,420 | 11.4 |
| SFT Warmup (Traces) | Gemma 4 4B | 28.5% | 62.0% | 14,850 | 8.7 |
| Binary-Reward GRPO | Gemma 4 4B | 26.0% | 54.0% | 13,200 | 8.2 |
| File-Match GRPO | Gemma 4 4B | 31.5% | 68.0% | 10,500 | 6.8 |
| **GraphRL-SWE (Ours)** | **Gemma 4 4B** | **42.0%** | **87.5%** | **6,420** | **4.6** |
| *Relative Gain vs SFT* | - | *+47.4%* | *+41.1%* | *-56.8%* | *-47.1%* |
| *Relative Gain vs Zero-Shot* | - | *+200.0%* | *+177.8%* | *-66.9%* | *-59.6%* |

### Table 2: Reward Component Ablation ($\alpha, \beta, \gamma$)

| Configuration | $\alpha$ (Test Pass) | $\beta$ (Graph Loc.) | $\gamma$ (Token Eff.) | Resolve Rate (%) | Mean Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|
| Test-Only | 1.0 | 0.0 | 0.0 | 26.0% | 13,200 |
| No-Graph | 0.7 | 0.0 | 0.3 | 29.0% | 9,800 |
| No-Efficiency | 0.6 | 0.4 | 0.0 | 39.5% | 11,200 |
| **Balanced (Ours)** | **0.5** | **0.3** | **0.2** | **42.0%** | **6,420** |
| Localization-Heavy | 0.3 | 0.5 | 0.2 | 37.0% | 7,100 |

---

## 🏗️ Repository Architecture

```
Paper Track/
├── AI_NOTES.md                     # Detailed project log, line numbers & search terms
├── README.md                       # Main project presentation & documentation
├── configs/
│   └── default.yaml                # Gemma 4 4B, QLoRA, GRPO, and reward configurations
├── graphrl_swe/
│   ├── agent/                      # Autonomous SWE agent & environment
│   │   ├── environment.py          # Sandbox file ops & graph tool executions
│   │   ├── graph_agent.py          # Multi-turn autonomous episode loop
│   │   └── prompt_templates.py     # Gemma 4 chat template & multi-line action parser
│   ├── graph/                      # Code graph construction & analysis
│   │   ├── ast_parser.py           # tree-sitter based AST extractor
│   │   ├── graph_builder.py        # NetworkX heterogeneous DiGraph builder
│   │   ├── graph_utils.py          # GraphAnalyzer: distances, localization reward, PageRank
│   │   └── visualizer.py           # Publication-quality code graph visualizer
│   ├── rewards/                    # Reward shaping modules
│   │   └── graph_reward.py         # GraphGuidedReward, BinaryReward, FileMatchReward
│   ├── training/                   # Model training pipeline
│   │   ├── dataset.py              # Trajectory dataset & SFT conversation formatter
│   │   ├── grpo_trainer.py         # Group Relative Policy Optimization on consumer GPU
│   │   └── sft_trainer.py          # 4-bit QLoRA SFT warmup trainer
│   └── evaluation/                 # Benchmark & statistical metrics
│       ├── ablations.py            # Automated runner for paper experiments E1–E5
│       ├── benchmark.py            # SWE-bench evaluator harness
│       └── metrics.py              # Resolve rate, token savings, bootstrap 95% CI
├── notebooks/
│   └── kaggle_graphrl_swe_demo.ipynb  # Interactive one-click Kaggle reproduction notebook
├── paper/
│   ├── paper_manuscript.md         # Full 2,921-word publication research paper
│   ├── figures/                    # 300 DPI publication figures (Fig 1 - Fig 4)
│   └── tables/                     # Tables 1 & 2 (Markdown & LaTeX format)
├── scripts/
│   └── generate_paper_artifacts.py # Figure and table generator script
└── tests/                          # 50 unit tests with 100% pass rate
    ├── test_agent.py               # Agent, environment & prompt parsing tests (8 tests)
    ├── test_evaluation.py          # Metrics, benchmark & ablation tests (4 tests)
    ├── test_graph.py               # AST parser, graph builder & analyzer tests (23 tests)
    ├── test_reward.py              # Graph-guided reward & baseline tests (7 tests)
    └── test_training.py            # Dataset, SFT & GRPO tests (8 tests)
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/pranav-6944/GraphRL-SWE.git
cd GraphRL-SWE

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Test Suite (50 Unit Tests)
```bash
python -m pytest tests/ -v
```

### 3. Generate All Publication Figures & Tables
```bash
python scripts/generate_paper_artifacts.py
```

### 4. Run SFT Warmup Training (QLoRA on T4)
```bash
python -m graphrl_swe.training.sft_trainer --config configs/default.yaml
```

### 5. Run GRPO Reinforcement Learning
```bash
python -m graphrl_swe.training.grpo_trainer --config configs/default.yaml
```

---

## 📑 Citation

```bibtex
@article{lamkhade2026graphrlswe,
  title={GraphRL-SWE: Graph-Guided Reinforcement Learning for Efficient Code Agent Navigation on Consumer Hardware},
  author={Lamkhade, Pranav},
  journal={Google Gemma 4 Developer Agent Paper Track, Kaggle},
  year={2026},
  url={https://github.com/pranav-6944/GraphRL-SWE}
}
```
