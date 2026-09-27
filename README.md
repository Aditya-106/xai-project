# How Interpretable are Reasoning Explanations from Prompting Large Language Models?

## Paper Reproduction

This repository reproduces key experiments from:

> **Yeo Wei Jie, Ranjan Satapathy, Rick Goh, and Erik Cambria.** "How Interpretable are Reasoning Explanations from Prompting Large Language Models?" *Findings of NAACL 2024*, pages 2148–2164.

**Original Paper:** [ACL Anthology](https://aclanthology.org/2024.findings-naacl.138/)  
**Original Code:** [GitHub](https://github.com/SenticNet/CoT_interpretability)

---

## Overview

This project investigates whether Chain-of-Thought (CoT) reasoning explanations from LLMs are genuinely interpretable. We evaluate interpretability across three dimensions:

1. **Faithfulness** — Does the explanation reflect the model's actual reasoning?
2. **Robustness** — Does the explanation remain reliable under perturbations?
3. **Utility** — Does the explanation help a downstream model solve the task?

We reproduce the paper's proposed **Self-Entailment-Alignment Chain-of-Thought (SEA-CoT)** method, which selects the best explanation from multiple candidates using entailment + token-overlap scoring.

---

## Experimental Setup

| Component | Configuration |
|-----------|--------------|
| **Model** | Llama-2-70B-chat-GPTQ (4-bit quantized) |
| **Datasets** | StrategyQA, OpenBookQA, QASC |
| **Prompting Methods** | CoT, SC-CoT, QD, Self-Refine, SEA-CoT |
| **Evaluation Metrics** | Paraphrase (P), Counterfactual Unfaithfulness (CF-UF), Mistake Flip (M), Simulatability (LAS) |

---

## Reproduced Results

### Result 1: SEA-CoT Ablation on StrategyQA (Paper Table 1)

| Method | P ↓ | CF-UF ↓ | M ↑ | S ↑ |
|--------|------|---------|------|------|
| Random | 6.10 | 6.44 | 62.17 | 11.87 |
| Max | 1.80 | 6.60 | 61.80 | 12.59 |
| Overlap | 1.56 | 5.04 | 70.83 | 14.88 |
| Entailment | 2.38 | 5.46 | 69.99 | 13.46 |
| **O&E (SEA-CoT)** | **1.20** | **3.81** | **61.24** | **16.97** |

### Result 2: Main Interpretability Comparison (Paper Figure 5)

*See `results/figures/` for generated comparison plots.*

### Result 3: Cross-Prompting Interpretability Evaluation

*See `results/tables/` for full tables.*

---

## Project Structure

```
├── README.md
├── requirements.txt
├── configs/                    # Experiment configurations
├── data/                       # Dataset files
│   ├── strategyqa/
│   ├── obqa/
│   └── qasc/
├── prompts/                    # Prompt templates
├── src/                        # Core source code
│   ├── model.py               # LLM loading and inference
│   ├── generate.py             # Reasoning chain generation
│   ├── prompting/              # Prompting method implementations
│   │   ├── cot.py
│   │   ├── sc_cot.py
│   │   ├── sea_cot.py
│   │   ├── question_decomp.py
│   │   └── self_refine.py
│   ├── scoring/                # SEA-CoT scoring components
│   │   ├── entailment.py
│   │   └── overlap.py
│   ├── perturbation/           # Perturbation generation
│   │   ├── paraphrase.py
│   │   ├── counterfactual.py
│   │   └── mistake.py
│   ├── evaluation/             # Evaluation metrics
│   │   ├── faithfulness.py
│   │   ├── robustness.py
│   │   └── utility.py
│   └── data_loader.py          # Dataset loading
├── experiments/                # Experiment runner scripts
│   ├── run_generation.py
│   ├── run_perturbation.py
│   ├── run_evaluation.py
│   └── run_ablation.py
├── results/                    # Output results
│   ├── raw/
│   ├── tables/
│   └── figures/
├── notebooks/
│   └── analysis.ipynb
└── report/
    └── replication_summary.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 2. Download Datasets
```bash
python src/data_loader.py --download
```

### 3. Generate Reasoning Chains
```bash
# Generate CoT baselines
python experiments/run_generation.py --method cot --dataset strategyqa
python experiments/run_generation.py --method sc_cot --dataset strategyqa
python experiments/run_generation.py --method sea_cot --dataset strategyqa
```

### 4. Generate Perturbations
```bash
python experiments/run_perturbation.py --type paraphrase --dataset strategyqa
python experiments/run_perturbation.py --type counterfactual --dataset strategyqa
python experiments/run_perturbation.py --type mistake --dataset strategyqa
```

### 5. Run Evaluation
```bash
python experiments/run_evaluation.py --dataset strategyqa
```

### 6. Run SEA-CoT Ablation
```bash
python experiments/run_ablation.py --dataset strategyqa
```

---

## Hardware Requirements

- **Full reproduction (70B):** NVIDIA A100 80GB or equivalent
- **Reduced reproduction (13B/7B):** NVIDIA RTX 3090/4090 24GB
- **Cloud options:** Google Colab Pro+, AWS p4d instances, Lambda Labs

---

## Differences from Original Paper

| Aspect | Original | Our Reproduction |
|--------|----------|-----------------|
| Model | Llama-2-70B-chat-GPTQ | Same / smaller variant |
| Dataset size | Full test sets | Full / subset |
| Perturbation | GPT-3.5 (para/mistake), GPT-4 (CF) | Same / alternative |
| Seeds | Multiple | Same |

---

## Citation

```bibtex
@inproceedings{weijie2024interpretable,
  title={How Interpretable are Reasoning Explanations from Prompting Large Language Models?},
  author={Yeo Wei Jie and Ranjan Satapathy and Rick Goh and Erik Cambria},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
  pages={2148--2164},
  year={2024}
}
```
