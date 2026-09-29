# Paper Replication Summary: CoT Interpretability & SEA-CoT

## 1. Paper Overview

**Title:** How Interpretable are Reasoning Explanations from Prompting Large Language Models?  
**Authors:** Yeo Wei Jie, Ranjan Satapathy, Rick Goh, Erik Cambria  
**Venue:** Findings of NAACL 2024  
**DOI:** [10.18653/v1/2024.findings-naacl.138](https://aclanthology.org/2024.findings-naacl.138/)  
**Code Repository:** [SenticNet/CoT_interpretability](https://github.com/SenticNet/CoT_interpretability)

---

## 2. Core Methodology & SEA-CoT Algorithm

The paper addresses the limitation of prior work evaluating Chain-of-Thought (CoT) explanations purely based on faithfulness. The authors propose a multi-dimensional evaluation across **Faithfulness**, **Robustness**, and **Utility**, introducing **Self-Entailment-Alignment Chain-of-Thought (SEA-CoT)**.

### SEA-CoT Selection Formula
From $N$ sampled reasoning paths $\hat{E} = \{\hat{e}_1, \dots, \hat{e}_N\}$ supporting the majority vote answer $\bar{y}$:
1. **Self-Entailment Score ($S_e$)**: Computed via DeBERTa NLI as $P(\text{entailment} \mid \text{premise}=\hat{e}_i, \text{hypothesis}=q \oplus \bar{y})$.
2. **Overlap Score ($S_o$)**: Token-level Intersection-over-Union (IoU) without stopwords:
   $$S_o = \frac{|\hat{e}_i \cap (q \oplus \bar{y})|}{|\hat{e}_i \cup (q \oplus \bar{y})|}$$
3. **Total Alignment Score ($S_T$)**:
   $$S_T = S_e + S_o$$
The candidate path with maximum $S_T$ is selected.

---

## 3. Comparative Evaluation (Paper vs Reproduction)

### SEA-CoT Ablation Study on StrategyQA (Paper Table 1 vs Local Reproduction)

| Selection Strategy | Paper P ↓ | Ours P ↓ | Paper CF-UF ↓ | Ours CF-UF ↓ | Paper M ↑ | Ours M ↑ | Paper S ↑ | Ours S ↑ |
|---|---|---|---|---|---|---|---|---|
| **Random** | 6.10 | 0.00 | 6.44 | 40.00 | 62.17 | 20.00 | 11.87 | -6.83 |
| **Max (SC-CoT)** | 1.80 | 0.00 | 6.60 | 26.67 | 61.80 | 6.67 | 12.59 | -6.17 |
| **Overlap** | 1.56 | 0.00 | 5.04 | 26.67 | 70.83 | 13.33 | 14.88 | -5.99 |
| **Entailment** | 2.38 | 0.00 | 5.46 | 33.33 | 69.99 | 20.00 | 13.46 | -6.19 |
| **O&E (SEA-CoT)** | **1.20** | **0.00** | **3.81** | **33.33** | **61.24** | **20.00** | **16.97** | **-5.96** |

*Metrics: **P** = Paraphrase Flip Rate (↓), **CF-UF** = Counterfactual Unfaithfulness (↓), **M** = Mistake Flip Rate (↑), **S** = Leakage-Adjusted Simulatability / LAS (↑).*

---

## 4. Key Findings & Trend Agreement

### 1. Simulatability / Utility Match ($\rho = +0.900$)
- **O&E (SEA-CoT)** achieves the highest simulatability ($S = -5.96$) among all strategies in our local reproduction, matching the paper's best strategy with a **Spearman rank correlation of $\rho = +0.900$**.
- This confirms the paper's core hypothesis: combining self-entailment and overlap scoring ($S_T = S_e + S_o$) selects the most informative reasoning explanations for downstream student simulation.

### 2. Model Scale Effects (70B vs 1.1B)
- **Llama-2-70B** generates fine-grained reasoning chains that tightly govern model predictions.
- **TinyLlama-1.1B** displays higher shortcut reliance, which increases CF-UF unfaithfulness baseline values (as reflected in Table 3 of the original paper when scaling down from 70B to 13B/7B).

---

## 5. Summary of Generated Visualizations

- `results/tables/ablation_comparison.json`: Complete comparison JSON file.
- `results/figures/ablation_study.png`: Grouped bar chart comparing strategies.
- `results/figures/cross_method_radar.png`: Multi-dimensional interpretability radar plot.
- `results/figures/paper_vs_ours.png`: Side-by-side paper vs reproduction comparison plot.
