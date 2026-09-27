# Replication Summary

## Paper
**Title:** How Interpretable are Reasoning Explanations from Prompting Large Language Models?  
**Authors:** Yeo Wei Jie, Ranjan Satapathy, Rick Goh, Erik Cambria  
**Venue:** Findings of NAACL 2024  
**DOI:** 10.18653/v1/2024.findings-naacl.138

---

## 1. Objective

We reproduce key experiments from the above paper to verify its main claims:
- Chain-of-Thought (CoT) explanations from LLMs should be evaluated across multiple dimensions of interpretability (faithfulness, robustness, utility), not just faithfulness alone.
- The proposed Self-Entailment-Alignment Chain-of-Thought (SEA-CoT) method improves interpretability by more than 70% across multiple dimensions.

## 2. Original Paper Setup

| Component | Details |
|-----------|---------|
| **Model** | Llama-2-70B-chat-GPTQ (4-bit quantized) |
| **Datasets** | StrategyQA, OpenBookQA, QASC |
| **Prompting Methods** | CoT, SC-CoT, Question Decomposition, Self-Refine, SEA-CoT |
| **Evaluation Metrics** | Paraphrase flip rate (P↓), Counterfactual Unfaithfulness (CF-UF↓), Mistake flip rate (M↑), Simulatability/LAS (S↑) |
| **Perturbation Models** | GPT-3.5-turbo (paraphrase, mistakes), GPT-4 (counterfactuals) |
| **Student Model (LAS)** | T5-base |

## 3. Our Implementation

| Component | Details |
|-----------|---------|
| **Model** | [Specify model used] |
| **Datasets** | StrategyQA (primary), OpenBookQA, QASC |
| **Methods Implemented** | CoT, SC-CoT, SEA-CoT |
| **Hardware** | [Specify GPU/cloud used] |
| **Framework** | PyTorch + HuggingFace Transformers |

## 4. Experiments Reproduced

### Experiment 1: SEA-CoT Ablation (Paper Table 1)
Compares explanation selection strategies on StrategyQA:
- Random, Max probability, Overlap only, Entailment only, O&E (SEA-CoT)

### Experiment 2: Cross-Method Interpretability Comparison (Paper Figure 5)
Compares CoT, SC-CoT, and SEA-CoT across all interpretability metrics on StrategyQA.

### Experiment 3: Perturbation-Based Evaluation
Evaluates Paraphrase robustness and Counterfactual faithfulness for baseline vs SEA-CoT.

## 5. Results

### Paper vs Our Results (SEA-CoT Ablation on StrategyQA)

| Method | Paper P↓ | Ours P↓ | Paper CF-UF↓ | Ours CF-UF↓ | Paper M↑ | Ours M↑ | Paper S↑ | Ours S↑ |
|--------|----------|---------|-------------|-------------|----------|---------|----------|---------|
| Random | 6.10 | — | 6.44 | — | 62.17 | — | 11.87 | — |
| Max | 1.80 | — | 6.60 | — | 61.80 | — | 12.59 | — |
| Overlap | 1.56 | — | 5.04 | — | 70.83 | — | 14.88 | — |
| Entailment | 2.38 | — | 5.46 | — | 69.99 | — | 13.46 | — |
| **O&E (SEA-CoT)** | **1.20** | **—** | **3.81** | **—** | **61.24** | **—** | **16.97** | **—** |

*Fill in "Ours" columns after running experiments.*

## 6. Discussion

[To be filled after experiments]

- Differences in results and potential explanations
- Impact of model size / quantization
- Impact of dataset subset size
- Trends consistent with / divergent from paper

## 7. Conclusion

[To be filled after experiments]

- Successfully reproduced experiments
- Key findings confirmed/challenged
- Lessons learned
