#!/usr/bin/env python3
"""
run_all.py — End-to-End Pipeline Runner (Paper Replication)
=====================================================================
Reproduces key experiments from the paper:
  "How Interpretable are Reasoning Explanations from Prompting
   Large Language Models?" — Yeo et al., NAACL 2024 Findings

Ablation Strategies Evaluated (Table 1):
  1. Random: Randomly select one candidate explanation
  2. Max: Select candidate explanation with highest log probability (SC-CoT)
  3. Overlap: Select candidate by highest token IoU overlap S_o
  4. Entailment: Select candidate by highest DeBERTa NLI score S_e
  5. O&E (SEA-CoT): Select candidate by highest S_T = S_e + S_o

Evaluation Metrics:
  - Paraphrase flip rate (P ↓)
  - Counterfactual Unfaithfulness (CF-UF ↓)
  - Mistake flip rate (M ↑)
  - Simulatability / LAS (S ↑)
"""

import json
import os
import sys
import random
import re
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer,
)

# ═══════════════════════════════════════════════════════════════
# PAPER'S REPORTED RESULTS (from Table 1, NAACL 2024)
# ═══════════════════════════════════════════════════════════════

PAPER_ABLATION = {
    "Random":     {"P": 6.10, "CF-UF": 6.44, "M": 62.17, "S": 11.87},
    "Max":        {"P": 1.80, "CF-UF": 6.60, "M": 61.80, "S": 12.59},
    "Overlap":    {"P": 1.56, "CF-UF": 5.04, "M": 70.83, "S": 14.88},
    "Entailment": {"P": 2.38, "CF-UF": 5.46, "M": 69.99, "S": 13.46},
    "O&E (SEA)":  {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
}

PAPER_CROSS_METHOD = {
    "CoT":         {"P": 3.41, "CF-UF": 6.54, "M": 57.39, "S": 9.27},
    "SC-CoT":      {"P": 1.80, "CF-UF": 6.60, "M": 61.80, "S": 12.59},
    "QD":          {"P": 5.12, "CF-UF": 7.86, "M": 54.85, "S": 7.78},
    "Self-Refine": {"P": 3.90, "CF-UF": 6.82, "M": 60.05, "S": 10.50},
    "SEA-CoT":     {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
}


# ═══════════════════════════════════════════════════════════════
# 1. MODEL CLASSES
# ═══════════════════════════════════════════════════════════════

def parse_answer(text: str) -> str:
    """Extract yes/no answer cleanly using regex."""
    text_lower = text.lower()
    match = re.search(r'answer:\s*(yes|no)', text_lower)
    if match:
        return match.group(1)
    if text_lower.startswith('yes'):
        return 'yes'
    if text_lower.startswith('no'):
        return 'no'
    if re.search(r'\byes\b', text_lower):
        return 'yes'
    if re.search(r'\bno\b', text_lower):
        return 'no'
    return 'yes'


class LightweightLM:
    """LLM wrapper supporting chat templates & sequence log-probabilities."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"  Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  LLM loaded on {self.device} ({param_count:.1f}M params)")

    def generate_cot(self, question: str, max_new_tokens: int = 120, temperature: float = 0.7) -> dict:
        """Generate a single CoT path with sequence log probability."""
        messages = [
            {"role": "system", "content": "You are a precise reasoning assistant. Think step by step and end your response with 'Answer: Yes' or 'Answer: No'."},
            {"role": "user", "content": f"Question: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 0.01) if temperature > 0.0 else 1.0,
                top_p=0.9 if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        text = self.tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
        try:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            avg_log_prob = float(transition_scores[0].mean().cpu().numpy()) if len(transition_scores[0]) > 0 else -10.0
        except Exception:
            avg_log_prob = -10.0

        ans = parse_answer(text)
        return {"text": text, "log_prob": avg_log_prob, "answer": ans}

    def predict_given_reasoning(self, reasoning: str, question: str) -> str:
        """Predict yes/no given a specific reasoning chain in prompt context."""
        messages = [
            {"role": "system", "content": "Based ONLY on the provided reasoning, answer the question with 'Answer: Yes' or 'Answer: No'."},
            {"role": "user", "content": f"Reasoning: {reasoning}\nQuestion: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=25, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return parse_answer(text)


class NLIScorer:
    """NLI entailment scorer automatically mapping label indices."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        print(f"  Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", {})
        self.entailment_idx = 1
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                self.entailment_idx = int(idx)
                break
        print(f"  NLI model loaded (entailment_label_idx={self.entailment_idx})")

    def score(self, premise: str, hypothesis: str) -> float:
        if not premise.strip() or not hypothesis.strip():
            return 0.0
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        return probs[0][self.entailment_idx].item()


class T5StudentModel:
    """T5 student model for Leakage-Adjusted Simulatability (LAS)."""

    def __init__(self, model_name: str = "google-t5/t5-small"):
        print(f"  Loading T5 student: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"  T5 student loaded")

    def predict_probs(self, prompt: str, targets: list = ["yes", "no"]) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        probs = {}
        for target in targets:
            labels = self.tokenizer(target, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss.item()
                probs[target] = float(np.exp(-loss))
        total = sum(probs.values())
        if total > 0:
            return {k: v / total for k, v in probs.items()}
        return {k: 0.5 for k in targets}


# ═══════════════════════════════════════════════════════════════
# 2. SCORING & PERTURBATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def calculate_overlap_score(explanation: str, question_answer: str) -> float:
    """IoU token overlap between explanation and question+answer hypothesis."""
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'and',
                  'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                  'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                  'than', 'too', 'very', 'just', 'because', 'if', 'it', 'its',
                  'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
                  'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them'}
    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(t for t in tokens if t not in stop_words)

    set_e = tokenize(explanation)
    set_qa = tokenize(question_answer)
    if not set_e or not set_qa:
        return 0.0
    intersection = set_e & set_qa
    union = set_e | set_qa
    return len(intersection) / len(union) if union else 0.0


def generate_paraphrase(exp: str) -> str:
    """Generate lexical & syntactic paraphrase of explanation."""
    replacements = [
        (r'\bTherefore\b', 'Consequently'),
        (r'\bthus\b', 'hence'),
        (r'\bbecause\b', 'since'),
        (r'\bshows that\b', 'demonstrates that'),
        (r'\bimportant\b', 'vital'),
        (r'\blarge\b', 'substantial'),
        (r'\bsmall\b', 'modest'),
    ]
    res = exp
    for pattern, rep in replacements:
        res = re.sub(pattern, rep, res, flags=re.IGNORECASE)
    return res


def generate_mistake(exp: str) -> str:
    """Insert a logical mistake/contradiction into explanation."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', exp) if s.strip()]
    if not sents:
        return "However, the opposite conclusion is true. " + exp
    mid = len(sents) // 2
    sents[mid] = "However, it is actually false that " + sents[mid].lower()
    return '. '.join(sents)


def generate_counterfactual(question: str, orig_ans: str) -> tuple:
    """Generate counterfactual question flipping the answer condition."""
    cf_ans = 'no' if orig_ans.lower() == 'yes' else 'yes'
    cf_q = "Is it NOT true that " + question[0].lower() + question[1:] if question else question
    return cf_q, cf_ans


# ═══════════════════════════════════════════════════════════════
# 3. ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_trend_agreement(paper_results: dict, our_results: dict, metric: str) -> dict:
    """Compute Spearman rank correlation between paper and our strategy rankings."""
    strategies = list(paper_results.keys())
    paper_vals = [paper_results[s][metric] for s in strategies]
    our_vals = [our_results[s][metric] for s in strategies]

    paper_ranks = np.argsort(np.argsort(paper_vals))
    our_ranks = np.argsort(np.argsort(our_vals))

    n = len(strategies)
    d_sq = sum((float(p) - float(o)) ** 2 for p, o in zip(paper_ranks, our_ranks))
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1)) if n > 1 else 1.0

    if metric in ("P", "CF-UF"):
        paper_best = strategies[np.argmin(paper_vals)]
        our_best = strategies[np.argmin(our_vals)]
    else:
        paper_best = strategies[np.argmax(paper_vals)]
        our_best = strategies[np.argmax(our_vals)]

    return {
        "spearman_rho": round(rho, 3),
        "paper_best": paper_best,
        "our_best": our_best,
        "best_matches": paper_best == our_best,
    }


# ═══════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run complete reproduction pipeline")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--n-paths", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--nli-model", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--t5-model", type=str, default="google-t5/t5-small")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CoT Interpretability Reproduction — Full Pipeline Run     ║")
    print("║  Paper: Yeo et al., NAACL 2024 Findings                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ─── 1. Load Dataset ───
    print("\n[1/6] Loading StrategyQA dataset...")
    data_path = Path("data/strategyqa/processed.jsonl")
    if not data_path.exists():
        print("  Downloading StrategyQA dataset...")
        os.system(f"{sys.executable} src/data_loader.py --download --dataset strategyqa")

    dataset = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    dataset = dataset[:args.max_samples]
    print(f"  Loaded {len(dataset)} examples")

    # ─── 2. Load Models ───
    print("\n[2/6] Loading models...")
    lm = LightweightLM(args.model_name)
    nli = NLIScorer(args.nli_model)
    t5_student = T5StudentModel(args.t5_model)

    # ─── 3. Generate Reasoning Chains & Compute Scores ───
    print(f"\n[3/6] Generating reasoning chains ({args.n_paths} paths per example)...")
    all_data = []
    t0 = time.time()

    for item in tqdm(dataset, desc="  Generating"):
        paths = []
        for _ in range(args.n_paths):
            path_dict = lm.generate_cot(item["question"])
            paths.append(path_dict)

        answers = [p["answer"] for p in paths]
        majority = Counter(answers).most_common(1)[0][0]
        hypothesis = f"Question: {item['question']} Answer: {majority}"

        scored_paths = []
        for p in paths:
            if p["answer"] == majority:
                s_e = nli.score(p["text"], hypothesis)
                s_o = calculate_overlap_score(p["text"], hypothesis)
                s_t = s_e + s_o
            else:
                s_e, s_o, s_t = 0.0, 0.0, -10.0
            scored_paths.append({**p, "s_e": round(s_e, 4), "s_o": round(s_o, 4), "s_t": round(s_t, 4)})

        all_data.append({
            "item": item,
            "majority": majority,
            "paths": scored_paths,
            "is_correct": majority.lower() == item["answer"].lower()
        })

    gen_time = time.time() - t0
    acc = sum(d["is_correct"] for d in all_data) / len(all_data) * 100
    print(f"  Done in {gen_time:.1f}s | Majority-vote accuracy: {acc:.1f}%")

    # ─── 4. Run Ablation Strategies & Interpretability Evaluation ───
    print("\n[4/6] Running SEA-CoT ablation & evaluating interpretability metrics...")
    strategies = {
        "Random": "random",
        "Max": "max",
        "Overlap": "overlap",
        "Entailment": "entailment",
        "O&E (SEA)": "oe"
    }

    our_results = {}

    for strat_name, strat_key in strategies.items():
        explanations = []
        for d in all_data:
            supp = [p for p in d["paths"] if p["answer"] == d["majority"]]
            if not supp:
                supp = d["paths"]

            if strat_key == "random":
                chosen = random.choice(supp)
            elif strat_key == "max":
                chosen = max(supp, key=lambda x: x["log_prob"])
            elif strat_key == "overlap":
                chosen = max(supp, key=lambda x: x["s_o"])
            elif strat_key == "entailment":
                chosen = max(supp, key=lambda x: x["s_e"])
            elif strat_key == "oe":
                chosen = max(supp, key=lambda x: x["s_t"])

            explanations.append(chosen["text"])

        # Metric 1: P ↓ (Paraphrase flip rate)
        p_flips = 0
        for d, exp in zip(all_data, explanations):
            pred1 = lm.predict_given_reasoning(exp, d["item"]["question"])
            para = generate_paraphrase(exp)
            pred2 = lm.predict_given_reasoning(para, d["item"]["question"])
            if pred1 != pred2:
                p_flips += 1
        p_val = (p_flips / len(all_data)) * 100

        # Metric 2: CF-UF ↓ (Counterfactual Unfaithfulness)
        cf_unfaithful = 0
        for d, exp in zip(all_data, explanations):
            cf_q, cf_ans = generate_counterfactual(d["item"]["question"], d["majority"])
            pred_cf = lm.predict_given_reasoning(exp, cf_q)
            if pred_cf == cf_ans:
                cf_unfaithful += 1
        cf_val = (cf_unfaithful / len(all_data)) * 100

        # Metric 3: M ↑ (Mistake flip rate)
        m_flips = 0
        for d, exp in zip(all_data, explanations):
            mistake_exp = generate_mistake(exp)
            pred_m = lm.predict_given_reasoning(mistake_exp, d["item"]["question"])
            if pred_m != d["majority"]:
                m_flips += 1
        m_val = (m_flips / len(all_data)) * 100

        # Metric 4: S ↑ (Simulatability / LAS)
        las_diffs = []
        for d, exp in zip(all_data, explanations):
            q = d["item"]["question"]
            gold = d["item"]["answer"].lower()
            p_wout = t5_student.predict_probs(f"question: {q} answer yes or no:")
            p_with = t5_student.predict_probs(f"explanation: {exp} question: {q} answer yes or no:")
            diff = (p_with.get(gold, 0.5) - p_wout.get(gold, 0.5)) * 100
            las_diffs.append(diff)
        s_val = float(np.mean(las_diffs))

        our_results[strat_name] = {
            "P": round(p_val, 2),
            "CF-UF": round(cf_val, 2),
            "M": round(m_val, 2),
            "S": round(s_val, 2)
        }
        print(f"  → Strategy {strat_name:12s}: P={p_val:5.2f}↓  CF-UF={cf_val:5.2f}↓  M={m_val:5.2f}↑  S={s_val:5.2f}↑")

    # ─── 5. Compare with Paper & Compute Spearman Rank Correlation ───
    print("\n[5/6] Comparing results with original paper (Table 1)...")
    trend_analysis = {}
    for metric in ["P", "CF-UF", "M", "S"]:
        trend_analysis[metric] = compute_trend_agreement(PAPER_ABLATION, our_results, metric)

    print("\n" + "="*80)
    print(f"{'Strategy':<12} | {'Paper P↓':<9} {'Ours P↓':<9} | {'Paper CF↓':<9} {'Ours CF↓':<9} | {'Paper M↑':<9} {'Ours M↑':<9} | {'Paper S↑':<9} {'Ours S↑':<9}")
    print("-" * 80)
    for s in PAPER_ABLATION:
        p_p, p_o = PAPER_ABLATION[s]["P"], our_results[s]["P"]
        cf_p, cf_o = PAPER_ABLATION[s]["CF-UF"], our_results[s]["CF-UF"]
        m_p, m_o = PAPER_ABLATION[s]["M"], our_results[s]["M"]
        s_p, s_o = PAPER_ABLATION[s]["S"], our_results[s]["S"]
        print(f"{s:<12} | {p_p:<9.2f} {p_o:<9.2f} | {cf_p:<9.2f} {cf_o:<9.2f} | {m_p:<9.2f} {m_o:<9.2f} | {s_p:<9.2f} {s_o:<9.2f}")
    print("="*80)

    print("\nTREND CORRELATION ANALYSIS (Spearman ρ):")
    for m, info in trend_analysis.items():
        print(f"  Metric {m:5s}: Spearman ρ = {info['spearman_rho']:+.3f} | Paper Best: {info['paper_best']:10s} | Our Best: {info['our_best']:10s}")

    # ─── 6. Save Table & Update Files ───
    print("\n[6/6] Saving outputs...")
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "paper": PAPER_ABLATION,
        "ours": our_results,
        "config": {
            "model": args.model_name,
            "nli_model": args.nli_model,
            "t5_model": args.t5_model,
            "n_samples": len(dataset),
            "n_paths": args.n_paths,
            "seed": args.seed,
            "accuracy": round(acc, 2)
        },
        "trend_analysis": trend_analysis
    }

    with open(out_dir / "ablation_comparison.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"  Saved comparison JSON to {out_dir / 'ablation_comparison.json'}")
    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()
