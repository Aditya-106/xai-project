"""
SEA-CoT Ablation Study — reproduces Table 1 from the paper.

Compares explanation selection strategies on StrategyQA:
1. Random:      Randomly select one explanation
2. Max:         Select by highest probability (standard SC-CoT)
3. Overlap:     Select by highest overlap score S_o only
4. Entailment:  Select by highest entailment score S_e only
5. O&E:         Select by S_T = S_e + S_o (SEA-CoT)

Usage:
    python experiments/run_ablation.py --dataset strategyqa --seed 42
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_strategyqa, load_openbookqa, load_qasc
from src.model import get_model
from src.prompting.cot import STRATEGY_QA_TEMPLATE, OPENBOOK_QA_TEMPLATE, parse_cot_output
from src.scoring.entailment import EntailmentScorer
from src.scoring.overlap import calculate_overlap_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run SEA-CoT ablation study (Paper Table 1).")
    parser.add_argument("--dataset", type=str, default="strategyqa",
                        choices=["strategyqa", "obqa", "qasc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-paths", type=int, default=5,
                        help="Number of reasoning paths per question")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit examples for debugging")
    parser.add_argument("--model-size", type=str, default="70B",
                        choices=["7B", "13B", "70B"])
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--lightweight", action="store_true",
                        help="Use lightweight model (TinyLlama-1.1B) for CPU/Mac")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing pre-generated reasoning paths")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--use-cached-paths", action="store_true",
                        help="Use pre-generated paths from data dir instead of generating new ones")
    return parser.parse_args()


def get_dataset(name: str):
    """Load dataset by name."""
    loaders = {
        "strategyqa": load_strategyqa,
        "obqa": load_openbookqa,
        "qasc": load_qasc
    }
    return loaders[name]()


def build_prompt(question: str, choices: list, dataset_type: str) -> str:
    """Build a CoT prompt for a given question."""
    if dataset_type == "strategyqa":
        return STRATEGY_QA_TEMPLATE.format(question=question, choices=str(choices))
    elif len(choices) >= 4:
        return OPENBOOK_QA_TEMPLATE.format(
            question=question,
            choice_a=choices[0], choice_b=choices[1],
            choice_c=choices[2], choice_d=choices[3]
        )
    else:
        return STRATEGY_QA_TEMPLATE.format(question=question, choices=str(choices))


def generate_paths(model, question: str, choices: list, dataset_type: str, n_paths: int = 5):
    """Generate N reasoning paths for a question."""
    prompt = build_prompt(question, choices, dataset_type)

    # Generate N paths with temperature sampling
    outputs = model.generate(prompt, temperature=0.7, num_return_sequences=n_paths)

    paths = []
    answers = []
    for out in outputs:
        reasoning, ans = parse_cot_output(out, choices)
        paths.append(reasoning)
        answers.append(ans)

    return paths, answers


def select_explanation(paths, answers, question, majority_answer,
                       strategy, entailment_scorer=None):
    """
    Select an explanation from paths supporting the majority answer
    using different strategies.

    Strategies:
      - 'random':      Random selection
      - 'max':         First supporting path (simulates highest probability)
      - 'overlap':     Highest overlap score S_o
      - 'entailment':  Highest entailment score S_e
      - 'oe':          Highest S_T = S_e + S_o (SEA-CoT)
    """
    # Filter to paths supporting the majority answer
    supporting = [(p, a) for p, a in zip(paths, answers) if a == majority_answer]

    if not supporting:
        # Fallback: use first path
        return paths[0] if paths else "", {}

    hypothesis = f"{question} {majority_answer}"

    if strategy == "random":
        idx = random.randint(0, len(supporting) - 1)
        return supporting[idx][0], {"strategy": "random", "index": idx}

    elif strategy == "max":
        # In the original paper, "Max" selects by highest cumulative probability.
        # Without access to logprobs, we approximate by taking the first path.
        return supporting[0][0], {"strategy": "max"}

    elif strategy == "overlap":
        best_path = ""
        best_score = -1.0
        scores = []
        for path, _ in supporting:
            s_o = calculate_overlap_score(path, hypothesis)
            scores.append(s_o)
            if s_o > best_score:
                best_score = s_o
                best_path = path
        return best_path, {"strategy": "overlap", "s_o": best_score, "all_scores": scores}

    elif strategy == "entailment":
        best_path = ""
        best_score = -1.0
        scores = []
        for path, _ in supporting:
            s_e = entailment_scorer.calculate_entailment_score(path, hypothesis)
            scores.append(s_e)
            if s_e > best_score:
                best_score = s_e
                best_path = path
        return best_path, {"strategy": "entailment", "s_e": best_score, "all_scores": scores}

    elif strategy == "oe":
        best_path = ""
        best_score = -1.0
        scores = []
        for path, _ in supporting:
            s_e = entailment_scorer.calculate_entailment_score(path, hypothesis)
            s_o = calculate_overlap_score(path, hypothesis)
            s_t = s_e + s_o
            scores.append({"s_e": s_e, "s_o": s_o, "s_t": s_t})
            if s_t > best_score:
                best_score = s_t
                best_path = path
        return best_path, {"strategy": "oe", "s_t": best_score, "all_scores": scores}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    args = parse_args()
    random.seed(args.seed)

    # --- Paper's reported results for StrategyQA (Table 1) ---
    paper_results = {
        "Random":     {"P": 6.10, "CF-UF": 6.44, "M": 62.17, "S": 11.87},
        "Max":        {"P": 1.80, "CF-UF": 6.60, "M": 61.80, "S": 12.59},
        "Overlap":    {"P": 1.56, "CF-UF": 5.04, "M": 70.83, "S": 14.88},
        "Entailment": {"P": 2.38, "CF-UF": 5.46, "M": 69.99, "S": 13.46},
        "O&E":        {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
    }

    # 1. Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    print(f"  Using {len(dataset)} examples")

    # 2. Load model
    print(f"Loading model...")
    size_num = args.model_size.replace("B", "")
    model = get_model(use_api=args.use_api, size=size_num, api_model=args.api_model,
                      lightweight=args.lightweight)

    # 3. Load entailment scorer
    print("Loading entailment scorer...")
    entailment_scorer = EntailmentScorer()

    # 4. Strategies to compare
    strategies = {
        "Random": "random",
        "Max": "max",
        "Overlap": "overlap",
        "Entailment": "entailment",
        "O&E": "oe"
    }

    # 5. Generate paths for all examples (shared across strategies)
    print(f"\nGenerating {args.num_paths} reasoning paths per question...")
    all_example_paths = []
    for item in tqdm(dataset, desc="Generating paths"):
        try:
            paths, answers = generate_paths(
                model, item["question"], item["choices"],
                args.dataset, args.num_paths
            )
            all_example_paths.append({
                "item": item,
                "paths": paths,
                "answers": answers,
                "majority_answer": Counter(answers).most_common(1)[0][0] if answers else ""
            })
        except Exception as e:
            print(f"\n  Error on {item['id']}: {e}")
            all_example_paths.append({
                "item": item,
                "paths": [],
                "answers": [],
                "majority_answer": ""
            })

    # 6. Apply each selection strategy and collect selected explanations
    print("\nApplying selection strategies...")
    strategy_selections = {}

    for name, strategy in strategies.items():
        print(f"  Strategy: {name}")
        selections = []
        for ep in all_example_paths:
            if not ep["paths"]:
                selections.append({"reasoning": "", "answer": ""})
                continue

            selected_reasoning, meta = select_explanation(
                ep["paths"], ep["answers"],
                ep["item"]["question"], ep["majority_answer"],
                strategy, entailment_scorer
            )
            selections.append({
                "id": ep["item"]["id"],
                "question": ep["item"]["question"],
                "gold_answer": ep["item"]["answer"],
                "predicted_answer": ep["majority_answer"],
                "reasoning": selected_reasoning,
                "strategy": name,
                "meta": meta
            })
        strategy_selections[name] = selections

    # 7. Save selections for each strategy
    for name, selections in strategy_selections.items():
        out_path = Path(args.output_dir) / "ablation" / args.dataset / f"{name.lower().replace('&', 'and')}_seed{args.seed}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for s in selections:
                f.write(json.dumps(s) + "\n")

    # 8. Print comparison table
    print(f"\n{'='*70}")
    print(f"  SEA-CoT Ablation Results — {args.dataset}")
    print(f"  (Paper Table 1 comparison)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'P ↓':>8} {'CF-UF ↓':>10} {'M ↑':>8} {'S ↑':>8}")
    print(f"{'-'*70}")

    for name in strategies:
        paper = paper_results.get(name, {})
        print(f"{name:<15} {paper.get('P', '-'):>8} {paper.get('CF-UF', '-'):>10} "
              f"{paper.get('M', '-'):>8} {paper.get('S', '-'):>8}  (paper)")

    print(f"{'-'*70}")
    print(f"  NOTE: Run experiments/run_evaluation.py on each strategy's output")
    print(f"  to get your reproduced metrics for comparison.")
    print(f"{'='*70}")

    # 9. Save ablation summary
    out_file = Path(args.output_dir) / "tables" / f"ablation_{args.dataset}_seed{args.seed}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "num_paths": args.num_paths,
        "model_size": args.model_size,
        "num_examples": len(dataset),
        "paper_results": paper_results,
        "strategies_applied": list(strategies.keys()),
        "output_paths": {
            name: str(Path(args.output_dir) / "ablation" / args.dataset /
                       f"{name.lower().replace('&', 'and')}_seed{args.seed}.jsonl")
            for name in strategies
        }
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAblation summary saved to: {out_file}")
    print(f"Strategy selections saved to: {Path(args.output_dir) / 'ablation' / args.dataset}/")


if __name__ == "__main__":
    main()
