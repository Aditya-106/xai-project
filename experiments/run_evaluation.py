"""
Run all evaluation metrics on generated reasoning chains.

Usage:
    python experiments/run_evaluation.py --dataset strategyqa --method sea_cot --seed 42
    python experiments/run_evaluation.py --dataset strategyqa --method cot --metrics paraphrase,mistake
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.evaluation.robustness import evaluate_paraphrase_robustness
from src.evaluation.faithfulness import evaluate_counterfactual, evaluate_mistake_sensitivity
from src.evaluation.utility import evaluate_las


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation metrics on generated explanations.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["strategyqa", "obqa", "qasc"])
    parser.add_argument("--method", type=str, required=True,
                        help="Prompting method (cot, sc_cot, sea_cot, etc.) or path to JSONL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", type=str, default="all",
                        help="Comma-separated metrics: paraphrase,counterfactual,mistake,las (or 'all')")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model-size", type=str, default="70B")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: str):
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    args = parse_args()

    # Determine which metrics to run
    if args.metrics == "all":
        metrics = ["paraphrase", "counterfactual", "mistake", "las"]
    else:
        metrics = [m.strip() for m in args.metrics.split(",")]

    # 1. Load generated explanations
    explanations_path = Path(args.data_dir) / str(args.seed) / args.method / f"{args.dataset}.jsonl"
    if not explanations_path.exists():
        print(f"ERROR: Explanations file not found: {explanations_path}")
        print(f"  Run: python experiments/run_generation.py --method {args.method} --dataset {args.dataset} --seed {args.seed}")
        sys.exit(1)

    explanations = load_jsonl(str(explanations_path))
    if args.max_samples:
        explanations = explanations[:args.max_samples]
    print(f"Loaded {len(explanations)} explanations from {explanations_path}")

    # 2. Load model for evaluation
    print(f"Loading model for evaluation...")
    size_num = args.model_size.replace("B", "")
    model = get_model(use_api=args.use_api, size=size_num, api_model=args.api_model)

    # 3. Run each metric
    results = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "num_examples": len(explanations),
        "metrics": {}
    }

    # --- Paraphrase Robustness (P ↓) ---
    if "paraphrase" in metrics:
        print("\n--- Evaluating Paraphrase Robustness (P) ---")
        perturbation_path = Path(args.data_dir) / str(args.seed) / args.method / "perturbations" / f"paraphrase_{args.dataset}.jsonl"
        if perturbation_path.exists():
            paraphrased = load_jsonl(str(perturbation_path))
            if args.max_samples:
                paraphrased = paraphrased[:args.max_samples]

            # Prepare data format for evaluation
            dataset_items = [{"question": e["question"], "choices": e["choices"]} for e in explanations]
            orig_exps = [{"explanation": e["reasoning"]} for e in explanations]
            para_exps = [{"paraphrased_explanation": p.get("paraphrased_explanation", p.get("reasoning", ""))}
                         for p in paraphrased]

            p_score = evaluate_paraphrase_robustness(model, dataset_items, orig_exps, para_exps)
            results["metrics"]["paraphrase_flip_rate"] = round(p_score * 100, 2)
            print(f"  Paraphrase Flip Rate (P): {results['metrics']['paraphrase_flip_rate']}%")
        else:
            print(f"  SKIPPED: Paraphrase perturbations not found at {perturbation_path}")
            print(f"  Run: python experiments/run_perturbation.py --type paraphrase --method {args.method} --dataset {args.dataset}")

    # --- Counterfactual Unfaithfulness (CF-UF ↓) ---
    if "counterfactual" in metrics:
        print("\n--- Evaluating Counterfactual Unfaithfulness (CF-UF) ---")
        cf_path = Path(args.data_dir) / str(args.seed) / "perturbations" / f"counterfactual_{args.dataset}.jsonl"
        if cf_path.exists():
            cf_data = load_jsonl(str(cf_path))
            if args.max_samples:
                cf_data = cf_data[:args.max_samples]

            dataset_items = [{"question": e["question"], "choices": e["choices"], "answer": e["gold_answer"]}
                             for e in explanations]
            cfuf_score = evaluate_counterfactual(model, dataset_items, cf_data)
            results["metrics"]["counterfactual_unfaithfulness"] = round(cfuf_score * 100, 2)
            print(f"  Counterfactual Unfaithfulness (CF-UF): {results['metrics']['counterfactual_unfaithfulness']}%")
        else:
            print(f"  SKIPPED: Counterfactual perturbations not found at {cf_path}")
            print(f"  Run: python experiments/run_perturbation.py --type counterfactual --dataset {args.dataset}")

    # --- Mistake Flip Rate (M ↑) ---
    if "mistake" in metrics:
        print("\n--- Evaluating Mistake Sensitivity (M) ---")
        mistake_path = Path(args.data_dir) / str(args.seed) / args.method / "perturbations" / f"mistake_{args.dataset}.jsonl"
        if mistake_path.exists():
            mistake_data = load_jsonl(str(mistake_path))
            if args.max_samples:
                mistake_data = mistake_data[:args.max_samples]

            dataset_items = [{"question": e["question"], "choices": e["choices"],
                              "explanation": e["reasoning"]} for e in explanations]
            m_score = evaluate_mistake_sensitivity(model, dataset_items, mistake_data)
            results["metrics"]["mistake_flip_rate"] = round(m_score * 100, 2)
            print(f"  Mistake Flip Rate (M): {results['metrics']['mistake_flip_rate']}%")
        else:
            print(f"  SKIPPED: Mistake perturbations not found at {mistake_path}")
            print(f"  Run: python experiments/run_perturbation.py --type mistake --method {args.method} --dataset {args.dataset}")

    # --- LAS / Simulatability (S ↑) ---
    if "las" in metrics:
        print("\n--- Evaluating Simulatability (LAS) ---")
        exp_texts = [e["reasoning"] for e in explanations]
        dataset_items = [{"question": e["question"], "choices": e["choices"], "answer": e["gold_answer"]}
                         for e in explanations]

        # Use first 80% as train, last 20% as test for LAS
        split_idx = int(len(dataset_items) * 0.8)
        train_data = dataset_items[:split_idx]
        test_data = dataset_items[split_idx:]
        test_explanations = exp_texts[split_idx:]

        las_score = evaluate_las(train_data, test_data, test_explanations)
        results["metrics"]["las"] = round(las_score * 100, 2)
        print(f"  LAS (Simulatability): {results['metrics']['las']}%")

    # 4. Print summary
    print(f"\n{'='*60}")
    print(f"  Evaluation Results: {args.method} on {args.dataset}")
    print(f"{'='*60}")
    for metric, value in results["metrics"].items():
        direction = "↓" if metric in ("paraphrase_flip_rate", "counterfactual_unfaithfulness") else "↑"
        print(f"  {metric:<35} {value:>8.2f}  ({direction} better)")
    print(f"{'='*60}")

    # 5. Save results
    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}_{args.method}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
