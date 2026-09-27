"""
Generate perturbations (paraphrase, counterfactual, mistake) for evaluation.

Usage:
    python experiments/run_perturbation.py --type paraphrase --method cot --dataset strategyqa --seed 42
    python experiments/run_perturbation.py --type counterfactual --dataset strategyqa --seed 42
    python experiments/run_perturbation.py --type mistake --method sea_cot --dataset strategyqa --seed 42
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perturbation.paraphrase import generate_paraphrase
from src.perturbation.counterfactual import generate_counterfactual
from src.perturbation.mistake import generate_mistake
from src.data_loader import load_strategyqa, load_openbookqa, load_qasc


def parse_args():
    parser = argparse.ArgumentParser(description="Generate perturbations for evaluation.")
    parser.add_argument("--type", type=str, required=True,
                        choices=["paraphrase", "counterfactual", "mistake"],
                        help="Type of perturbation to generate")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["strategyqa", "obqa", "qasc"])
    parser.add_argument("--method", type=str, default="cot",
                        help="Prompting method whose explanations to perturb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
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

    # Set API key if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key

    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Perturbation generation requires an OpenAI API key.")
        print("  Set via: export OPENAI_API_KEY='your-key' or --openai-key 'your-key'")
        sys.exit(1)

    # For paraphrase/mistake: load method-specific explanations
    # For counterfactual: load original dataset (shared across methods)
    if args.type in ("paraphrase", "mistake"):
        explanations_path = Path(args.data_dir) / str(args.seed) / args.method / f"{args.dataset}.jsonl"
        if not explanations_path.exists():
            print(f"ERROR: Explanations not found: {explanations_path}")
            print(f"  Run: python experiments/run_generation.py --method {args.method} --dataset {args.dataset}")
            sys.exit(1)
        data = load_jsonl(str(explanations_path))
    else:
        # Counterfactual: generate from original dataset
        loaders = {"strategyqa": load_strategyqa, "obqa": load_openbookqa, "qasc": load_qasc}
        data = loaders[args.dataset]()

    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Processing {len(data)} examples for {args.type} perturbation...")

    results = []

    if args.type == "paraphrase":
        # Generate paraphrased explanations
        out_dir = Path(args.data_dir) / str(args.seed) / args.method / "perturbations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"paraphrase_{args.dataset}.jsonl"

        for item in tqdm(data, desc="Paraphrasing"):
            try:
                paraphrased = generate_paraphrase(item["reasoning"])
                result = {**item, "paraphrased_explanation": paraphrased}
                results.append(result)
            except Exception as e:
                print(f"\n  Error: {e}")
                results.append({**item, "paraphrased_explanation": item["reasoning"]})

    elif args.type == "counterfactual":
        # Generate counterfactual questions (shared across methods)
        out_dir = Path(args.data_dir) / str(args.seed) / "perturbations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"counterfactual_{args.dataset}.jsonl"

        is_binary = args.dataset == "strategyqa"
        for item in tqdm(data, desc="Generating counterfactuals"):
            try:
                answer = item.get("answer", item.get("gold_answer", ""))
                if is_binary:
                    cf_answer = "no" if answer.lower() == "yes" else "yes"
                else:
                    choices = item["choices"]
                    cf_answer = next(c for c in choices if c != answer)

                cf_question = generate_counterfactual(item["question"], answer, cf_answer)
                result = {
                    **item,
                    "counterfactual_question": cf_question,
                    "counterfactual_answer": cf_answer
                }
                results.append(result)
            except Exception as e:
                print(f"\n  Error: {e}")
                results.append({**item, "counterfactual_question": item["question"],
                                "counterfactual_answer": ""})

    elif args.type == "mistake":
        # Insert mistakes into explanations
        out_dir = Path(args.data_dir) / str(args.seed) / args.method / "perturbations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mistake_{args.dataset}.jsonl"

        for item in tqdm(data, desc="Inserting mistakes"):
            try:
                mistake_exp = generate_mistake(item["reasoning"])
                result = {**item, "mistake_explanation": mistake_exp}
                results.append(result)
            except Exception as e:
                print(f"\n  Error: {e}")
                results.append({**item, "mistake_explanation": item["reasoning"]})

    # Save results
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} perturbations to: {out_path}")


if __name__ == "__main__":
    main()
