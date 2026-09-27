"""
Main script to generate reasoning chains using different prompting methods.

Usage:
    python experiments/run_generation.py --method cot --dataset strategyqa --max-samples 50
    python experiments/run_generation.py --method sea_cot --dataset strategyqa --num-paths 5
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_strategyqa, load_openbookqa, load_qasc
from src.model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning chains.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["cot", "sc_cot", "sea_cot", "question_decomp", "self_refine"],
                        help="Prompting method to use")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["strategyqa", "obqa", "qasc"],
                        help="Dataset to evaluate on")
    parser.add_argument("--model-size", type=str, default="70B",
                        choices=["7B", "13B", "70B"],
                        help="Llama-2 model size")
    parser.add_argument("--num-paths", type=int, default=5,
                        help="Number of reasoning paths for SC-CoT/SEA-CoT")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of examples (for debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Base output directory")
    parser.add_argument("--use-api", action="store_true",
                        help="Use HuggingFace Inference API instead of local model")
    parser.add_argument("--api-model", type=str, default=None,
                        help="API model name (for --use-api)")
    return parser.parse_args()


def get_dataset(name: str):
    """Load a dataset by name."""
    if name == "strategyqa":
        return load_strategyqa()
    elif name == "obqa":
        return load_openbookqa()
    elif name == "qasc":
        return load_qasc()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_prompting_fn(method: str):
    """Import and return the appropriate prompting function."""
    if method == "cot":
        from src.prompting.cot import generate_cot
        return generate_cot
    elif method == "sc_cot":
        from src.prompting.sc_cot import generate_sc_cot
        return generate_sc_cot
    elif method == "sea_cot":
        from src.prompting.sea_cot import generate_sea_cot
        return generate_sea_cot
    elif method == "question_decomp":
        from src.prompting.question_decomp import generate_qd
        return generate_qd
    elif method == "self_refine":
        from src.prompting.self_refine import generate_self_refine
        return generate_self_refine
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # 1. Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    print(f"  Total examples: {len(dataset)}")

    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"  Using first {args.max_samples} examples")

    # 2. Load model
    print(f"Loading model: Llama-2-{args.model_size}")
    size_num = args.model_size.replace("B", "")
    model = get_model(
        use_api=args.use_api,
        size=size_num,
        api_model=args.api_model
    )
    print("  Model loaded successfully")

    # 3. Get prompting function
    prompting_fn = get_prompting_fn(args.method)

    # 4. Generate reasoning chains
    results = []
    all_paths = []  # Store all paths for SC-CoT/SEA-CoT (needed for ablation)
    correct = 0

    print(f"\nGenerating {args.method} reasoning chains...")
    for item in tqdm(dataset, desc=f"Generating ({args.method})"):
        try:
            # Build kwargs based on method
            if args.method in ("sc_cot", "sea_cot"):
                reasoning, predicted_answer = prompting_fn(
                    model=model,
                    question=item["question"],
                    choices=item["choices"],
                    dataset_type=args.dataset,
                    n_paths=args.num_paths
                )
            elif args.method in ("cot", "question_decomp", "self_refine"):
                reasoning, predicted_answer = prompting_fn(
                    model=model,
                    question=item["question"],
                    choices=item["choices"],
                    dataset_type=args.dataset
                )
            else:
                raise ValueError(f"Unsupported method: {args.method}")

            # Check correctness
            is_correct = predicted_answer.lower().strip() == item["answer"].lower().strip()
            if is_correct:
                correct += 1

            result = {
                "id": item["id"],
                "question": item["question"],
                "choices": item["choices"],
                "gold_answer": item["answer"],
                "predicted_answer": predicted_answer,
                "reasoning": reasoning,
                "method": args.method,
                "is_correct": is_correct
            }
            results.append(result)

        except Exception as e:
            print(f"\n  Error on example {item['id']}: {e}")
            results.append({
                "id": item["id"],
                "question": item["question"],
                "choices": item["choices"],
                "gold_answer": item["answer"],
                "predicted_answer": "",
                "reasoning": "",
                "method": args.method,
                "is_correct": False,
                "error": str(e)
            })

    # 5. Print summary
    accuracy = correct / len(dataset) * 100 if dataset else 0
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Method:   {args.method}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Model:    Llama-2-{args.model_size}")
    print(f"  Samples:  {len(results)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    # 6. Save results
    output_path = Path(args.output_dir) / str(args.seed) / args.method / f"{args.dataset}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
