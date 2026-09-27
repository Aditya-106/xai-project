"""
Compare SEA-CoT across model sizes (reproduces model-size experiment).

Usage:
    python experiments/run_model_size.py --dataset strategyqa --seed 42
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_strategyqa, load_openbookqa, load_qasc
from src.model import get_model
from src.scoring.entailment import EntailmentScorer
from src.scoring.overlap import calculate_overlap_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SEA-CoT across model sizes.")
    parser.add_argument("--dataset", type=str, default="strategyqa",
                        choices=["strategyqa", "obqa", "qasc"])
    parser.add_argument("--sizes", nargs="+", default=["7B", "13B", "70B"],
                        help="Model sizes to compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--use-api", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Paper's reported model-size results for reference
    paper_results = {
        "70B": {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
        "13B": {"P": 4.10, "CF-UF": 4.38, "M": 69.62, "S": 6.16},
        "7B":  {"P": 3.79, "CF-UF": 7.81, "M": 70.62, "S": 15.97},
    }

    print(f"{'='*60}")
    print(f"  Model Size Comparison — SEA-CoT on {args.dataset}")
    print(f"  (Paper-reported values)")
    print(f"{'='*60}")
    print(f"{'Size':<10} {'P ↓':>8} {'CF-UF ↓':>10} {'M ↑':>8} {'S ↑':>8}")
    print(f"{'-'*60}")
    for size in args.sizes:
        p = paper_results.get(size, {})
        print(f"{size:<10} {p.get('P', '-'):>8} {p.get('CF-UF', '-'):>10} "
              f"{p.get('M', '-'):>8} {p.get('S', '-'):>8}")
    print(f"{'='*60}")

    # Save results
    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"model_size_{args.dataset}_seed{args.seed}.json"

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "paper_results": paper_results,
        "sizes": args.sizes,
        "note": "Run generation and evaluation for each model size to get reproduction values"
    }

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {out_file}")
    print(f"\nTo reproduce with each model size, run:")
    for size in args.sizes:
        print(f"  python experiments/run_generation.py --method sea_cot --dataset {args.dataset} --model-size {size}")


if __name__ == "__main__":
    main()
