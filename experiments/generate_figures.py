"""
Generate publication-quality comparison figures for the reproduction.

Creates 3 figures:
1. SEA-CoT Ablation Bar Chart (Paper Table 1)
2. Cross-Method Interpretability Comparison (Paper Figure 5)
3. Paper vs Our Results Comparison

Usage:
    python experiments/generate_figures.py
    python experiments/generate_figures.py --results-dir results/tables --output-dir results/figures
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12

# ────────────────────────────────────────────────────────
# Paper's reported results (from Table 1 and Figure 5)
# ────────────────────────────────────────────────────────

# Table 1: SEA-CoT Ablation on StrategyQA
PAPER_ABLATION = {
    "Random":     {"P": 6.10, "CF-UF": 6.44, "M": 62.17, "S": 11.87},
    "Max":        {"P": 1.80, "CF-UF": 6.60, "M": 61.80, "S": 12.59},
    "Overlap":    {"P": 1.56, "CF-UF": 5.04, "M": 70.83, "S": 14.88},
    "Entailment": {"P": 2.38, "CF-UF": 5.46, "M": 69.99, "S": 13.46},
    "O&E (SEA)":  {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
}

# Table 3/Figure 5: Cross-method on StrategyQA (from paper)
PAPER_CROSS_METHOD = {
    "CoT":         {"P": 3.41, "CF-UF": 6.54, "M": 57.39, "S": 9.27},
    "SC-CoT":      {"P": 1.80, "CF-UF": 6.60, "M": 61.80, "S": 12.59},
    "QD":          {"P": 5.12, "CF-UF": 7.86, "M": 54.85, "S": 7.78},
    "Self-Refine": {"P": 3.90, "CF-UF": 6.82, "M": 60.05, "S": 10.50},
    "SEA-CoT":     {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
}

# Table: Model size comparison on StrategyQA
PAPER_MODEL_SIZE = {
    "70B": {"P": 1.20, "CF-UF": 3.81, "M": 61.24, "S": 16.97},
    "13B": {"P": 4.10, "CF-UF": 4.38, "M": 69.62, "S": 6.16},
    "7B":  {"P": 3.79, "CF-UF": 7.81, "M": 70.62, "S": 15.97},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-quality figures.")
    parser.add_argument("--results-dir", type=str, default="results/tables",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Directory to save figures")
    parser.add_argument("--use-paper-data", action="store_true", default=True,
                        help="Use paper's reported data (default: True)")
    return parser.parse_args()


def load_our_results(results_dir: str, pattern: str) -> dict:
    """Try to load our reproduced results from JSON files."""
    results = {}
    results_path = Path(results_dir)
    if results_path.exists():
        for f in results_path.glob(pattern):
            with open(f, "r") as fp:
                data = json.load(fp)
                if "method" in data and "metrics" in data:
                    results[data["method"]] = data["metrics"]
    return results


def figure_1_ablation(output_dir: str, paper_data: dict, our_data: dict = None):
    """
    Figure 1: SEA-CoT Ablation Study (reproduces Paper Table 1).
    Grouped bar chart comparing selection strategies.
    """
    strategies = list(paper_data.keys())
    metrics = ["P", "CF-UF", "M", "S"]
    metric_labels = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("SEA-CoT Ablation Study on StrategyQA\n(Reproducing Paper Table 1)", fontsize=14, fontweight='bold')

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [paper_data[s][metric] for s in strategies]

        bars = ax.bar(range(len(strategies)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([s.replace(" (SEA)", "\n(SEA)") for s in strategies],
                           rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Score")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        # Highlight best
        if metric in ("P", "CF-UF"):  # Lower is better
            best_idx = values.index(min(values))
        else:  # Higher is better
            best_idx = values.index(max(values))
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig1_ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 1: Ablation Study saved")


def figure_2_cross_method(output_dir: str, paper_data: dict, our_data: dict = None):
    """
    Figure 2: Cross-Method Interpretability Comparison (reproduces Paper Figure 5).
    Radar/spider chart showing normalized interpretability scores.
    """
    methods = list(paper_data.keys())
    metrics = ["P", "CF-UF", "M", "S"]

    # Normalize scores to [0, 1] for radar chart
    # For P and CF-UF: lower is better, so invert
    all_values = {m: [] for m in metrics}
    for method in methods:
        for metric in metrics:
            all_values[metric].append(paper_data[method][metric])

    normalized = {}
    for method in methods:
        normalized[method] = []
        for metric in metrics:
            vals = all_values[metric]
            min_v, max_v = min(vals), max(vals)
            raw = paper_data[method][metric]
            if max_v == min_v:
                norm = 1.0
            elif metric in ("P", "CF-UF"):
                # Lower is better → invert
                norm = 1 - (raw - min_v) / (max_v - min_v)
            else:
                # Higher is better
                norm = (raw - min_v) / (max_v - min_v)
            normalized[method].append(norm)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title("Cross-Method Interpretability Comparison\n(Normalized Scores, Higher = Better)",
                 fontsize=14, fontweight='bold', pad=20)

    colors_map = {'CoT': '#4C72B0', 'SC-CoT': '#55A868', 'QD': '#C44E52',
                  'Self-Refine': '#8172B2', 'SEA-CoT': '#CCB974'}

    for method in methods:
        values = normalized[method] + normalized[method][:1]
        color = colors_map.get(method, '#333333')
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["P ↓\n(lower=better)", "CF-UF ↓\n(lower=better)",
                         "M ↑\n(higher=better)", "S ↑\n(higher=better)"])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig2_cross_method_radar.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also create a grouped bar chart version
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Cross-Method Comparison on StrategyQA\n(Reproducing Paper Figure 5)",
                 fontsize=14, fontweight='bold')

    metric_labels = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]
    x = np.arange(len(methods))
    colors_list = [colors_map.get(m, '#333') for m in methods]

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [paper_data[m][metric] for m in methods]
        bars = ax.bar(x, values, color=colors_list, edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig2_cross_method_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 2: Cross-Method Comparison saved (radar + bars)")


def figure_3_paper_comparison(output_dir: str, paper_data: dict, our_data: dict):
    """
    Figure 3: Paper vs Our Results side-by-side comparison.
    """
    # Compare SEA-CoT results
    metrics = ["P", "CF-UF", "M", "S"]
    metric_labels = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]

    paper_vals = [paper_data.get("O&E (SEA)", paper_data.get("SEA-CoT", {})).get(m, 0) for m in metrics]

    if our_data:
        our_vals = [our_data.get(m, 0) for m in metrics]
    else:
        # Placeholder — will be filled after running experiments
        our_vals = [0] * len(metrics)

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, paper_vals, width, label='Paper (Reported)',
                   color='#4C72B0', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, our_vals, width, label='Our Reproduction',
                   color='#C44E52', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score')
    ax.set_title('Paper vs Our Reproduction — SEA-CoT on StrategyQA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig3_paper_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 3: Paper vs Ours Comparison saved")


def figure_4_model_size(output_dir: str, paper_data: dict):
    """
    Figure 4: Model Size Comparison for SEA-CoT on StrategyQA.
    """
    sizes = list(paper_data.keys())
    metrics = ["P", "CF-UF", "M", "S"]
    metric_labels = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("SEA-CoT: Model Size Comparison on StrategyQA", fontsize=14, fontweight='bold')

    colors = ['#4C72B0', '#55A868', '#C44E52']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [paper_data[s][metric] for s in sizes]
        bars = ax.bar(range(len(sizes)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"Llama-2-{s}" for s in sizes], fontsize=9)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig4_model_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 4: Model Size Comparison saved")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load our reproduced results
    our_results = load_our_results(args.results_dir, "*.json")

    print("Generating publication-quality figures...")
    print(f"  Output directory: {out_dir}")

    # Figure 1: Ablation
    figure_1_ablation(str(out_dir), PAPER_ABLATION)

    # Figure 2: Cross-Method
    figure_2_cross_method(str(out_dir), PAPER_CROSS_METHOD)

    # Figure 3: Paper vs Ours
    our_sea_cot = our_results.get("sea_cot", {})
    figure_3_paper_comparison(str(out_dir), PAPER_CROSS_METHOD, our_sea_cot)

    # Figure 4: Model Size
    figure_4_model_size(str(out_dir), PAPER_MODEL_SIZE)

    print(f"\nAll figures saved to: {out_dir}/")
    print("  - fig1_ablation_study.png")
    print("  - fig2_cross_method_radar.png")
    print("  - fig2_cross_method_bars.png")
    print("  - fig3_paper_comparison.png")
    print("  - fig4_model_size.png")


if __name__ == "__main__":
    main()
