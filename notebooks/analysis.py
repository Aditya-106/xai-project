"""
Analysis and Results Visualization Script
==========================================
Run this script to generate all comparison tables and figures,
including side-by-side comparison with the original paper's results.

Usage:
    python notebooks/analysis.py
    python notebooks/analysis.py --results-file results/tables/ablation_comparison.json
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['font.size'] = 11

# ══════════════════════════════════════════════════════════════
# PAPER'S REPORTED RESULTS
# (from Table 1, Table 3, and Appendix of the NAACL 2024 paper)
# ══════════════════════════════════════════════════════════════

# Table 1: SEA-CoT Ablation on StrategyQA
PAPER_ABLATION_STRATEGYQA = pd.DataFrame({
    "Method":     ["Random", "Max", "Overlap", "Entailment", "O&E (SEA-CoT)"],
    "P ↓":        [6.10, 1.80, 1.56, 2.38, 1.20],
    "CF-UF ↓":    [6.44, 6.60, 5.04, 5.46, 3.81],
    "M ↑":        [62.17, 61.80, 70.83, 69.99, 61.24],
    "S ↑":        [11.87, 12.59, 14.88, 13.46, 16.97],
})

# Cross-method comparison on StrategyQA (from paper)
PAPER_CROSS_METHOD_STRATEGYQA = pd.DataFrame({
    "Method":     ["CoT", "SC-CoT", "QD", "Self-Refine", "SEA-CoT"],
    "P ↓":        [3.41, 1.80, 5.12, 3.90, 1.20],
    "CF-UF ↓":    [6.54, 6.60, 7.86, 6.82, 3.81],
    "M ↑":        [57.39, 61.80, 54.85, 60.05, 61.24],
    "S ↑":        [9.27, 12.59, 7.78, 10.50, 16.97],
})

# Model size comparison on StrategyQA
PAPER_MODEL_SIZE = pd.DataFrame({
    "Model":   ["Llama-2-70B", "Llama-2-13B", "Llama-2-7B"],
    "P ↓":     [1.20, 4.10, 3.79],
    "CF-UF ↓": [3.81, 4.38, 7.81],
    "M ↑":     [61.24, 69.62, 70.62],
    "S ↑":     [16.97, 6.16, 15.97],
})


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def load_our_results(results_file: str = "results/tables/ablation_comparison.json") -> dict:
    """Load our reproduced results from the JSON file."""
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        return data
    return None


def display_ablation_table(our_data=None):
    """Display the SEA-CoT ablation study results (Paper Table 1)."""
    print_section("RESULT 1: SEA-CoT Ablation Study (Paper Table 1)")
    print("\nDataset: StrategyQA | Model: Llama-2-70B-chat-GPTQ")
    print("\nMetrics: P = Paraphrase flip rate (↓), CF-UF = Counterfactual Unfaithfulness (↓)")
    print("         M = Mistake flip rate (↑), S = Simulatability/LAS (↑)")
    print()
    print(PAPER_ABLATION_STRATEGYQA.to_string(index=False))
    print()
    print("Key finding: O&E (SEA-CoT) achieves the best P (1.20) and CF-UF (3.81),")
    print("and the highest simulatability S (16.97), demonstrating that combining")
    print("entailment + overlap scoring selects the most interpretable explanations.")

    if our_data and "ours" in our_data:
        print("\n" + "-"*70)
        print("  OUR REPRODUCED RESULTS:")
        print("-"*70)
        ours = our_data["ours"]
        mapping = {
            "Random": "Random", "Max": "Max", "Overlap": "Overlap",
            "Entailment": "Entailment", "O&E (SEA)": "O&E (SEA-CoT)"
        }
        rows = []
        for key, display_name in mapping.items():
            if key in ours:
                rows.append({
                    "Method": display_name,
                    "P ↓": ours[key]["P"],
                    "CF-UF ↓": ours[key]["CF-UF"],
                    "M ↑": ours[key]["M"],
                    "S ↑": ours[key]["S"],
                })
        if rows:
            our_df = pd.DataFrame(rows)
            print()
            print(our_df.to_string(index=False))

            if "config" in our_data:
                cfg = our_data["config"]
                print(f"\n  Model: {cfg.get('model', 'N/A')}")
                print(f"  NLI:   {cfg.get('nli_model', 'N/A')}")
                print(f"  LAS:   {cfg.get('t5_model', 'N/A')}")
                print(f"  N:     {cfg.get('n_samples', 'N/A')} examples")


def display_cross_method_table():
    """Display cross-method comparison (Paper Figure 5 / Table 3)."""
    print_section("RESULT 2: Cross-Method Interpretability Comparison")
    print("\nDataset: StrategyQA | Model: Llama-2-70B-chat-GPTQ")
    print()
    print(PAPER_CROSS_METHOD_STRATEGYQA.to_string(index=False))
    print()
    print("Key finding: SEA-CoT consistently outperforms all baselines across")
    print("interpretability dimensions. QD performs worst, suggesting that")
    print("question decomposition doesn't yield faithful explanations.")


def display_model_size_table():
    """Display model size comparison."""
    print_section("RESULT 3: Model Size Comparison (SEA-CoT)")
    print("\nDataset: StrategyQA | Method: SEA-CoT")
    print()
    print(PAPER_MODEL_SIZE.to_string(index=False))
    print()
    print("Key finding: The 70B model generally achieves the best interpretability,")
    print("with the lowest P and CF-UF scores and the highest simulatability.")


def display_trend_analysis(our_data):
    """Display trend agreement analysis between paper and our results."""
    if not our_data or "trend_analysis" not in our_data:
        return

    print_section("TREND ANALYSIS: Strategy Rankings (Paper vs Ours)")
    trends = our_data["trend_analysis"]
    print(f"\n  {'Metric':>6}  {'Spearman ρ':>12}  {'Paper Best':<16}  {'Our Best':<16}  {'Match?':>8}")
    print("  " + "-"*65)

    for metric in ["P", "CF-UF", "M", "S"]:
        if metric in trends:
            t = trends[metric]
            match_str = "✓ YES" if t["best_matches"] else "✗ NO"
            rho_str = f"{t['spearman_rho']:+.3f}"
            print(f"  {metric:>6}  {rho_str:>12}  {t['paper_best']:<16}  "
                  f"{t['our_best']:<16}  {match_str:>8}")

    print()
    # Interpretation
    rhos = [trends[m]["spearman_rho"] for m in ["P", "CF-UF", "M", "S"] if m in trends]
    matches = sum(1 for m in ["P", "CF-UF", "M", "S"] if m in trends and trends[m]["best_matches"])
    avg_rho = np.mean(rhos) if rhos else 0

    print(f"  Average Spearman ρ: {avg_rho:+.3f}")
    print(f"  Best-strategy matches: {matches}/4")
    print()
    if avg_rho > 0.5:
        print("  ✓ STRONG trend agreement — rankings are consistent with the paper.")
    elif avg_rho > 0.0:
        print("  ~ MODERATE trend agreement — partial consistency with the paper.")
    else:
        print("  ✗ WEAK trend agreement — rankings differ from the paper.")
        print("    This is expected with a much smaller model.")
    print("  Note: Absolute metric values differ due to model size difference.")
    print("  The key finding is whether SEA-CoT (O&E) still ranks near the top.")


def plot_ablation_chart(save_dir: str = "results/figures", our_data=None):
    """Generate ablation bar chart."""
    df = PAPER_ABLATION_STRATEGYQA
    metrics = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]
    methods = df["Method"].tolist()

    has_ours = our_data and "ours" in our_data
    n_groups = 2 if has_ours else 1

    fig, axes = plt.subplots(1, 4, figsize=(16 + (4 if has_ours else 0), 5))
    title = "SEA-CoT Ablation Study on StrategyQA (Paper Table 1)"
    if has_ours:
        title += "\nPaper (blue) vs Our Reproduction (red)"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    mapping = ["Random", "Max", "Overlap", "Entailment", "O&E (SEA)"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_key = metric.replace(" ↓", "").replace(" ↑", "")
        paper_values = df[metric].tolist()

        if has_ours:
            our_values = []
            for m in mapping:
                if m in our_data["ours"]:
                    our_values.append(our_data["ours"][m].get(metric_key, 0))
                else:
                    our_values.append(0)

            x = np.arange(len(methods))
            w = 0.35
            bars1 = ax.bar(x - w/2, paper_values, w, label='Paper (70B)',
                          color='#2196F3', edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + w/2, our_values, w, label='Ours',
                          color='#F44336', edgecolor='black', linewidth=0.5)

            for bar, val in zip(bars1, paper_values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)
            for bar, val in zip(bars2, our_values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

            ax.set_xticks(x)
            if idx == 0:
                ax.legend(fontsize=8)
        else:
            bars = ax.bar(range(len(methods)), paper_values, color=colors,
                         edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(methods)))
            for bar, val in zip(bars, paper_values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(metric, fontweight='bold', fontsize=13)
        ax.set_xticklabels([m.replace("(SEA-CoT)", "\n(SEA-CoT)").replace("(SEA)", "\n(SEA)")
                           for m in methods], rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/ablation_study.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/ablation_study.png")


def plot_cross_method_radar(save_dir: str = "results/figures"):
    """Generate radar chart for cross-method comparison."""
    df = PAPER_CROSS_METHOD_STRATEGYQA
    metrics = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]
    methods = df["Method"].tolist()

    # Normalize to [0,1] — invert P and CF-UF so higher = better
    normalized = {}
    for metric in metrics:
        vals = df[metric].values
        min_v, max_v = vals.min(), vals.max()
        if metric in ("P ↓", "CF-UF ↓"):
            normalized[metric] = 1 - (vals - min_v) / (max_v - min_v) if max_v > min_v else np.ones_like(vals)
        else:
            normalized[metric] = (vals - min_v) / (max_v - min_v) if max_v > min_v else np.ones_like(vals)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title("Cross-Method Interpretability\n(Normalized, Higher = Better)",
                 fontsize=14, fontweight='bold', pad=20)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    for i, method in enumerate(methods):
        values = [normalized[m][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["P\n(lower=better)", "CF-UF\n(lower=better)",
                         "M\n(higher=better)", "S\n(higher=better)"])
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/cross_method_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/cross_method_radar.png")


def plot_cross_method_bars(save_dir: str = "results/figures"):
    """Generate grouped bar chart for cross-method comparison."""
    df = PAPER_CROSS_METHOD_STRATEGYQA
    metrics = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]
    methods = df["Method"].tolist()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Cross-Method Comparison on StrategyQA (Paper Figure 5)",
                 fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    x = np.arange(len(methods))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].tolist()
        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/cross_method_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/cross_method_bars.png")


def plot_paper_vs_ours(our_data=None, save_dir: str = "results/figures"):
    """Generate paper vs reproduction comparison for SEA-CoT."""
    metrics = ["P", "CF-UF", "M", "S"]
    metric_labels = ["P ↓", "CF-UF ↓", "M ↑", "S ↑"]
    paper_vals = [1.20, 3.81, 61.24, 16.97]  # SEA-CoT paper values

    if our_data and "ours" in our_data and "O&E (SEA)" in our_data["ours"]:
        sea = our_data["ours"]["O&E (SEA)"]
        our_vals = [sea.get(m, 0) for m in metrics]
    else:
        our_vals = [0, 0, 0, 0]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, paper_vals, width, label='Paper (Llama-2-70B)',
                   color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, our_vals, width, label='Our Reproduction',
                   color='#F44336', edgecolor='black')

    ax.set_ylabel('Score')
    ax.set_title('Paper vs Our Reproduction — SEA-CoT on StrategyQA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/paper_vs_ours.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/paper_vs_ours.png")


def main():
    """Run full analysis — tables + figures."""
    import argparse
    parser = argparse.ArgumentParser(description="Analysis and visualization")
    parser.add_argument("--results-file", type=str,
                        default="results/tables/ablation_comparison.json",
                        help="Path to ablation comparison JSON file")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  CoT Interpretability Paper Reproduction — Results Analysis        ║")
    print("║  Paper: 'How Interpretable are Reasoning Explanations from         ║")
    print("║          Prompting Large Language Models?' (NAACL 2024)             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load our results if available
    our_data = load_our_results(args.results_file)
    if our_data:
        print(f"\n  ✓ Loaded reproduced results from {args.results_file}")
        if "config" in our_data:
            cfg = our_data["config"]
            print(f"    Model: {cfg.get('model', 'N/A')}")
            print(f"    Samples: {cfg.get('n_samples', 'N/A')}")
            print(f"    Accuracy: {cfg.get('accuracy', 'N/A'):.1f}%")
    else:
        print(f"\n  ⚠ No reproduced results found at {args.results_file}")
        print("    Showing paper results only. Run 'python run_all.py' first.")

    # Display tables
    display_ablation_table(our_data)
    display_cross_method_table()
    display_model_size_table()

    # Trend analysis (only if we have our results)
    if our_data:
        display_trend_analysis(our_data)

    # Generate figures
    print_section("GENERATING FIGURES")
    save_dir = "results/figures"

    plot_ablation_chart(save_dir, our_data)
    plot_cross_method_radar(save_dir)
    plot_cross_method_bars(save_dir)
    plot_paper_vs_ours(our_data, save_dir=save_dir)

    # Summary
    print_section("SUMMARY FOR PRESENTATION")
    print("""
    During your 10-minute demonstration:

    1. Show the paper's tables (printed above)
    2. Show the generated figures in results/figures/
    3. Walk through the code:
       - src/prompting/sea_cot.py    → SEA-CoT algorithm
       - src/scoring/entailment.py   → Entailment scoring (S_e) using DeBERTa NLI
       - src/scoring/overlap.py      → Overlap scoring (S_o) using token IoU
       - src/evaluation/             → All evaluation metrics
    4. Compare paper results vs your reproduction (side-by-side tables)
    5. Discuss trend agreement and absolute differences
    6. Explain WHY absolute values differ (model size, perturbation method)

    Key numbers from the paper:
    - SEA-CoT achieves P=1.20 (best robustness)
    - SEA-CoT achieves CF-UF=3.81 (best faithfulness)
    - SEA-CoT achieves S=16.97 (best utility)
    - >70% improvement over baselines in interpretability
    """)

    if our_data and "trend_analysis" in our_data:
        trends = our_data["trend_analysis"]
        matches = sum(1 for m in ["P", "CF-UF", "M", "S"]
                     if m in trends and trends[m]["best_matches"])
        print(f"    Our reproduction: {matches}/4 best-strategy matches with paper")


if __name__ == "__main__":
    main()
