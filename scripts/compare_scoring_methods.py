#!/usr/bin/env python3
"""
Compare Scoring Methods Visualization

This script compares VLMEvalKit scores with DCVLR standalone scorer results
and creates visualizations showing the differences per benchmark.

Usage:
    python scripts/compare_scoring_methods.py
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_dcvlr_summary(file_path):
    """Parse DCVLR scoring summary file and extract benchmark accuracies."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    # Define benchmark mappings (DCVLR name -> VLMEvalKit name)
    benchmark_mappings = {
        "VMCBench_DEV": "VMCBench_DEV_accuracy",
        "Omni3DBench": "Omni3DBench_accuracy", 
        "OlympiadBench": "OlympiadBench_accuracy",
        "LiveXivVQA": "LiveXiv_accuracy",
        "atomic_dataset": "atomic_dataset_accuracy",
        "electro_dataset": "electro_dataset_accuracy",
        "mechanics_dataset": "mechanics_dataset_accuracy",
        "optics_dataset": "optics_dataset_accuracy",
        "quantum_dataset": "quantum_dataset_accuracy",
        "statistics_dataset": "statistics_dataset_accuracy"
    }
    
    # Extract individual benchmark accuracies
    for dcvlr_name, vlm_name in benchmark_mappings.items():
        # Pattern to match benchmark accuracy in the per-benchmark breakdown
        pattern = rf"{re.escape(dcvlr_name)}:\s+[\d.]+/\d+\s+=\s+[\d.]+\s+\((\d+\.\d+)%\)"
        match = re.search(pattern, content)
        if match:
            results[vlm_name] = float(match.group(1))
        
    return results


def create_comparison_visualization(vlm_scores, dcvlr_scores, output_dir):
    """Create comparison visualization of scoring methods."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find common benchmarks
    common_benchmarks = set(vlm_scores.keys()) & set(dcvlr_scores.keys())
    
    if not common_benchmarks:
        print("Warning: No common benchmarks found between the two scoring methods")
        return
    
    # Prepare data for plotting
    benchmarks = sorted(list(common_benchmarks))
    vlm_values = [vlm_scores[bench] for bench in benchmarks]
    dcvlr_values = [dcvlr_scores[bench] for bench in benchmarks]
    differences = [dcvlr_values[i] - vlm_values[i] for i in range(len(benchmarks))]
    
    # Clean up benchmark names for display
    display_names = []
    for bench in benchmarks:
        clean_name = bench.replace('_accuracy', '').replace('_dataset', '')
        if clean_name == 'LiveXiv':
            clean_name = 'LiveXivVQA'
        display_names.append(clean_name)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Colors
    vlm_color = '#1f77b4'
    dcvlr_color = '#ff7f0e'
    diff_color = '#2ca02c'
    
    # Subplot 1: Side-by-side comparison
    x = np.arange(len(benchmarks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, vlm_values, width, label='VLMEvalKit', color=vlm_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, dcvlr_values, width, label='DCVLR Scorer', color=dcvlr_color, alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    ax1.set_xlabel('Benchmarks', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Scoring Method Comparison: VLMEvalKit vs DCVLR Standalone Scorer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(vlm_values), max(dcvlr_values)) * 1.1)
    
    # Subplot 2: Difference plot
    colors = [diff_color if diff >= 0 else '#d62728' for diff in differences]
    bars3 = ax2.bar(x, differences, color=colors, alpha=0.8)
    
    # Add value labels on difference bars
    for bar, diff in zip(bars3, differences):
        height = bar.get_height()
        ax2.annotate(f'{diff:+.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Benchmarks', fontsize=12)
    ax2.set_ylabel('Score Difference (DCVLR - VLMEvalKit)', fontsize=12)
    ax2.set_title('Score Differences: Positive = DCVLR Higher, Negative = VLMEvalKit Higher', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add reference line at zero
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / 'scoring_method_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'scoring_method_comparison.pdf', format='pdf', bbox_inches='tight')
    
    print(f"Comparison visualization saved to: {output_path}")
    
    # Create a detailed summary table
    summary_df = pd.DataFrame({
        'Benchmark': display_names,
        'VLMEvalKit_Score': vlm_values,
        'DCVLR_Score': dcvlr_values,
        'Difference': differences,
        'Abs_Difference': [abs(d) for d in differences]
    })
    
    # Save summary table
    summary_path = Path(output_dir) / 'scoring_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Detailed comparison saved to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SCORING METHOD COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\nBenchmarks compared: {len(benchmarks)}")
    print(f"Mean absolute difference: {np.mean([abs(d) for d in differences]):.2f}%")
    print(f"Max difference: {max(differences):.2f}% ({display_names[differences.index(max(differences))]})")
    print(f"Min difference: {min(differences):.2f}% ({display_names[differences.index(min(differences))]})")
    
    # Count where each method performs better
    dcvlr_better = sum(1 for d in differences if d > 0)
    vlm_better = sum(1 for d in differences if d < 0)
    tied = sum(1 for d in differences if d == 0)
    
    print(f"\nDCVLR Scorer performs better: {dcvlr_better} benchmarks")
    print(f"VLMEvalKit performs better: {vlm_better} benchmarks") 
    print(f"Tied: {tied} benchmarks")
    
    # Show detailed breakdown
    print(f"\nDetailed breakdown:")
    for i, (bench, vlm_val, dcvlr_val, diff) in enumerate(zip(display_names, vlm_values, dcvlr_values, differences)):
        print(f"  {bench}: VLMEvalKit={vlm_val:.1f}%, DCVLR={dcvlr_val:.1f}%, Diff={diff:+.1f}%")
    
    plt.close()


def main():
    # File paths
    base_dir = Path("/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/VLMEvalKit")
    vlm_file = base_dir / "results/full/Qwen2.5-VL-7B-Instruct/cumulative_vlmevalkit_results.csv"
    dcvlr_file = base_dir / "results/full/Qwen2.5-VL-7B-Instruct/vmcbench_scoring_summary.txt"
    output_dir = base_dir / "results/analysis"
    
    # Check if files exist
    if not vlm_file.exists():
        print(f"Error: VLMEvalKit results file not found: {vlm_file}")
        return
    
    if not dcvlr_file.exists():
        print(f"Error: DCVLR summary file not found: {dcvlr_file}")
        return
    
    print("Loading VLMEvalKit results...")
    # Load VLMEvalKit scores
    vlm_df = pd.read_csv(vlm_file)
    if len(vlm_df) == 0:
        print("Error: VLMEvalKit results file is empty")
        return
    
    vlm_scores = vlm_df.iloc[0].to_dict()  # First (and only) row
    print(f"Loaded {len(vlm_scores)} VLMEvalKit scores")
    
    print("Loading DCVLR scorer results...")
    # Load DCVLR scores
    dcvlr_scores = parse_dcvlr_summary(str(dcvlr_file))
    print(f"Loaded {len(dcvlr_scores)} DCVLR scores")
    
    # Create visualization
    print("Creating comparison visualization...")
    create_comparison_visualization(vlm_scores, dcvlr_scores, str(output_dir))


if __name__ == "__main__":
    main()