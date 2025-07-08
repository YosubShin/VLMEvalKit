#!/usr/bin/env python3
"""
DCVLR Results Analysis Script

This script reads DCVLR (Data Curation for Vision-Language Reasoning) scoring summary files 
from multiple models, creates a comparison spreadsheet, and generates visualization plots.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_benchmark_accuracy(file_content, benchmark_name):
    """Extract accuracy percentage for a specific benchmark from file content."""
    # Pattern to match benchmark accuracy in the per-benchmark breakdown
    pattern = rf"{re.escape(benchmark_name)}:\s+\d+(?:\.\d+)?/\d+\s+=\s+[\d.]+\s+\((\d+\.\d+)%\)"
    match = re.search(pattern, file_content)
    if match:
        return float(match.group(1))
    return None


def extract_model_name(file_path):
    """Extract model name from file path."""
    # Get the parent directory name which contains the model name
    parent_dir = Path(file_path).parent.name
    
    # Clean up the model name for display
    if parent_dir == "Qwen2.5-VL-7B-Instruct":
        return "Qwen2.5-VL-7B-Instruct (Base)"
    elif parent_dir == "Qwen2.5-VL-7B-Instruct-openr1":
        return "Qwen2.5-VL-7B-Instruct-OpenR1"
    elif "walton-multimodal-cold-start-r1-format" in parent_dir:
        return "Qwen2.5-VL-7B-Walton-R1"
    elif "MM-MathInstruct-to-r1-format-filtered" in parent_dir:
        return "Qwen2.5-VL-7B-MathInstruct-R1"
    else:
        return parent_dir


def parse_summary_file(file_path):
    """Parse a DCVLR scoring summary file and extract benchmark accuracies."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # List of benchmarks to extract
    benchmarks = [
        "VMCBench_DEV", "Omni3DBench", "OlympiadBench", "LiveXivVQA",
        "atomic_dataset", "electro_dataset", "mechanics_dataset", 
        "optics_dataset", "quantum_dataset", "statistics_dataset"
    ]
    
    # Extract overall accuracy
    overall_match = re.search(r"Overall accuracy:\s+[\d.]+\s+\((\d+\.\d+)%\)", content)
    overall_acc = float(overall_match.group(1)) if overall_match else None
    
    # Extract individual benchmark accuracies
    results = {}
    for benchmark in benchmarks:
        acc = parse_benchmark_accuracy(content, benchmark)
        if acc is not None:
            results[benchmark] = acc
    
    # Add overall accuracy
    if overall_acc is not None:
        results['Overall'] = overall_acc
        
    return results


def create_visualization(df, output_dir):
    """Create bar plot visualization of DCVLR results."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    benchmarks = [col for col in df.columns if col != 'Overall']
    x = np.arange(len(benchmarks))
    width = 0.2
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color scheme for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot bars for each model
    for i, (idx, row) in enumerate(df.iterrows()):
        values = [row[bench] for bench in benchmarks]
        bars = ax.bar(x + i*width, values, width, label=idx, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Benchmarks', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('DCVLR Benchmark Performance Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add a horizontal line at 50% for reference
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Reference')
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'dcvlr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'dcvlr_comparison.pdf'), format='pdf', bbox_inches='tight')
    
    # Create a second plot focused on physics benchmarks
    physics_benchmarks = ["atomic_dataset", "electro_dataset", "mechanics_dataset", 
                         "optics_dataset", "quantum_dataset", "statistics_dataset"]
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Get indices of physics benchmarks
    physics_indices = [benchmarks.index(b) for b in physics_benchmarks if b in benchmarks]
    x_physics = np.arange(len(physics_benchmarks))
    
    # Plot physics benchmarks only
    for i, (idx, row) in enumerate(df.iterrows()):
        values = [row[bench] for bench in physics_benchmarks]
        bars = ax2.bar(x_physics + i*width, values, width, label=idx, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    ax2.set_xlabel('Physics Benchmarks', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('DCVLR Physics Benchmark Performance (Detailed View)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_physics + width * 1.5)
    ax2.set_xticklabels([b.replace('_dataset', '') for b in physics_benchmarks], rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 20)  # Lower scale for physics benchmarks
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dcvlr_physics_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create overall performance bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    models = df.index.tolist()
    overall_scores = df['Overall'].tolist()
    
    bars = ax3.bar(range(len(models)), overall_scores, color=colors[:len(models)], alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, overall_scores):
        height = bar.get_height()
        ax3.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax3.set_xlabel('Models', fontsize=12)
    ax3.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax3.set_title('DCVLR Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(overall_scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dcvlr_overall_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')


def main():
    # Base directory
    base_dir = Path("/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/VLMEvalKit")
    
    # List of summary files to analyze
    summary_files = [
        "results/full/Qwen2.5-VL-7B-Instruct/vmcbench_scoring_summary.txt",
        "results/full/Qwen2.5-VL-7B-Instruct-openr1/vmcbench_scoring_summary.txt",
        "results/full/penfever_qwen2_5_vl_7b_walton-multimodal-cold-start-r1-format/vmcbench_scoring_summary.txt",
        "results/full/penfever_qwen2_5_vl_7b_MM-MathInstruct-to-r1-format-filtered/vmcbench_scoring_summary.txt"
    ]
    
    # Parse all files
    all_results = {}
    for file_path in summary_files:
        full_path = base_dir / file_path
        if full_path.exists():
            model_name = extract_model_name(str(full_path))
            results = parse_summary_file(str(full_path))
            all_results[model_name] = results
            print(f"Parsed {model_name}: {len(results)} benchmarks found")
        else:
            print(f"Warning: File not found - {full_path}")
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(all_results, orient='index')
    
    # Sort columns (put Overall last)
    cols = [col for col in df.columns if col != 'Overall']
    cols.sort()
    if 'Overall' in df.columns:
        cols.append('Overall')
    df = df[cols]
    
    # Round values for display
    df = df.round(1)
    
    # Create output directory
    output_dir = base_dir / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    excel_path = output_dir / "dcvlr_comparison.xlsx"
    df.to_excel(excel_path, index=True)
    print(f"\nSaved Excel file to: {excel_path}")
    
    # Also save as CSV for easy viewing
    csv_path = output_dir / "dcvlr_comparison.csv"
    df.to_csv(csv_path, index=True)
    print(f"Saved CSV file to: {csv_path}")
    
    # Print the DataFrame
    print("\nDCVLR Results Comparison:")
    print("=" * 80)
    print(df.to_string())
    
    # Create visualizations
    create_visualization(df, str(output_dir))
    print(f"\nCreated visualizations in: {output_dir}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    
    # Best performing model per benchmark
    print("\nBest performing model per benchmark:")
    for col in df.columns:
        best_model = df[col].idxmax()
        best_score = df[col].max()
        print(f"  {col}: {best_model} ({best_score}%)")
    
    # Average performance across all benchmarks (excluding Overall)
    benchmark_cols = [col for col in df.columns if col != 'Overall']
    avg_performance = df[benchmark_cols].mean(axis=1).sort_values(ascending=False)
    print("\nAverage performance across benchmarks:")
    for model, avg in avg_performance.items():
        print(f"  {model}: {avg:.1f}%")


if __name__ == "__main__":
    main()