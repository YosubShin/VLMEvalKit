#!/usr/bin/env python3
"""
Extract VLMEvalKit Results Script

This script scans a directory for various result files and extracts accuracy scores
from different benchmarks in their specific formats.

Usage:
    python scripts/extract_vlmevalkit_results.py results/full/Qwen2.5-VL-7B-Instruct
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import glob
import re


def calculate_gpt4o_mini_accuracy(file_path):
    """Calculate accuracy from GPT-4o-mini judged XLSX files."""
    try:
        df = pd.read_excel(file_path)
        if 'res' in df.columns:
            true_count = (df['res'] == 'true').sum()
            total_count = len(df)
            accuracy = (true_count / total_count) * 100 if total_count > 0 else 0
            return accuracy
        else:
            print(f"Warning: 'res' column not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_livexiv_accuracy(file_path):
    """Extract LiveXivVQA accuracy from CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'Overall' in df.columns and len(df) > 0:
            # The Overall value is in decimal form (0-1), convert to percentage
            return df['Overall'].iloc[0] * 100
        else:
            print(f"Warning: 'Overall' column not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_olympiadbench_accuracy(file_path):
    """Extract OlympiadBench accuracy from score result CSV."""
    try:
        df = pd.read_csv(file_path)
        if 'AVG' in df.columns and len(df) > 0:
            # The AVG value is already in percentage form (0-100)
            return df['AVG'].iloc[0]
        else:
            print(f"Warning: 'AVG' column not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_omni3dbench_accuracy(file_path):
    """Extract Omni3DBench accuracy as simple average of first row."""
    try:
        df = pd.read_csv(file_path)
        if len(df) > 0:
            # Calculate simple average of all numeric values in first row
            first_row = df.iloc[0]
            numeric_values = []
            for value in first_row:
                try:
                    # Try to convert to float, skip if not numeric
                    num_val = float(value)
                    numeric_values.append(num_val)
                except (ValueError, TypeError):
                    continue
            
            if numeric_values:
                # Values are in decimal form (0-1), convert to percentage
                return (sum(numeric_values) / len(numeric_values)) * 100
            else:
                print(f"Warning: No numeric values found in first row of {file_path}")
                return None
        else:
            print(f"Warning: Empty dataframe in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_vmcbench_dev_accuracy(file_path):
    """Extract VMCBench_DEV accuracy from CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'Overall' in df.columns and len(df) > 0:
            # The Overall value is already in percentage form (0-100)
            return df['Overall'].iloc[0]
        else:
            print(f"Warning: 'Overall' column not found in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLMEvalKit results from various file formats"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory path to scan for result files"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    dir_path = Path(args.directory)
    
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} does not exist")
        sys.exit(1)
    
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a directory")
        sys.exit(1)
    
    print(f"Scanning directory: {dir_path}")
    
    # Extract model name from directory
    model_name = dir_path.name
    
    # Dictionary to store results
    results = {}
    
    # 1. Process GPT-4o-mini XLSX files
    print("\nSearching for GPT-4o-mini judged XLSX files...")
    xlsx_pattern = str(dir_path / f"{model_name}_*_gpt-4o-mini.xlsx")
    xlsx_files = glob.glob(xlsx_pattern)
    
    for xlsx_file in xlsx_files:
        # Extract benchmark name from filename
        filename = Path(xlsx_file).name
        match = re.match(rf"{re.escape(model_name)}_(.+)_gpt-4o-mini\.xlsx", filename)
        if match:
            benchmark = match.group(1)
            accuracy = calculate_gpt4o_mini_accuracy(xlsx_file)
            if accuracy is not None:
                results[f"{benchmark}_accuracy"] = accuracy
                print(f"  Found {benchmark}: {accuracy:.2f}%")
    
    # 2. Process LiveXivVQA CSV
    print("\nSearching for LiveXivVQA accuracy file...")
    livexiv_pattern = str(dir_path / f"{model_name}_LiveXivVQA_acc.csv")
    livexiv_files = glob.glob(livexiv_pattern)
    
    if livexiv_files:
        accuracy = extract_livexiv_accuracy(livexiv_files[0])
        if accuracy is not None:
            results["LiveXiv_accuracy"] = accuracy
            print(f"  Found LiveXivVQA: {accuracy:.2f}%")
    
    # 3. Process OlympiadBench score result
    print("\nSearching for OlympiadBench score result file...")
    olympiad_pattern = str(dir_path / f"{model_name}_OlympiadBench_score_result.csv")
    olympiad_files = glob.glob(olympiad_pattern)
    
    if olympiad_files:
        accuracy = extract_olympiadbench_accuracy(olympiad_files[0])
        if accuracy is not None:
            results["OlympiadBench_accuracy"] = accuracy
            print(f"  Found OlympiadBench: {accuracy:.2f}%")
    
    # 4. Process Omni3DBench CSV
    print("\nSearching for Omni3DBench accuracy file...")
    omni3d_pattern = str(dir_path / f"{model_name}_Omni3DBench_acc.csv")
    omni3d_files = glob.glob(omni3d_pattern)
    
    if omni3d_files:
        accuracy = extract_omni3dbench_accuracy(omni3d_files[0])
        if accuracy is not None:
            results["Omni3DBench_accuracy"] = accuracy
            print(f"  Found Omni3DBench: {accuracy:.2f}%")
    
    # 5. Process VMCBench_DEV CSV
    print("\nSearching for VMCBench_DEV accuracy file...")
    vmcbench_pattern = str(dir_path / f"{model_name}_VMCBench_DEV_acc.csv")
    vmcbench_files = glob.glob(vmcbench_pattern)
    
    if vmcbench_files:
        accuracy = extract_vmcbench_dev_accuracy(vmcbench_files[0])
        if accuracy is not None:
            results["VMCBench_DEV_accuracy"] = accuracy
            print(f"  Found VMCBench_DEV: {accuracy:.2f}%")
    
    # Save results to CSV
    output_file = dir_path / "cumulative_vlmevalkit_results.csv"
    
    if results:
        # Convert to DataFrame with single row
        df = pd.DataFrame([results])
        
        # Sort columns alphabetically for consistency
        df = df[sorted(df.columns)]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Display results
        print("\nSummary of extracted results:")
        print("-" * 50)
        for key, value in sorted(results.items()):
            print(f"{key}: {value:.2f}%")
        print(f"\nTotal benchmarks found: {len(results)}")
    else:
        print("\nNo results found!")
        print("Make sure the directory contains the expected files with correct naming patterns.")
        print("\nExpected file patterns:")
        print(f"  - {model_name}_*_gpt-4o-mini.xlsx")
        print(f"  - {model_name}_LiveXivVQA_acc.csv")
        print(f"  - {model_name}_OlympiadBench_score_result.csv")
        print(f"  - {model_name}_Omni3DBench_acc.csv")
        print(f"  - {model_name}_VMCBench_DEV_acc.csv")


if __name__ == "__main__":
    main()