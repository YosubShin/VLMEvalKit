#!/usr/bin/env python3
"""
Filter TSV rows based on verdict_sum from XLSX file to retain intermediate difficulty questions.

Usage:
    python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --exclude-sum 0 8
"""

import sys
import pandas as pd
import argparse
import json
import numpy as np


def sample_tsv_by_verdict_sum(tsv_path, xlsx_path, output_path, sample_config=None, seed=42):
    """
    Sample TSV rows based on verdict_sum distribution.

    Args:
        tsv_path: Path to input TSV file
        xlsx_path: Path to XLSX file with index and verdict_sum columns
        output_path: Path to output sampled TSV file
        sample_config: Dict mapping verdict_sum to desired count (e.g., {0: 55, 1: 48, 2: 41})
        seed: Random seed for reproducible sampling
    """
    np.random.seed(seed)

    # Read the TSV file
    print(f"Reading TSV from {tsv_path}...")
    tsv_df = pd.read_csv(tsv_path, sep='\t')
    print(f"TSV shape: {tsv_df.shape}")

    # Read the XLSX file (first sheet by default)
    print(f"Reading XLSX from {xlsx_path}...")
    xlsx_df = pd.read_excel(xlsx_path, sheet_name=0)
    print(f"XLSX shape: {xlsx_df.shape}")

    # Get the rightmost column (should be verdict_sum)
    verdict_sum_col = xlsx_df.columns[-1]
    print(f"Using rightmost column: '{verdict_sum_col}' as verdict_sum")

    # Check required columns
    if 'index' not in tsv_df.columns:
        raise ValueError("TSV file must have an 'index' column")
    if 'index' not in xlsx_df.columns:
        raise ValueError("XLSX file must have an 'index' column")

    # Show original distribution
    print(f"\nOriginal verdict_sum distribution:")
    value_counts = xlsx_df[verdict_sum_col].value_counts().sort_index()
    for val, count in value_counts.items():
        print(f"  verdict_sum={val}: {count} questions")

    # Sample based on configuration
    selected_indices = []
    total_selected = 0

    for verdict_sum, desired_count in sample_config.items():
        # Get rows with this verdict_sum
        subset = xlsx_df[xlsx_df[verdict_sum_col] == verdict_sum]
        available_count = len(subset)

        if available_count == 0:
            print(f"  verdict_sum={verdict_sum}: 0 available, skipping")
            continue

        # Sample the desired number (or all if not enough available)
        actual_count = min(desired_count, available_count)
        if actual_count < available_count:
            sampled_subset = subset.sample(n=actual_count, random_state=seed + verdict_sum)
        else:
            sampled_subset = subset

        selected_indices.extend(sampled_subset['index'].tolist())
        total_selected += actual_count

        print(f"  verdict_sum={verdict_sum}: selected {actual_count}/{available_count} (wanted {desired_count})")

    print(f"\nTotal selected: {total_selected} questions")

    # Filter TSV to include only selected indices
    selected_indices_set = set(selected_indices)
    filtered_tsv = tsv_df[tsv_df['index'].isin(selected_indices_set)]
    print(f"Filtered TSV shape: {filtered_tsv.shape}")

    # Save to output file
    print(f"Saving sampled TSV to {output_path}...")
    filtered_tsv.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(filtered_tsv)} rows to {output_path}")

    # Print final statistics
    print("\nFinal sampled distribution:")
    final_counts = xlsx_df[xlsx_df['index'].isin(selected_indices_set)][verdict_sum_col].value_counts().sort_index()
    for val, count in final_counts.items():
        print(f"  verdict_sum={val}: {count} questions")

    return filtered_tsv


def filter_tsv_by_verdict_sum(tsv_path, xlsx_path, output_path, exclude_sums=None, k=None):
    """
    Filter TSV rows based on verdict_sum to retain intermediate difficulty questions.

    Args:
        tsv_path: Path to input TSV file
        xlsx_path: Path to XLSX file with index and verdict_sum columns
        output_path: Path to output filtered TSV file
        exclude_sums: List of verdict_sum values to exclude (e.g., [0, 8] for k=8)
        k: Number of times each question was asked (for auto-exclude if exclude_sums not provided)
    """
    # Read the TSV file
    print(f"Reading TSV from {tsv_path}...")
    tsv_df = pd.read_csv(tsv_path, sep='\t')
    print(f"TSV shape: {tsv_df.shape}")

    # Read the XLSX file (first sheet by default)
    print(f"Reading XLSX from {xlsx_path}...")
    xlsx_df = pd.read_excel(xlsx_path, sheet_name=0)
    print(f"XLSX shape: {xlsx_df.shape}")

    # Get the rightmost column (should be verdict_sum)
    verdict_sum_col = xlsx_df.columns[-1]
    print(f"Using rightmost column: '{verdict_sum_col}' as verdict_sum")

    # Check required columns
    if 'index' not in tsv_df.columns:
        raise ValueError("TSV file must have an 'index' column")
    if 'index' not in xlsx_df.columns:
        raise ValueError("XLSX file must have an 'index' column")

    # Auto-determine exclude_sums if not provided
    if exclude_sums is None:
        if k is not None:
            # Auto-exclude 0 and k (all failed and all passed)
            exclude_sums = [0, k]
            print(f"Auto-excluding verdict_sum values: {exclude_sums} (too easy/too hard for k={k})")
        else:
            # Try to infer k from the data
            max_verdict_sum = xlsx_df[verdict_sum_col].max()
            exclude_sums = [0, max_verdict_sum]
            print(f"Auto-excluding verdict_sum values: {exclude_sums} (inferred k={max_verdict_sum})")
    else:
        print(f"Excluding verdict_sum values: {exclude_sums}")

    # Show distribution of verdict_sum values
    print(f"\nVerdict_sum distribution:")
    value_counts = xlsx_df[verdict_sum_col].value_counts().sort_index()
    for val, count in value_counts.items():
        status = " (excluded)" if val in exclude_sums else ""
        print(f"  verdict_sum={val}: {count} questions{status}")

    # Filter XLSX to exclude specified verdict_sum values
    filtered_df = xlsx_df[~xlsx_df[verdict_sum_col].isin(exclude_sums)]
    print(f"\nRetaining {len(filtered_df)} questions with intermediate difficulty")

    # Get the indices to keep
    filtered_indices = set(filtered_df['index'].tolist())

    # Filter TSV rows where index is in filtered_indices
    filtered_tsv = tsv_df[tsv_df['index'].isin(filtered_indices)]
    print(f"Filtered TSV shape: {filtered_tsv.shape}")

    # Save to output file
    print(f"Saving filtered TSV to {output_path}...")
    filtered_tsv.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(filtered_tsv)} rows to {output_path}")

    # Print some statistics
    print("\nStatistics:")
    print(f"Original TSV rows: {len(tsv_df)}")
    print(f"Questions excluded (too easy/hard): {len(xlsx_df) - len(filtered_df)}")
    print(f"Questions retained (intermediate): {len(filtered_df)}")
    print(f"Matched and filtered TSV rows: {len(filtered_tsv)}")

    # Show which indices were not found in TSV (if any)
    not_found = filtered_indices - set(tsv_df['index'].tolist())
    if not_found:
        print(f"\nWarning: {len(not_found)} indices from XLSX not found in TSV:")
        print(f"First 10 missing indices: {list(not_found)[:10]}")

    return filtered_tsv


def main():
    parser = argparse.ArgumentParser(
        description='Filter or sample TSV rows based on verdict_sum',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter: Auto-exclude too easy (0) and too hard (8) for k=8 questions
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --k 8

  # Filter: Explicitly exclude specific verdict_sum values
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --exclude-sum 0 8

  # Sample: Specify exact counts for each verdict_sum value
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --sample '{"0": 55, "1": 48, "2": 41, "3": 30, "4": 25, "5": 20, "6": 15, "7": 10, "8": 5}'

  # Sample: With custom seed for reproducibility
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --sample '{"2": 100, "3": 100, "4": 100}' --seed 123
        """
    )
    parser.add_argument('tsv_file', help='Path to input TSV file')
    parser.add_argument('xlsx_file', help='Path to XLSX file with verdict_sum in rightmost column')
    parser.add_argument('output_file', help='Path to output filtered/sampled TSV file')

    # Filtering options
    parser.add_argument('--exclude-sum', nargs='+', type=int,
                       help='Verdict_sum values to exclude (e.g., 0 8 for k=8)')
    parser.add_argument('--k', type=int,
                       help='Number of times each question was asked (auto-excludes 0 and k)')

    # Sampling options
    parser.add_argument('--sample', type=str,
                       help='JSON string mapping verdict_sum to desired count (e.g., \'{"0": 55, "1": 48}\')')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')

    args = parser.parse_args()

    try:
        if args.sample:
            # Parse the sample configuration
            sample_config = json.loads(args.sample)
            # Convert string keys to integers
            sample_config = {int(k): v for k, v in sample_config.items()}

            sample_tsv_by_verdict_sum(args.tsv_file, args.xlsx_file, args.output_file,
                                    sample_config=sample_config, seed=args.seed)
        else:
            # Use filtering mode
            filter_tsv_by_verdict_sum(args.tsv_file, args.xlsx_file, args.output_file,
                                      exclude_sums=args.exclude_sum, k=args.k)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()