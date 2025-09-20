#!/usr/bin/env python3
"""
Sample random rows from a TSV file.

Usage:
    python sample_random_tsv.py input.tsv output.tsv --num-rows 100
    python sample_random_tsv.py input.tsv output.tsv --match-count other_file.tsv
"""

import sys
import pandas as pd
import argparse
import random


def sample_random_rows(tsv_path, output_path, num_rows=None, match_file=None, seed=None):
    """
    Sample random rows from a TSV file.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output sampled TSV file
        num_rows: Number of rows to sample (if specified)
        match_file: Path to file to match row count from
        seed: Random seed for reproducibility
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")

    # Read the input TSV
    print(f"Reading TSV from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Original TSV shape: {df.shape}")

    # Determine number of rows to sample
    if match_file:
        print(f"Matching row count from {match_file}...")
        # Detect file type and read accordingly
        if match_file.endswith('.xlsx'):
            match_df = pd.read_excel(match_file)
        elif match_file.endswith('.csv'):
            match_df = pd.read_csv(match_file)
        elif match_file.endswith('.tsv'):
            match_df = pd.read_csv(match_file, sep='\t')
        else:
            # Try to read as TSV by default
            match_df = pd.read_csv(match_file, sep='\t')

        num_rows = len(match_df)
        print(f"Matching {num_rows} rows from reference file")
    elif num_rows is None:
        raise ValueError("Must specify either --num-rows or --match-count")

    # Check if we have enough rows
    if num_rows > len(df):
        print(f"Warning: Requested {num_rows} rows but only {len(df)} available.")
        print("Sampling all available rows without replacement.")
        num_rows = len(df)

    # Sample random rows
    print(f"Sampling {num_rows} random rows...")
    sampled_df = df.sample(n=num_rows, replace=False, random_state=seed)

    # Sort by index to maintain some order (optional - remove if you want random order)
    # sampled_df = sampled_df.sort_values('index') if 'index' in sampled_df.columns else sampled_df

    print(f"Sampled TSV shape: {sampled_df.shape}")

    # Save to output file
    print(f"Saving sampled TSV to {output_path}...")
    sampled_df.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(sampled_df)} rows to {output_path}")

    # Print statistics
    print("\nStatistics:")
    print(f"Original rows: {len(df)}")
    print(f"Sampled rows: {len(sampled_df)}")
    print(f"Sampling rate: {len(sampled_df)/len(df)*100:.1f}%")

    # Show first few indices if index column exists
    if 'index' in sampled_df.columns:
        indices = sampled_df['index'].head(10).tolist()
        print(f"First 10 sampled indices: {indices}")

    return sampled_df


def main():
    parser = argparse.ArgumentParser(
        description='Sample random rows from a TSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 100 random rows
  python sample_random_tsv.py input.tsv output.tsv --num-rows 100

  # Sample same number of rows as in another file
  python sample_random_tsv.py input.tsv output.tsv --match-count reference.xlsx

  # Sample with specific seed for reproducibility
  python sample_random_tsv.py input.tsv output.tsv --num-rows 50 --seed 42
        """
    )

    parser.add_argument('input_file', help='Path to input TSV file')
    parser.add_argument('output_file', help='Path to output sampled TSV file')

    # Mutually exclusive group for specifying row count
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument('--num-rows', type=int,
                             help='Number of rows to randomly sample')
    count_group.add_argument('--match-count',
                             help='Path to file to match row count from (TSV/CSV/XLSX)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')

    args = parser.parse_args()

    try:
        sample_random_rows(
            args.input_file,
            args.output_file,
            num_rows=args.num_rows,
            match_file=args.match_count,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()