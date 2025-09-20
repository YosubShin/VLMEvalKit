#!/usr/bin/env python3
"""
Filter TSV rows based on matching index and verdict=0 from XLSX file.

Usage:
    python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv
"""

import sys
import pandas as pd
import argparse


def filter_tsv_by_verdict(tsv_path, xlsx_path, output_path):
    """
    Filter TSV rows where index matches XLSX index and verdict is 0.

    Args:
        tsv_path: Path to input TSV file
        xlsx_path: Path to XLSX file with index and verdict columns
        output_path: Path to output filtered TSV file
    """
    # Read the TSV file
    print(f"Reading TSV from {tsv_path}...")
    tsv_df = pd.read_csv(tsv_path, sep='\t')
    print(f"TSV shape: {tsv_df.shape}")

    # Read the XLSX file
    print(f"Reading XLSX from {xlsx_path}...")
    xlsx_df = pd.read_excel(xlsx_path)
    print(f"XLSX shape: {xlsx_df.shape}")

    # Check required columns
    if 'index' not in tsv_df.columns:
        raise ValueError("TSV file must have an 'index' column")
    if 'index' not in xlsx_df.columns:
        raise ValueError("XLSX file must have an 'index' column")
    if 'verdict' not in xlsx_df.columns:
        raise ValueError("XLSX file must have a 'verdict' column")

    # Filter XLSX for verdict = 0
    failed_verdicts = xlsx_df[xlsx_df['verdict'] == 0]
    print(f"Found {len(failed_verdicts)} rows with verdict = 0")

    # Get the indices with verdict = 0
    failed_indices = set(failed_verdicts['index'].tolist())

    # Filter TSV rows where index is in failed_indices
    filtered_tsv = tsv_df[tsv_df['index'].isin(failed_indices)]
    print(f"Filtered TSV shape: {filtered_tsv.shape}")

    # Save to output file
    print(f"Saving filtered TSV to {output_path}...")
    filtered_tsv.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(filtered_tsv)} rows to {output_path}")

    # Print some statistics
    print("\nStatistics:")
    print(f"Original TSV rows: {len(tsv_df)}")
    print(f"XLSX rows with verdict=0: {len(failed_verdicts)}")
    print(f"Matched and filtered rows: {len(filtered_tsv)}")

    # Show which indices were not found in TSV (if any)
    not_found = failed_indices - set(tsv_df['index'].tolist())
    if not_found:
        print(f"\nWarning: {len(not_found)} indices from XLSX not found in TSV:")
        print(f"First 10 missing indices: {list(not_found)[:10]}")

    return filtered_tsv


def main():
    parser = argparse.ArgumentParser(
        description='Filter TSV rows based on XLSX verdict values'
    )
    parser.add_argument('tsv_file', help='Path to input TSV file')
    parser.add_argument('xlsx_file', help='Path to XLSX file with index and verdict columns')
    parser.add_argument('output_file', help='Path to output filtered TSV file')

    args = parser.parse_args()

    try:
        filter_tsv_by_verdict(args.tsv_file, args.xlsx_file, args.output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()