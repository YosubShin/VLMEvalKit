#!/usr/bin/env python3
"""
Filter TSV rows based on matching index and verdict=0 from XLSX file.

Usage:
    python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv
"""

import sys
import pandas as pd
import argparse


def filter_tsv_by_verdict(tsv_path, xlsx_path, output_path, verdict_value=0):
    """
    Filter TSV rows where index matches XLSX index and verdict equals specified value.

    Args:
        tsv_path: Path to input TSV file
        xlsx_path: Path to XLSX file with index and verdict columns
        output_path: Path to output filtered TSV file
        verdict_value: Verdict value to filter for (default: 0)
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

    # Filter XLSX for specified verdict value
    filtered_verdicts = xlsx_df[xlsx_df['verdict'] == verdict_value]
    print(f"Found {len(filtered_verdicts)} rows with verdict = {verdict_value}")

    # Get the indices with specified verdict value
    filtered_indices = set(filtered_verdicts['index'].tolist())

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
    print(f"XLSX rows with verdict={verdict_value}: {len(filtered_verdicts)}")
    print(f"Matched and filtered rows: {len(filtered_tsv)}")

    # Show which indices were not found in TSV (if any)
    not_found = filtered_indices - set(tsv_df['index'].tolist())
    if not_found:
        print(f"\nWarning: {len(not_found)} indices from XLSX not found in TSV:")
        print(f"First 10 missing indices: {list(not_found)[:10]}")

    return filtered_tsv


def main():
    parser = argparse.ArgumentParser(
        description='Filter TSV rows based on XLSX verdict values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter for verdict=0 (default - failed cases)
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv

  # Filter for verdict=1 (successful cases)
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --verdict 1

  # Filter for any specific verdict value
  python filter_tsv_by_verdict.py input.tsv results.xlsx output.tsv --verdict 2
        """
    )
    parser.add_argument('tsv_file', help='Path to input TSV file')
    parser.add_argument('xlsx_file', help='Path to XLSX file with index and verdict columns')
    parser.add_argument('output_file', help='Path to output filtered TSV file')
    parser.add_argument('--verdict', type=int, default=0,
                       help='Verdict value to filter for (default: 0 for failed cases)')

    args = parser.parse_args()

    try:
        filter_tsv_by_verdict(args.tsv_file, args.xlsx_file, args.output_file, args.verdict)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()