#!/usr/bin/env python3
"""
Filter TSV file by token count in the answer column.

Usage:
    python filter_tsv_by_token_count.py input.tsv output.tsv --min-tokens 200 --max-tokens 13500
    python filter_tsv_by_token_count.py input.tsv output.tsv  # Uses default min=200, max=13500
"""

import sys
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def count_tokens(text, tokenizer):
    """Count tokens in text using tokenizer."""
    if pd.isna(text) or text == '':
        return 0
    try:
        return len(tokenizer.encode(str(text)))
    except:
        # Fallback to approximate word count * 1.3 if encoding fails
        return int(len(str(text).split()) * 1.3)


def plot_token_histogram(token_counts, output_path, min_tokens, max_tokens):
    """Plot histogram of token counts."""
    plt.figure(figsize=(12, 6))

    # Filter out zeros for better visualization
    non_zero_counts = [c for c in token_counts if c > 0]

    # Create histogram
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(non_zero_counts, bins=50, edgecolor='black', alpha=0.7)

    # Add vertical lines for min/max thresholds
    plt.axvline(min_tokens, color='red', linestyle='--', label=f'Min: {min_tokens}')
    plt.axvline(max_tokens, color='red', linestyle='--', label=f'Max: {max_tokens}')

    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Counts in Answer Column')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Log scale version
    plt.subplot(1, 2, 2)
    plt.hist(non_zero_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(min_tokens, color='red', linestyle='--', label=f'Min: {min_tokens}')
    plt.axvline(max_tokens, color='red', linestyle='--', label=f'Max: {max_tokens}')

    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of Token Counts (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    histogram_path = output_path.replace('.tsv', '_token_histogram.png')
    plt.savefig(histogram_path, dpi=100, bbox_inches='tight')
    print(f"Histogram saved to {histogram_path}")

    # Also show if possible
    try:
        plt.show()
    except:
        pass


def filter_tsv_by_tokens(tsv_path, output_path, min_tokens=200, max_tokens=13500, model="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Filter TSV rows based on token count in answer column.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output filtered TSV file
        min_tokens: Minimum number of tokens required
        max_tokens: Maximum number of tokens allowed
        model: Model to use for tokenization (default: Qwen/Qwen2.5-VL-7B-Instruct)
    """
    # Initialize tokenizer
    print(f"Loading tokenizer for {model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except:
        # Fallback to base Qwen tokenizer if specific model not found
        print("Failed to load specific model tokenizer, using Qwen2.5-7B-Instruct tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    # Read the input TSV
    print(f"Reading TSV from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Total rows: {len(df)}")

    # Check if answer column exists
    if 'answer' not in df.columns:
        print("Error: 'answer' column not found in TSV file")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Count tokens for each answer
    print("Counting tokens in answers...")
    df['token_count'] = df['answer'].apply(lambda x: count_tokens(x, tokenizer))

    # Calculate statistics before filtering
    token_counts = df['token_count'].values

    print("\n=== Token Count Statistics (Before Filtering) ===")
    print(f"Total answers: {len(token_counts)}")
    print(f"Mean tokens: {np.mean(token_counts):.1f}")
    print(f"Median tokens: {np.median(token_counts):.1f}")
    print(f"Std dev: {np.std(token_counts):.1f}")
    print(f"Min tokens: {np.min(token_counts)}")
    print(f"Max tokens: {np.max(token_counts)}")

    # Distribution percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(token_counts, p)
        print(f"  {p:3d}%: {val:8.0f} tokens")

    # Count how many fall in different ranges
    print("\nToken count ranges:")
    ranges = [(0, 100), (100, 200), (200, 500), (500, 1000),
              (1000, 5000), (5000, 10000), (10000, 13500), (13500, 20000), (20000, float('inf'))]

    for low, high in ranges:
        count = np.sum((token_counts >= low) & (token_counts < high))
        pct = count / len(token_counts) * 100
        if high == float('inf'):
            print(f"  {low:5d}+     : {count:6d} ({pct:5.1f}%)")
        else:
            print(f"  {low:5d}-{high:<5d}: {count:6d} ({pct:5.1f}%)")

    # Filter based on token count
    print(f"\nFiltering answers with {min_tokens} <= tokens <= {max_tokens}...")
    filtered_df = df[(df['token_count'] >= min_tokens) & (df['token_count'] <= max_tokens)].copy()

    # Remove the temporary token_count column before saving
    filtered_df = filtered_df.drop('token_count', axis=1)

    print(f"Rows after filtering: {len(filtered_df)}")
    print(f"Rows removed: {len(df) - len(filtered_df)}")
    print(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")

    # Calculate statistics after filtering
    filtered_token_counts = df[(df['token_count'] >= min_tokens) & (df['token_count'] <= max_tokens)]['token_count'].values

    if len(filtered_token_counts) > 0:
        print("\n=== Token Count Statistics (After Filtering) ===")
        print(f"Total answers: {len(filtered_token_counts)}")
        print(f"Mean tokens: {np.mean(filtered_token_counts):.1f}")
        print(f"Median tokens: {np.median(filtered_token_counts):.1f}")
        print(f"Std dev: {np.std(filtered_token_counts):.1f}")
        print(f"Min tokens: {np.min(filtered_token_counts)}")
        print(f"Max tokens: {np.max(filtered_token_counts)}")

    # Save filtered TSV
    print(f"\nSaving filtered TSV to {output_path}...")
    filtered_df.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(filtered_df)} rows to {output_path}")

    # Plot histogram
    plot_token_histogram(token_counts, output_path, min_tokens, max_tokens)

    # Show which indices were filtered out
    if 'index' in df.columns:
        removed_df = df[(df['token_count'] < min_tokens) | (df['token_count'] > max_tokens)]
        if len(removed_df) > 0:
            print("\nSample of removed entries (first 10):")
            for idx, row in removed_df.head(10).iterrows():
                print(f"  Index {row['index']}: {row['token_count']} tokens")
                if row['token_count'] < min_tokens:
                    print(f"    (below minimum of {min_tokens})")
                else:
                    print(f"    (above maximum of {max_tokens})")

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description='Filter TSV file by token count in answer column',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter with custom thresholds
  python filter_tsv_by_token_count.py input.tsv output.tsv --min-tokens 200 --max-tokens 13500

  # Use default thresholds (200-13500)
  python filter_tsv_by_token_count.py input.tsv output.tsv

  # Use different tokenizer model
  python filter_tsv_by_token_count.py input.tsv output.tsv --model Qwen/Qwen2.5-7B-Instruct
        """
    )

    parser.add_argument('input_file', help='Path to input TSV file')
    parser.add_argument('output_file', help='Path to output filtered TSV file')
    parser.add_argument('--min-tokens', type=int, default=200,
                       help='Minimum number of tokens in answer (default: 200)')
    parser.add_argument('--max-tokens', type=int, default=13500,
                       help='Maximum number of tokens in answer (default: 13500)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Model to use for tokenization (default: Qwen/Qwen2.5-VL-7B-Instruct)')

    args = parser.parse_args()

    # Validate arguments
    if args.min_tokens < 0:
        print("Error: min-tokens must be non-negative")
        sys.exit(1)

    if args.max_tokens < args.min_tokens:
        print("Error: max-tokens must be greater than or equal to min-tokens")
        sys.exit(1)

    try:
        filter_tsv_by_tokens(
            args.input_file,
            args.output_file,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            model=args.model
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()