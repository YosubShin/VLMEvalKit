#!/usr/bin/env python3
"""
Filter TSV file by token counts in the question and answer columns.

Usage:
    python filter_tsv_by_token_count.py input.tsv output.tsv \
        --max-question-tokens 4096 --max-answer-tokens 13500 \
        [--min-question-tokens N] [--min-answer-tokens M]
"""

import sys
import os
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


def plot_token_histogram(token_counts, output_path, min_tokens, max_tokens, title, file_suffix):
    """Plot histogram of token counts for a column."""
    plt.figure(figsize=(12, 6))

    # Filter out zeros for better visualization
    non_zero_counts = [c for c in token_counts if c > 0]

    # Create histogram
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(non_zero_counts, bins=50, edgecolor='black', alpha=0.7)

    # Add vertical lines for min/max thresholds
    if min_tokens is not None:
        plt.axvline(min_tokens, color='red', linestyle='--', label=f'Min: {min_tokens}')
    if max_tokens is not None:
        plt.axvline(max_tokens, color='red', linestyle='--', label=f'Max: {max_tokens}')

    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Token Counts in {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Log scale version
    plt.subplot(1, 2, 2)
    plt.hist(non_zero_counts, bins=50, edgecolor='black', alpha=0.7)
    if min_tokens is not None:
        plt.axvline(min_tokens, color='red', linestyle='--', label=f'Min: {min_tokens}')
    if max_tokens is not None:
        plt.axvline(max_tokens, color='red', linestyle='--', label=f'Max: {max_tokens}')

    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of Token Counts (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    root, ext = os.path.splitext(output_path)
    histogram_path = f"{root}_{file_suffix}_token_histogram.png"
    plt.savefig(histogram_path, dpi=100, bbox_inches='tight')
    print(f"Histogram saved to {histogram_path}")

    # Also show if possible
    try:
        plt.show()
    except:
        pass


def filter_tsv_by_tokens(tsv_path,
                         output_path,
                         min_question_tokens=None,
                         max_question_tokens=None,
                         min_answer_tokens=None,
                         max_answer_tokens=None,
                         model="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Filter TSV rows based on token counts in question and answer columns.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output filtered TSV file
        min_question_tokens: Optional minimum tokens for question (no lower bound if None)
        max_question_tokens: Required maximum tokens for question
        min_answer_tokens: Optional minimum tokens for answer (no lower bound if None)
        max_answer_tokens: Required maximum tokens for answer
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

    # Check if required columns exist
    if 'question' not in df.columns:
        print("Error: 'question' column not found in TSV file")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    if 'answer' not in df.columns:
        print("Error: 'answer' column not found in TSV file")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Count tokens for question and answer
    print("Counting tokens in questions and answers...")
    df['question_token_count'] = df['question'].apply(lambda x: count_tokens(x, tokenizer))
    df['answer_token_count'] = df['answer'].apply(lambda x: count_tokens(x, tokenizer))

    # Calculate statistics before filtering
    q_counts = df['question_token_count'].values
    a_counts = df['answer_token_count'].values

    print("\n=== Question Token Count Statistics (Before Filtering) ===")
    print(f"Total questions: {len(q_counts)}")
    print(f"Mean tokens: {np.mean(q_counts):.1f}")
    print(f"Median tokens: {np.median(q_counts):.1f}")
    print(f"Std dev: {np.std(q_counts):.1f}")
    print(f"Min tokens: {np.min(q_counts)}")
    print(f"Max tokens: {np.max(q_counts)}")

    print("\n=== Answer Token Count Statistics (Before Filtering) ===")
    print(f"Total answers: {len(a_counts)}")
    print(f"Mean tokens: {np.mean(a_counts):.1f}")
    print(f"Median tokens: {np.median(a_counts):.1f}")
    print(f"Std dev: {np.std(a_counts):.1f}")
    print(f"Min tokens: {np.min(a_counts)}")
    print(f"Max tokens: {np.max(a_counts)}")

    # Distribution percentiles for answers (kept for continuity)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nAnswer Percentiles:")
    for p in percentiles:
        val = np.percentile(a_counts, p)
        print(f"  {p:3d}%: {val:8.0f} tokens")

    # Filter based on token counts (apply optional mins and required maxes)
    print("\nFiltering rows based on token thresholds...")
    mask = np.ones(len(df), dtype=bool)

    # Question constraints
    if min_question_tokens is not None:
        mask &= df['question_token_count'] >= min_question_tokens
    mask &= df['question_token_count'] <= max_question_tokens

    # Answer constraints
    if min_answer_tokens is not None:
        mask &= df['answer_token_count'] >= min_answer_tokens
    mask &= df['answer_token_count'] <= max_answer_tokens

    filtered_df = df[mask].copy()

    print(f"Rows after filtering: {len(filtered_df)}")
    print(f"Rows removed: {len(df) - len(filtered_df)}")
    print(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")

    # Calculate statistics after filtering
    if len(filtered_df) > 0:
        fq = filtered_df['question_token_count'].values
        fa = filtered_df['answer_token_count'].values
        print("\n=== Question Token Count Statistics (After Filtering) ===")
        print(f"Total questions: {len(fq)}")
        print(f"Mean tokens: {np.mean(fq):.1f}")
        print(f"Median tokens: {np.median(fq):.1f}")
        print(f"Std dev: {np.std(fq):.1f}")
        print(f"Min tokens: {np.min(fq)}")
        print(f"Max tokens: {np.max(fq)}")

        print("\n=== Answer Token Count Statistics (After Filtering) ===")
        print(f"Total answers: {len(fa)}")
        print(f"Mean tokens: {np.mean(fa):.1f}")
        print(f"Median tokens: {np.median(fa):.1f}")
        print(f"Std dev: {np.std(fa):.1f}")
        print(f"Min tokens: {np.min(fa)}")
        print(f"Max tokens: {np.max(fa)}")

    # Remove the temporary token_count columns before saving
    filtered_df = filtered_df.drop(['question_token_count', 'answer_token_count'], axis=1)

    # Save filtered TSV
    print(f"\nSaving filtered TSV to {output_path}...")
    filtered_df.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(filtered_df)} rows to {output_path}")

    # Plot histograms for both columns
    plot_token_histogram(q_counts, output_path, min_question_tokens, max_question_tokens, 'Question Column', 'question')
    plot_token_histogram(a_counts, output_path, min_answer_tokens, max_answer_tokens, 'Answer Column', 'answer')

    # Show which indices were filtered out (with reasons)
    if 'index' in df.columns:
        removed_df = df[~mask]
        if len(removed_df) > 0:
            print("\nSample of removed entries (first 10):")
            for idx, row in removed_df.head(10).iterrows():
                reasons = []
                qt = row['question_token_count']
                at = row['answer_token_count']
                if min_question_tokens is not None and qt < min_question_tokens:
                    reasons.append(f"question below min ({qt} < {min_question_tokens})")
                if qt > max_question_tokens:
                    reasons.append(f"question above max ({qt} > {max_question_tokens})")
                if min_answer_tokens is not None and at < min_answer_tokens:
                    reasons.append(f"answer below min ({at} < {min_answer_tokens})")
                if at > max_answer_tokens:
                    reasons.append(f"answer above max ({at} > {max_answer_tokens})")
                reason_str = '; '.join(reasons) if reasons else 'threshold mismatch'
                print(f"  Index {row['index']}: Q={qt} tokens, A={at} tokens -> {reason_str}")

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description='Filter TSV file by token counts in question and answer columns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter with custom thresholds
  python filter_tsv_by_token_count.py input.tsv output.tsv \
      --max-question-tokens 4096 --max-answer-tokens 13500 \
      --min-question-tokens 50 --min-answer-tokens 200

  # Use different tokenizer model
  python filter_tsv_by_token_count.py input.tsv output.tsv --model Qwen/Qwen2.5-7B-Instruct
        """
    )

    parser.add_argument('input_file', help='Path to input TSV file')
    parser.add_argument('output_file', help='Path to output filtered TSV file')
    # Arguments
    parser.add_argument('--min-question-tokens', type=int, default=None,
                       help='Optional minimum number of tokens in question (no lower bound if omitted)')
    parser.add_argument('--max-question-tokens', type=int, default=None,
                       help='Required maximum number of tokens in question')
    parser.add_argument('--min-answer-tokens', type=int, default=None,
                       help='Optional minimum number of tokens in answer (no lower bound if omitted)')
    parser.add_argument('--max-answer-tokens', type=int, default=None,
                       help='Required maximum number of tokens in answer')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Model to use for tokenization (default: Qwen/Qwen2.5-VL-7B-Instruct)')

    args = parser.parse_args()

    # Validate arguments
    if args.max_question_tokens is None:
        print("Error: --max-question-tokens is required")
        sys.exit(1)
    if args.max_answer_tokens is None:
        print("Error: --max-answer-tokens is required")
        sys.exit(1)

    if args.min_question_tokens is not None and args.min_question_tokens < 0:
        print("Error: min-question-tokens must be non-negative")
        sys.exit(1)
    if args.min_answer_tokens is not None and args.min_answer_tokens < 0:
        print("Error: min-answer-tokens must be non-negative")
        sys.exit(1)

    if args.max_question_tokens <= 0:
        print("Error: max-question-tokens must be positive")
        sys.exit(1)
    if args.max_answer_tokens <= 0:
        print("Error: max-answer-tokens must be positive")
        sys.exit(1)

    if args.min_question_tokens is not None and args.min_question_tokens > args.max_question_tokens:
        print("Error: max-question-tokens must be greater than or equal to min-question-tokens")
        sys.exit(1)
    if args.min_answer_tokens is not None and args.min_answer_tokens > args.max_answer_tokens:
        print("Error: max-answer-tokens must be greater than or equal to min-answer-tokens")
        sys.exit(1)

    try:
        filter_tsv_by_tokens(
            args.input_file,
            args.output_file,
            min_question_tokens=args.min_question_tokens,
            max_question_tokens=args.max_question_tokens,
            min_answer_tokens=args.min_answer_tokens,
            max_answer_tokens=args.max_answer_tokens,
            model=args.model
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()