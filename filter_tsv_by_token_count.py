#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.1",
#   "numpy>=1.26",
#   "matplotlib>=3.8",
#   "transformers>=4.44",
#   "pillow>=10.3",
#   "torch",          # needed by some HF processors for image token estimation
#   "torchvision",    # needed by some HF processors for image token estimation
# ]
# ///
"""
Filter TSV file by token counts in the question and answer columns (with optional total-token percentile trimming).

Usage:
    python filter_tsv_by_token_count.py input.tsv output.tsv \
        [--max-question-tokens 4096] [--max-answer-tokens 13500] \
        [--min-question-tokens N] [--min-answer-tokens M] \
        [--drop-top-total-percent X] [--image-token-column COL] [--default-image-tokens N] \
        [--no-special-tokens] [--compute-image-tokens] [--image-column COL] [--image-path-column COL]
"""

import sys
import os
import base64
from io import BytesIO
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import AutoTokenizer, AutoProcessor


def count_tokens(text, tokenizer, add_special_tokens=True):
    """Count tokens in text using tokenizer (with optional special tokens)."""
    if pd.isna(text) or text == '':
        return 0
    try:
        # Using tokenizer(..., return_length=True) is the most reliable across fast/slow tokenizers.
        encoded = tokenizer(
            str(text),
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
            truncation=False
        )
        # HF returns length as a scalar or a list; fallback to len(input_ids) if missing.
        if 'length' in encoded:
            length_val = encoded['length']
            if isinstance(length_val, (list, tuple, np.ndarray)):
                if len(length_val) == 0:
                    return 0
                length_val = length_val[0]
            # torch tensor or scalar-like
            try:
                return int(length_val)
            except Exception:
                pass
        if 'input_ids' in encoded:
            ids = encoded['input_ids']
            if isinstance(ids, (list, tuple)):
                if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
                    return len(ids[0])
                return len(ids)
            try:
                return len(ids)
            except Exception:
                pass
    except Exception as exc:
        print(f"Warning: tokenizer failed on text; falling back to word-count heuristic. Error: {exc}")

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
                         image_token_column=None,
                         image_column='image',
                         image_path_column=None,
                         default_image_tokens=0,
                         drop_top_total_percent=None,
                         add_special_tokens=True,
                         compute_image_tokens=False,
                         model="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Filter TSV rows based on token counts in question and answer columns.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output filtered TSV file
        min_question_tokens: Optional minimum tokens for question (no lower bound if None)
        max_question_tokens: Optional maximum tokens for question (no upper bound if None)
        min_answer_tokens: Optional minimum tokens for answer (no lower bound if None)
        max_answer_tokens: Optional maximum tokens for answer (no upper bound if None)
        image_token_column: Optional column name containing precomputed image token counts (used if compute_image_tokens=False)
        image_column: Column containing base64 encoded images (used when computing image tokens)
        image_path_column: Column containing file paths to images (fallback if base64 column missing/invalid)
        default_image_tokens: Fallback image token count to use when column is missing/NaN
        drop_top_total_percent: If set, drop the top X percent of rows by total tokens
        add_special_tokens: Whether to include model special tokens in counts (default: True)
        compute_image_tokens: If True, compute image token counts using the model processor
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

    # Optionally load processor for image token estimation
    processor = None
    if compute_image_tokens:
        print(f"Loading processor for {model} to compute image token counts...")
        try:
            processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        except Exception as exc:
            print(f"Warning: Failed to load processor for {model}. Falling back to default image tokens. Error: {exc}")
            processor = None

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
    df['question_token_count'] = df['question'].apply(lambda x: count_tokens(x, tokenizer, add_special_tokens=add_special_tokens))
    df['answer_token_count'] = df['answer'].apply(lambda x: count_tokens(x, tokenizer, add_special_tokens=add_special_tokens))

    # Add image token counts (computed, column, or default)
    if compute_image_tokens and processor is not None:
        print("Computing image token counts from images...")
        processor_warning_printed = False

        def load_pil_from_row(row):
            # Try base64 column first
            if image_column and image_column in df.columns:
                b64_val = row.get(image_column, None)
                if isinstance(b64_val, str) and b64_val.strip():
                    try:
                        return PILImage.open(BytesIO(base64.b64decode(b64_val))).convert('RGB')
                    except Exception:
                        pass
            # Fallback to image path column
            if image_path_column and image_path_column in df.columns:
                path_val = row.get(image_path_column, None)
                if isinstance(path_val, str) and os.path.exists(path_val):
                    try:
                        return PILImage.open(path_val).convert('RGB')
                    except Exception:
                        pass
            return None

        def estimate_image_tokens(img):
            if img is None:
                return default_image_tokens
            try:
                image_processor = getattr(processor, 'image_processor', processor)
                inputs = image_processor(images=img, return_tensors="pt")
                pixel_values = inputs.get('pixel_values', None) if hasattr(inputs, "get") else None
                if pixel_values is not None and hasattr(pixel_values, "ndim") and pixel_values.ndim == 4:
                    _, _, h, w = pixel_values.shape
                    # Try to infer patch size from processor (common for ViT-based vision encoders)
                    patch = getattr(image_processor, 'patch_size', None)
                    patch_h = patch_w = None
                    if isinstance(patch, dict):
                        patch_h = patch.get('height') or patch.get('shortest_edge') or patch.get('width')
                        patch_w = patch.get('width') or patch.get('shortest_edge') or patch.get('height')
                    elif isinstance(patch, (list, tuple)) and len(patch) == 2:
                        patch_h, patch_w = patch
                    elif isinstance(patch, int):
                        patch_h = patch_w = patch

                    if patch_h and patch_w and patch_h > 0 and patch_w > 0:
                        return int((h // patch_h) * (w // patch_w))
                    # Fallback: approximate using 14x14 patch size (common default)
                    return int((h * w) // (14 * 14))
            except Exception as exc:
                nonlocal processor_warning_printed
                if not processor_warning_printed:
                    print(f"Warning: failed to compute image tokens via processor; using heuristic/default. Error: {exc}")
                    processor_warning_printed = True
            # Heuristic fallback using raw image size and a 14x14 patch assumption
            try:
                w, h = img.size
                return max(int((h // 14) * (w // 14)), default_image_tokens)
            except Exception:
                return default_image_tokens

        df['image_token_count'] = df.apply(lambda row: estimate_image_tokens(load_pil_from_row(row)), axis=1)
        print("Computed image token counts.")
    elif image_token_column:
        if image_token_column in df.columns:
            df['image_token_count'] = pd.to_numeric(
                df[image_token_column],
                errors='coerce'
            ).fillna(default_image_tokens).astype(int)
            print(f"Using image token counts from column '{image_token_column}' with default {default_image_tokens} for missing values.")
        else:
            print(f"Warning: Column '{image_token_column}' not found. Using default image tokens={default_image_tokens} for all rows.")
            df['image_token_count'] = default_image_tokens
    else:
        df['image_token_count'] = default_image_tokens

    # Total token count = image + question + answer
    df['total_token_count'] = (
        df['image_token_count']
        + df['question_token_count']
        + df['answer_token_count']
    )

    # Calculate statistics before filtering
    q_counts = df['question_token_count'].values
    a_counts = df['answer_token_count'].values
    img_counts = df['image_token_count'].values
    total_counts = df['total_token_count'].values

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

    print("\n=== Image Token Count Statistics (Before Filtering) ===")
    print(f"Mean tokens: {np.mean(img_counts):.1f}")
    print(f"Median tokens: {np.median(img_counts):.1f}")
    print(f"Std dev: {np.std(img_counts):.1f}")
    print(f"Min tokens: {np.min(img_counts)}")
    print(f"Max tokens: {np.max(img_counts)}")

    print("\n=== Total Token Count Statistics (Before Filtering) ===")
    print(f"Mean tokens: {np.mean(total_counts):.1f}")
    print(f"Median tokens: {np.median(total_counts):.1f}")
    print(f"Std dev: {np.std(total_counts):.1f}")
    print(f"Min tokens: {np.min(total_counts)}")
    print(f"Max tokens: {np.max(total_counts)}")

    # Distribution percentiles for answers (kept for continuity)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nAnswer Percentiles:")
    for p in percentiles:
        val = np.percentile(a_counts, p)
        print(f"  {p:3d}%: {val:8.0f} tokens")

    # Filter based on token counts (apply optional mins and required maxes)
    print("\nFiltering rows based on token thresholds...")
    mask = np.ones(len(df), dtype=bool)
    total_cutoff = None

    # Question constraints
    if min_question_tokens is not None:
        mask &= df['question_token_count'] >= min_question_tokens
    if max_question_tokens is not None:
        mask &= df['question_token_count'] <= max_question_tokens

    # Answer constraints
    if min_answer_tokens is not None:
        mask &= df['answer_token_count'] >= min_answer_tokens
    if max_answer_tokens is not None:
        mask &= df['answer_token_count'] <= max_answer_tokens

    # Drop top X% by total tokens if requested
    if drop_top_total_percent is not None:
        percentile_to_keep = 100 - drop_top_total_percent
        total_cutoff = np.percentile(total_counts, percentile_to_keep)
        print(f"Dropping top {drop_top_total_percent}% samples with total tokens above {total_cutoff:.1f}")
        mask &= df['total_token_count'] <= total_cutoff

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

        fi = filtered_df['image_token_count'].values
        ft = filtered_df['total_token_count'].values

        print("\n=== Image Token Count Statistics (After Filtering) ===")
        print(f"Mean tokens: {np.mean(fi):.1f}")
        print(f"Median tokens: {np.median(fi):.1f}")
        print(f"Std dev: {np.std(fi):.1f}")
        print(f"Min tokens: {np.min(fi)}")
        print(f"Max tokens: {np.max(fi)}")

        print("\n=== Total Token Count Statistics (After Filtering) ===")
        print(f"Mean tokens: {np.mean(ft):.1f}")
        print(f"Median tokens: {np.median(ft):.1f}")
        print(f"Std dev: {np.std(ft):.1f}")
        print(f"Min tokens: {np.min(ft)}")
        print(f"Max tokens: {np.max(ft)}")

    # Remove the temporary token_count columns before saving
    filtered_df = filtered_df.drop(
        ['question_token_count', 'answer_token_count', 'image_token_count', 'total_token_count'],
        axis=1
    )

    # Save filtered TSV
    print(f"\nSaving filtered TSV to {output_path}...")
    filtered_df.to_csv(output_path, sep='\t', index=False)
    print(f"Successfully saved {len(filtered_df)} rows to {output_path}")

    # Plot histograms for both columns
    plot_token_histogram(q_counts, output_path, min_question_tokens, max_question_tokens, 'Question Column', 'question')
    plot_token_histogram(a_counts, output_path, min_answer_tokens, max_answer_tokens, 'Answer Column', 'answer')
    plot_token_histogram(total_counts, output_path, None, total_cutoff, 'Total Tokens (Image+Text)', 'total')

    # Show which indices were filtered out (with reasons)
    if 'index' in df.columns:
        removed_df = df[~mask]
        if len(removed_df) > 0:
            print("\nSample of removed entries (first 10):")
            for idx, row in removed_df.head(100).iterrows():
                reasons = []
                qt = row['question_token_count']
                at = row['answer_token_count']
                it = row['image_token_count']
                tt = row['total_token_count']
                if min_question_tokens is not None and qt < min_question_tokens:
                    reasons.append(f"question below min ({qt} < {min_question_tokens})")
                if max_question_tokens is not None and qt > max_question_tokens:
                    reasons.append(f"question above max ({qt} > {max_question_tokens})")
                if min_answer_tokens is not None and at < min_answer_tokens:
                    reasons.append(f"answer below min ({at} < {min_answer_tokens})")
                if max_answer_tokens is not None and at > max_answer_tokens:
                    reasons.append(f"answer above max ({at} > {max_answer_tokens})")
                if total_cutoff is not None and tt > total_cutoff:
                    reasons.append(f"total tokens in top {drop_top_total_percent}% ({tt} > {total_cutoff:.1f})")
                reason_str = '; '.join(reasons) if reasons else 'threshold mismatch'
                print(f"  Index {row['index']}: Q={qt} tokens, A={at} tokens, I={it} tokens, Total={tt} -> {reason_str}")

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
      --min-question-tokens 50 --min-answer-tokens 200 \
      --drop-top-total-percent 5 --image-token-column image_token_count

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
    parser.add_argument('--image-token-column', default=None,
                       help='Optional column name containing image token counts')
    parser.add_argument('--image-column', default='image',
                       help='Column name containing base64-encoded images (used with --compute-image-tokens)')
    parser.add_argument('--image-path-column', default=None,
                       help='Column name containing image file paths (fallback for --compute-image-tokens)')
    parser.add_argument('--default-image-tokens', type=int, default=0,
                       help='Fallback image token count when column is missing/NaN (default: 0)')
    parser.add_argument('--drop-top-total-percent', type=float, default=None,
                       help='Drop the top X percent of rows by total tokens (image + question + answer)')
    parser.add_argument('--no-special-tokens', action='store_true',
                       help='Exclude model special tokens from token counts')
    parser.add_argument('--compute-image-tokens', action='store_true',
                       help='Compute image token counts using the model processor (base64 or paths)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Model to use for tokenization (default: Qwen/Qwen2.5-VL-7B-Instruct)')

    args = parser.parse_args()

    # Validate arguments
    if args.min_question_tokens is not None and args.min_question_tokens < 0:
        print("Error: min-question-tokens must be non-negative")
        sys.exit(1)
    if args.min_answer_tokens is not None and args.min_answer_tokens < 0:
        print("Error: min-answer-tokens must be non-negative")
        sys.exit(1)

    if args.max_question_tokens is not None and args.max_question_tokens <= 0:
        print("Error: max-question-tokens must be positive")
        sys.exit(1)
    if args.max_answer_tokens is not None and args.max_answer_tokens <= 0:
        print("Error: max-answer-tokens must be positive")
        sys.exit(1)

    if args.drop_top_total_percent is not None:
        if args.drop_top_total_percent <= 0 or args.drop_top_total_percent >= 100:
            print("Error: drop-top-total-percent must be between 0 and 100")
            sys.exit(1)

    if args.min_question_tokens is not None and args.max_question_tokens is not None and args.min_question_tokens > args.max_question_tokens:
        print("Error: max-question-tokens must be greater than or equal to min-question-tokens")
        sys.exit(1)
    if args.min_answer_tokens is not None and args.max_answer_tokens is not None and args.min_answer_tokens > args.max_answer_tokens:
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
            image_token_column=args.image_token_column,
            image_column=args.image_column,
            image_path_column=args.image_path_column,
            default_image_tokens=args.default_image_tokens,
            drop_top_total_percent=args.drop_top_total_percent,
            add_special_tokens=not args.no_special_tokens,
            compute_image_tokens=args.compute_image_tokens,
            model=args.model
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
