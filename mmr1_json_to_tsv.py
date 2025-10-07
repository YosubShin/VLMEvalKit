#!/usr/bin/env python3
"""
Convert a conversation JSON array into a TSV compatible with downstream tools
like sample_random_tsv.py and create_walton_hf_dataset.py.

Input JSON format (array of items):
[
  {
    "conversations": [
      {"from": "human", "value": "<image> ... question text ..."},
      {"from": "gpt", "value": "<think>...</think>\n<answer>... answer ...</answer>"}
    ],
    "images": ["relative/or/absolute/path/to/image.png", ...]
  },
  ...
]

This script will:
- Optionally sample up to N items BEFORE any image IO/encoding to save time
- Extract the first human question and the first assistant answer (preferring
  content inside <answer>...</answer> if present)
- Encode the first image (if any) to base64
- Write a TSV with columns: index, image, question, answer

Usage:
  python json_to_tsv.py input.json output.tsv [--max-samples N] [--seed S]
                                            [--image-root DIR] [--verbose]
"""

import argparse
import base64
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert conversation JSON array to TSV with base64 images",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("output_tsv", help="Path to output TSV file")

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Limit number of randomly selected items to process. Sampling is performed "
            "BEFORE any image IO/encoding to save time. If omitted, process all items."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling (default: 42)",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "Optional root directory to prefix to image paths from JSON when not absolute."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def extract_question_answer(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract the first human question and first assistant answer from a conversation item.

    Retains the original human message (including any leading "<image>" tokens), and
    retains the assistant content including <think>...</think> and <answer>...</answer>
    blocks if present.
    """
    conversations = item.get("conversations", [])
    question: Optional[str] = None
    answer: Optional[str] = None

    # Find first human message: retain from first "<image>" (inclusive); if not found, retain as-is
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value", "")
            if isinstance(value, str):
                s = value.strip()
                pos = s.find("<image>")
                question = s[pos:].strip() if pos != -1 else s
            break

    # Find first assistant message (retain full content; do not extract only <answer>)
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") in {"gpt", "assistant"}:
            value = turn.get("value", "")
            if isinstance(value, str):
                answer = value.strip()
            break

    return question, answer


def resolve_first_image_path(item: Dict[str, Any], image_root: Optional[str]) -> Optional[str]:
    images = item.get("images")
    if not isinstance(images, list) or not images:
        return None
    raw_path = images[0]
    if not isinstance(raw_path, str) or not raw_path:
        return None

    if os.path.isabs(raw_path):
        return raw_path
    if image_root:
        return os.path.join(image_root, raw_path)
    return raw_path


def encode_image_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    # Load JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        try:
            items: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(items, list):
        print("Error: Input JSON must be an array of items", file=sys.stderr)
        sys.exit(1)

    total_items = len(items)
    log(f"Loaded {total_items} items from JSON", args.verbose)

    # Sampling before any expensive IO
    rng = random.Random(args.seed)
    indices = list(range(total_items))
    if args.max_samples is not None and args.max_samples < total_items:
        indices = rng.sample(indices, args.max_samples)
        log(
            f"Sampling {len(indices)} of {total_items} items before image encoding (seed={args.seed})",
            args.verbose,
        )
    else:
        log("Sampling disabled; processing all items", args.verbose)

    rows: List[Dict[str, Any]] = []
    skipped_no_convo = 0
    skipped_no_image = 0
    skipped_bad_image = 0

    for out_idx, item_idx in enumerate(indices):
        item = items[item_idx]

        question, answer = extract_question_answer(item)
        if not question or not answer:
            skipped_no_convo += 1
            continue

        image_path = resolve_first_image_path(item, args.image_root)
        if not image_path:
            skipped_no_image += 1
            continue

        b64 = encode_image_base64(image_path)
        if not b64:
            skipped_bad_image += 1
            continue

        rows.append(
            {
                "index": out_idx,
                "image": b64,
                "question": question,
                "answer": answer,
            }
        )

    if args.verbose:
        log(
            (
                f"Prepared {len(rows)} rows. Skipped: no_convo={skipped_no_convo}, "
                f"no_image={skipped_no_image}, bad_image={skipped_bad_image}"
            ),
            True,
        )

    if not rows:
        print("Error: No valid rows to write after filtering/skips", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows, columns=["index", "image", "question", "answer"])
    df.to_csv(args.output_tsv, sep="\t", index=False)
    print(
        f"Wrote {len(df)} rows to {args.output_tsv} (from {total_items} input items)")


if __name__ == "__main__":
    main()


