#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.19",
#   "huggingface_hub>=0.23",
#   "openpyxl>=3.1",
#   "pandas>=2.1",
#   "Pillow>=10.3",
# ]
# ///
"""
Filter the oumi-ai/walton-multimodal-cold-start-r1-format dataset by domain, optionally
sample a subset, and push the result as a Hugging Face dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter oumi-ai/walton-multimodal-cold-start-r1-format by domain labels derived "
            "from the full multimodal dataset, optionally sample, and create a Hugging Face dataset."
        )
    )
    parser.add_argument(
        "--base-excel",
        default="msc/multimodal_cold_start_with_domains.xlsx",
        help="Excel file containing problem/answer/domain columns (default: %(default)s).",
    )
    parser.add_argument(
        "--source-dataset",
        default="oumi-ai/walton-multimodal-cold-start-r1-format",
        help="Source Hugging Face dataset ID (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load from the source dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=["Geometry"],
        help="Domain name to exclude (can be provided multiple times). Default removes 'Geometry'.",
    )
    parser.add_argument(
        "--include-domain",
        action="append",
        help="Domain name to include (if provided, only these domains are kept before exclusion).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of random samples to keep after filtering (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/oumi_filtered_dataset",
        help="Directory to save the filtered dataset via save_to_disk (default: %(default)s).",
    )
    parser.add_argument(
        "--repo-id",
        help="Target Hugging Face dataset repo ID for push_to_hub (required when --push is set).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the filtered dataset to Hugging Face Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the dataset as private (only relevant with --push).",
    )
    parser.add_argument(
        "--commit-message",
        default="Add filtered oumi Walton dataset",
        help="Commit message for push_to_hub (default: %(default)s).",
    )
    return parser.parse_args()


def load_domain_lookup(path: Path) -> Dict[Tuple[str, str], Tuple[str, Optional[int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Base Excel file not found: {path}")

    df = pd.read_excel(path)
    for column in ("problem", "answer", "domain"):
        if column not in df.columns:
            raise ValueError(f"Missing required column '{column}' in {path}")

    grouped = df.groupby(["problem", "answer"])["domain"].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        example_pair = conflicts.index[0]
        print(
            f"Found {len(conflicts)} problem+answer pairs with conflicting domains "
            f"(e.g., {example_pair}). Resolving by majority vote (ties -> lowest example_id)."
        )

    agg_dict = {"count": ("domain", "size")}
    has_example_id = "example_id" in df.columns
    if has_example_id:
        agg_dict["first_example"] = ("example_id", "min")

    domain_stats = (
        df.groupby(["problem", "answer", "domain"])
        .agg(**agg_dict)
        .reset_index()
    )

    sort_cols = ["problem", "answer", "count"]
    ascending = [True, True, False]
    if has_example_id:
        sort_cols.append("first_example")
        ascending.append(True)

    domain_stats = domain_stats.sort_values(sort_cols, ascending=ascending)
    preferred = domain_stats.drop_duplicates(subset=["problem", "answer"], keep="first")

    lookup: Dict[Tuple[str, str], Tuple[str, Optional[int]]] = {}
    for row in preferred.itertuples(index=False):
        example_id = getattr(row, "first_example", None) if has_example_id else None
        if example_id is None or pd.isna(example_id):
            example_id_int: Optional[int] = None
        else:
            example_id_int = int(example_id)
        key = (row.problem, row.answer)
        lookup[key] = (row.domain, example_id_int)
    return lookup


def attach_domains(dataset, lookup: Dict[Tuple[str, str], Tuple[str, Optional[int]]]):
    def map_fn(example):
        key = (example["problem"], example["solution"])
        domain_info = lookup.get(key)
        if domain_info:
            domain, example_id = domain_info
        else:
            domain, example_id = "(blank)", None
        return {
            "domain": domain,
            "source_example_id": int(example_id) if example_id is not None else None,
        }

    return dataset.map(map_fn)


def filter_dataset(dataset: Dataset, include_domains: Optional[Iterable[str]], exclude_domains: Iterable[str]) -> Dataset:
    include_set = set(include_domains) if include_domains else None
    exclude_set = set(exclude_domains) if exclude_domains else set()

    if include_set:
        dataset = dataset.filter(lambda example: example["domain"] in include_set)

    if exclude_set:
        dataset = dataset.filter(lambda example: example["domain"] not in exclude_set)

    return dataset


def sample_dataset(dataset: Dataset, sample_size: Optional[int], seed: int) -> Dataset:
    if sample_size is None or sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(sample_size))


def push_dataset(dataset: Dataset, repo_id: str, commit_message: str, private: bool) -> None:
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(repo_id, private=private, commit_message=commit_message)
    print(f"Pushed filtered dataset to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_excel)
    output_dir = Path(args.output_dir)

    print("Loading domain lookup...")
    domain_lookup = load_domain_lookup(base_path)
    print(f"Loaded {len(domain_lookup)} unique problem+answer domain pairs.")

    print(f"Loading dataset {args.source_dataset}:{args.split} ...")
    dataset = load_dataset(args.source_dataset, split=args.split)
    print(f"Loaded {len(dataset)} rows from source dataset.")

    dataset = attach_domains(dataset, domain_lookup)
    missing_domains = sum(1 for d in dataset["domain"] if d == "(blank)")
    if missing_domains:
        print(f"Warning: {missing_domains} rows missing domain labels after attachment (filled with '(blank)').")

    dataset = filter_dataset(dataset, args.include_domain, args.exclude_domain)
    print(f"Remaining rows after domain filtering: {len(dataset)}")

    dataset = sample_dataset(dataset, args.sample_size, args.seed)
    print(f"Rows after sampling: {len(dataset)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Saved filtered dataset to {output_dir}")

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when using --push.")
        push_dataset(dataset, args.repo_id, args.commit_message, args.private)


if __name__ == "__main__":
    main()
