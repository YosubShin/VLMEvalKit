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
import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter oumi-ai/walton-multimodal-cold-start-r1-format by domain labels derived "
            "from the full multimodal dataset, optionally sample, and create a Hugging Face dataset."
        )
    )
    parser.add_argument(
        "--per-domain-cap",
        type=int,
        help="Maximum number of examples to retain per domain (applied before final sampling).",
    )
    parser.add_argument(
        "--full-domain",
        action="append",
        help="Domains that bypass the per-domain cap (can be provided multiple times).",
    )
    parser.add_argument(
        "--full-domain-ratio",
        type=float,
        help=(
            "Fraction of the final sample reserved for --full-domain entries. "
            "Requires --sample-size and at least one full domain."
        ),
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
        help="Domain name to exclude (can be provided multiple times).",
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
        "--weighting",
        choices=["uniform", "log"],
        default="uniform",
        help="Sampling strategy for the final subset (default: uniform random).",
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


def log_domain_counts(stage: str, dataset: Dataset) -> None:
    counts = Counter(dataset["domain"])
    print(f"Domain distribution after {stage} (total={len(dataset)}):")
    for domain, count in counts.most_common():
        print(f"  {domain}: {count}")


def limit_per_domain(
    dataset: Dataset,
    per_domain_cap: Optional[int],
    full_domains: Optional[Iterable[str]],
    seed: int,
) -> Dataset:
    if per_domain_cap is None or per_domain_cap <= 0:
        return dataset

    full_domains_set = {d for d in (full_domains or []) if d is not None}
    domain_indices: Dict[str, list[int]] = defaultdict(list)
    for idx, domain in enumerate(dataset["domain"]):
        domain_indices[domain].append(idx)

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for domain in sorted(domain_indices):
        indices = domain_indices[domain]
        if domain in full_domains_set:
            selected_indices.extend(indices)
            continue
        cap = min(per_domain_cap, len(indices))
        if cap >= len(indices):
            selected_indices.extend(indices)
        else:
            sampled = rng.sample(indices, cap)
            selected_indices.extend(sorted(sampled))

    selected_indices.sort()
    return dataset.select(selected_indices)


def perform_final_sampling(
    dataset: Dataset,
    sample_size: Optional[int],
    weighting: str,
    full_domain_ratio: Optional[float],
    full_domains: set[str],
    seed: int,
) -> Dataset:
    if sample_size is None or sample_size <= 0:
        print("Sample size not specified or non-positive; skipping final sampling.")
        log_domain_counts("final sampling", dataset)
        return dataset

    if not full_domains or full_domain_ratio is None or math.isclose(full_domain_ratio, 0.0):
        sampled = sample_dataset(dataset, sample_size, weighting, seed)
        print(f"Rows after sampling: {len(sampled)}")
        log_domain_counts("final sampling", sampled)
        return sampled

    full_subset = dataset.filter(lambda example: example["domain"] in full_domains)
    if len(full_subset) == 0:
        sampled = sample_dataset(dataset, sample_size, weighting, seed)
        print(f"Rows after sampling: {len(sampled)}")
        log_domain_counts("final sampling", sampled)
        return sampled

    other_subset = dataset.filter(lambda example: example["domain"] not in full_domains)

    quota = int(round(sample_size * full_domain_ratio))
    quota = max(0, min(sample_size, quota))
    if quota == 0:
        selected_full = full_subset.select(range(0))
    else:
        selected_full = uniform_sample(full_subset, quota, seed)

    selected_full_count = len(selected_full)
    print(
        f"Full-domain selection: kept {selected_full_count} of {len(full_subset)} candidates (quota={quota})."
    )
    log_domain_counts("full-domain selection", selected_full)

    remaining = max(sample_size - selected_full_count, 0)
    rest_sample = None
    if remaining > 0 and len(other_subset) > 0:
        rest_sample = sample_dataset(other_subset, remaining, weighting, seed + 1)
        print(f"Weighted sampling for remaining pool: selected {len(rest_sample)} rows (target={remaining}).")
        log_domain_counts("weighted sampling", rest_sample)

    parts = [ds for ds in (selected_full, rest_sample) if ds is not None and len(ds) > 0]
    if not parts:
        combined = dataset.select(range(0))
    elif len(parts) == 1:
        combined = parts[0]
    else:
        combined = concatenate_datasets(parts)

    print(f"Rows after combined sampling: {len(combined)}")
    log_domain_counts("final sampling", combined)
    return combined


def uniform_sample(dataset: Dataset, sample_size: Optional[int], seed: int) -> Dataset:
    if sample_size is None or sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(sample_size))


def log_weighted_sample(dataset: Dataset, sample_size: Optional[int], seed: int) -> Dataset:
    if sample_size is None or sample_size <= 0 or sample_size >= len(dataset):
        return dataset

    domain_indices: Dict[str, list[int]] = defaultdict(list)
    for idx, domain in enumerate(dataset["domain"]):
        domain_indices[domain].append(idx)

    weights = {domain: math.log1p(len(indices)) for domain, indices in domain_indices.items()}
    total_weight = sum(weights.values())
    if total_weight == 0:
        return uniform_sample(dataset, sample_size, seed)

    quotas = {}
    remainders = []
    allocated = 0

    for domain, weight in weights.items():
        target = sample_size * weight / total_weight
        count = min(len(domain_indices[domain]), int(math.floor(target)))
        quotas[domain] = count
        allocated += count
        remainder = target - count
        if count < len(domain_indices[domain]):
            remainders.append((remainder, domain))

    remaining = min(sample_size - allocated, sum(max(0, len(idx) - quotas[dom]) for dom, idx in domain_indices.items()))
    if remaining > 0 and remainders:
        remainders.sort(reverse=True)
        idx = 0
        while remaining > 0 and idx < len(remainders):
            _, domain = remainders[idx]
            available = len(domain_indices[domain]) - quotas[domain]
            if available > 0:
                quotas[domain] += 1
                remaining -= 1
            idx += 1
            if idx == len(remainders) and remaining > 0:
                idx = 0

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for domain, indices in domain_indices.items():
        quota = quotas.get(domain, 0)
        if quota >= len(indices):
            selected_indices.extend(indices)
        elif quota > 0:
            selected_indices.extend(sorted(rng.sample(indices, quota)))

    if len(selected_indices) < sample_size:
        missing = sample_size - len(selected_indices)
        unused = sorted(set(range(len(dataset))) - set(selected_indices))
        if unused:
            extras = unused if missing >= len(unused) else rng.sample(unused, missing)
            selected_indices.extend(sorted(extras))

    selected_indices = sorted(selected_indices[:sample_size])
    return dataset.select(selected_indices)


def sample_dataset(dataset: Dataset, sample_size: Optional[int], weighting: str, seed: int) -> Dataset:
    if weighting == "log":
        return log_weighted_sample(dataset, sample_size, seed)
    return uniform_sample(dataset, sample_size, seed)


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

    full_domains_set = {d for d in (args.full_domain or []) if d is not None}

    dataset = filter_dataset(dataset, args.include_domain, args.exclude_domain)
    print(f"Remaining rows after domain filtering: {len(dataset)}")
    log_domain_counts("domain filtering", dataset)

    if args.per_domain_cap:
        dataset = limit_per_domain(dataset, args.per_domain_cap, full_domains_set, args.seed)
        print(f"Rows after per-domain limiting: {len(dataset)}")
        log_domain_counts("per-domain limiting", dataset)

    dataset = perform_final_sampling(
        dataset,
        args.sample_size,
        args.weighting,
        args.full_domain_ratio,
        full_domains_set,
        args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Saved filtered dataset to {output_dir}")

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when using --push.")
        push_dataset(dataset, args.repo_id, args.commit_message, args.private)


if __name__ == "__main__":
    main()
