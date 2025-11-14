#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.19",
#   "huggingface_hub>=0.23",
#   "openpyxl>=3.1",
#   "pandas>=2.1",
# ]
# ///
"""
Download the WaltonFuture/Multimodal-Cold-Start dataset, associate MSC domain labels,
create an Excel export, and optionally push the augmented dataset to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download WaltonFuture/Multimodal-Cold-Start, add domain labels from the provided Excel "
            "sheet, export the merged table, and optionally push the result to Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--labels-file",
        default="msc/msc_labels_all51k.xlsx",
        help="Path to the Excel file containing example_id -> domain mappings (default: %(default)s).",
    )
    parser.add_argument(
        "--output-excel",
        default="msc/multimodal_cold_start_with_domains.xlsx",
        help="Path for the generated Excel workbook (default: %(default)s).",
    )
    parser.add_argument(
        "--source-dataset",
        default="WaltonFuture/Multimodal-Cold-Start",
        help="Source dataset on Hugging Face Hub (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to download from the source dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-repo",
        help="Target Hugging Face dataset repo ID (e.g., username/dataset-name). Required with --push.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the augmented dataset to Hugging Face Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the target repo as private when pushing.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add domain annotations sourced from MSC labels",
        help="Commit message used when pushing to Hugging Face Hub.",
    )
    return parser.parse_args()


def read_label_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label sheet not found: {path}")

    df = pd.read_excel(path)
    required_columns = {"example_id", "domain"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Label sheet {path} is missing required columns: {', '.join(sorted(missing))}")

    if df["example_id"].duplicated().any():
        duplicates = df.loc[df["example_id"].duplicated(), "example_id"].tolist()
        raise ValueError(f"Duplicate example_id values found in {path}: {duplicates[:5]}")

    df = df.copy()
    df["example_id"] = df["example_id"].astype(int)

    domain_series = df["domain"].astype("string")
    domain_series = domain_series.fillna("").str.strip()
    empty_mask = domain_series == ""
    if empty_mask.any():
        print(f"Found {empty_mask.sum()} unlabeled rows; setting their domain to 'Empty'.")
        domain_series[empty_mask] = "Empty"
    df["domain"] = domain_series

    return df


def align_labels_with_dataset(dataset: Dataset, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Return labels_df reordered to match dataset rows."""
    df = labels_df.copy()
    dataset_size = len(dataset)

    if "example_id" in dataset.column_names:
        dataset_ids = dataset["example_id"]
        df = df.set_index("example_id").reindex(dataset_ids)
        if df.isna().any().any():
            missing = df[df.isna().any(axis=1)].index.tolist()
            raise ValueError(f"Domain labels missing for dataset example_ids: {missing[:5]}")
        df = df.reset_index()
    else:
        df = df.sort_values("example_id").reset_index(drop=True)
        expected_ids = list(range(dataset_size))
        if df["example_id"].tolist() != expected_ids:
            raise ValueError(
                "Source dataset does not expose `example_id` but the label sheet order does not match "
                "the implicit example ordering (expected example_id to equal the row index)."
            )

    if len(df) != dataset_size:
        raise ValueError(
            f"Dataset size ({dataset_size}) and label sheet size ({len(df)}) differ."
        )

    return df


def add_domain_column(dataset: Dataset, ordered_labels: pd.DataFrame) -> Dataset:
    domains: List[str] = ordered_labels["domain"].astype(str).tolist()
    return dataset.add_column("domain", domains)


def export_excel(path: Path, dataset: Dataset, ordered_labels: pd.DataFrame) -> None:
    records = {
        "example_id": ordered_labels["example_id"],
    }
    if "msc_code" in ordered_labels.columns:
        records["msc_code"] = ordered_labels["msc_code"]
    records.update(
        {
            "domain": ordered_labels["domain"],
            "problem": dataset["problem"],
            "answer": dataset["answer"],
        }
    )

    export_df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(path, index=False)
    print(f"Saved Excel export with {len(export_df)} rows to {path}")


def push_dataset_to_hub(dataset: Dataset, repo_id: str, commit_message: str, private: bool) -> None:
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(repo_id, private=private, commit_message=commit_message)
    print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels_file)
    output_excel = Path(args.output_excel)

    labels_df = read_label_sheet(labels_path)
    print(f"Loaded {len(labels_df)} domain labels from {labels_path}")

    print(f"Downloading split '{args.split}' from {args.source_dataset}...")
    dataset = load_dataset(args.source_dataset, split=args.split)
    print(f"Loaded {len(dataset)} samples from {args.source_dataset}:{args.split}")

    ordered_labels = align_labels_with_dataset(dataset, labels_df)
    augmented_dataset = add_domain_column(dataset, ordered_labels)
    print("Attached domain labels to the dataset.")

    export_excel(output_excel, augmented_dataset, ordered_labels)

    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf-repo must be provided when using --push.")
        print(f"Pushing dataset with domain annotations to {args.hf_repo}...")
        push_dataset_to_hub(
            augmented_dataset,
            repo_id=args.hf_repo,
            commit_message=args.commit_message,
            private=args.private,
        )


if __name__ == "__main__":
    main()
