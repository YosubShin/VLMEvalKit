#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.19",
#   "openpyxl>=3.1",
#   "pandas>=2.1",
# ]
# ///
"""
Attach MSC domain labels to the oumi-ai/walton-multimodal-cold-start-r1-format dataset by
matching on the (problem, answer/solution) pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match domains from the full MSC label sheet to the oumi R1 dataset."
    )
    parser.add_argument(
        "--base-excel",
        default="msc/multimodal_cold_start_with_domains.xlsx",
        help="Excel file containing problem/answer/domain columns (default: %(default)s).",
    )
    parser.add_argument(
        "--source-dataset",
        default="oumi-ai/walton-multimodal-cold-start-r1-format",
        help="Hugging Face dataset ID to annotate (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: %(default)s).",
    )
    parser.add_argument(
        "--output-excel",
        default="msc/oumi_walton_r1_with_domains.xlsx",
        help="Path to the Excel file that will store the annotated dataset (default: %(default)s).",
    )
    return parser.parse_args()


def load_base_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base Excel file not found: {path}")

    df = pd.read_excel(path)
    required = {"problem", "answer", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Base Excel missing columns: {', '.join(sorted(missing))}")

    grouped = df.groupby(["problem", "answer"])["domain"].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        example_pair = conflicts.index[0]
        print(
            f"Found {len(conflicts)} problem+answer pairs with conflicting domains "
            f"(e.g., {example_pair}). Resolving by majority vote (ties -> lowest example_id)."
        )

    def resolve_group(group: pd.DataFrame) -> pd.Series:
        if "example_id" in group.columns:
            stats = (
                group.groupby("domain")
                .agg(count=("domain", "size"), first_example=("example_id", "min"))
                .sort_values(["count", "first_example"], ascending=[False, True])
            )
        else:
            stats = (
                group.groupby("domain")
                .agg(count=("domain", "size"))
                .sort_values("count", ascending=False)
            )
        chosen_domain = stats.index[0]
        filtered = group[group["domain"] == chosen_domain]
        if "example_id" in filtered.columns:
            filtered = filtered.sort_values("example_id")
        return filtered.iloc[0]

    dedup = (
        df.groupby(["problem", "answer"], group_keys=False)
        .apply(resolve_group)
        .reset_index(drop=True)
    )
    dedup = dedup.rename(columns={"answer": "solution"})
    return dedup


def annotate_dataset(base_df: pd.DataFrame, dataset_id: str, split: str) -> pd.DataFrame:
    dataset = load_dataset(dataset_id, split=split)
    df = dataset.to_pandas()

    merge_columns = ["problem", "solution", "domain"]
    if "example_id" in base_df.columns:
        merge_columns.append("example_id")

    # Merge on (problem, solution)
    merged = df.merge(
        base_df[merge_columns],
        on=["problem", "solution"],
        how="left",
    )

    missing = merged["domain"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows could not be matched to a domain.")
        sample = merged[merged["domain"].isna()].head()
        for _, row in sample.iterrows():
            print("Unmatched problem snippet:", str(row["problem"])[:120].replace("\n", " "))
        merged["domain"] = merged["domain"].fillna("(blank)")

    return merged


def export_excel(df: pd.DataFrame, path: Path) -> None:
    columns_to_save = [
        col
        for col in [
            "example_id",
            "domain",
            "problem",
            "solution",
            "original_question",
            "original_answer",
        ]
        if col in df.columns
    ]
    export_df = df[columns_to_save].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(path, index=False)
    print(f"Wrote {len(export_df)} annotated rows to {path}")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_excel)
    output_path = Path(args.output_excel)

    base_df = load_base_table(base_path)
    merged_df = annotate_dataset(base_df, args.source_dataset, args.split)
    export_excel(merged_df, output_path)


if __name__ == "__main__":
    main()
