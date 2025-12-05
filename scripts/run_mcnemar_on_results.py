#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.1",
#   "openpyxl>=3.1",
# ]
# ///
"""
Fetch *_scored.xlsx files from two VLMEvalKit benchmarking runs and run McNemar's
test for every benchmark present in both runs (and overall across benchmarks).

The script optionally rsyncs the remote directories before analysis.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from scipy.stats import binomtest  # type: ignore
except Exception:  # pragma: no cover - SciPy optional
    binomtest = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run McNemar's test across all *_scored.xlsx pairs found in two benchmark run directories."
        )
    )
    parser.add_argument("run_a", help="Directory for run A (local path or remote path on --remote-host).")
    parser.add_argument("run_b", help="Directory for run B (local path or remote path on --remote-host).")
    parser.add_argument(
        "--remote-host",
        default="koa",
        help="SSH/rsync host alias for fetching remote directories (default: %(default)s). "
             "If the provided path already exists locally, rsync is skipped.",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/mcnemar_cache",
        help="Local directory used to store rsynced results (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for McNemar's test (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete cached directories after the script finishes.",
    )
    parser.add_argument(
        "--benchmark-filter",
        action="append",
        help="If provided, only benchmarks containing the given substring will be analyzed (can be repeated).",
    )
    return parser.parse_args()


def prepare_run_dir(path_str: str, cache_root: Path, remote_host: str) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path.resolve()

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path_str.strip("/")[-120:])
    local_dir = cache_root / safe_name
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_spec = f"{remote_host}:{path_str.rstrip('/')}/"
    rsync_cmd = [
        "rsync",
        "-av",
        "--include",
        "*/",
        "--include",
        "*_scored.xlsx",
        "--exclude",
        "*",
        remote_spec,
        str(local_dir),
    ]
    print(f"Syncing {remote_spec} -> {local_dir} ...")
    subprocess.run(rsync_cmd, check=True)
    return local_dir


def extract_benchmark_name(filename: str) -> str:
    match = re.search(r"checkpoints_(.+?)_scored", filename)
    if match:
        return match.group(1)
    stem = Path(filename).stem
    if "_scored" in stem:
        stem = stem.split("_scored")[0]
    if "checkpoints_" in stem:
        stem = stem.split("checkpoints_")[-1]
    return stem


def collect_scored_files(run_dir: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(run_dir.rglob("*_scored.xlsx")):
        benchmark = extract_benchmark_name(path.name)
        duplicates[benchmark].append(path)

    for benchmark, paths in duplicates.items():
        if len(paths) > 1:
            raise RuntimeError(
                f"Multiple *_scored.xlsx files found for benchmark '{benchmark}' in {run_dir}:\n"
                + "\n".join(f"  - {p}" for p in paths)
            )
        files[benchmark] = paths[0]
    return files


def load_hits(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path)
    if "index" in df.columns:
        id_col = "index"
    elif "Unnamed: 0" in df.columns:
        id_col = "Unnamed: 0"
    else:
        raise ValueError(f"Could not find an 'index' column in {path}")

    if "hit" not in df.columns:
        raise ValueError(f"Column 'hit' not found in {path}")

    subset = df[[id_col, "hit"]].copy()
    subset = subset.rename(columns={id_col: "index"})
    subset = subset.drop_duplicates(subset=["index"])
    subset["hit"] = subset["hit"].astype(float)

    mask = subset["hit"].isin({0.0, 1.0})
    skipped = (~mask).sum()
    if skipped:
        print(
            f"Warning: {path.name} contains {skipped} non-binary hit values; "
            "skipping them for McNemar's test."
        )

    binary = subset[mask].copy()
    binary["hit"] = binary["hit"].astype(int)
    return binary, subset.copy()


def mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if binomtest is not None:
        test = binomtest(min(b, c), n, 0.5, alternative="two-sided")
        return min(1.0, test.pvalue)
    stat = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(stat / 2))


def analyze_pair(
    bench: str, path_a: Path, path_b: Path
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hits_a_binary, hits_a_full = load_hits(path_a)
    hits_b_binary, hits_b_full = load_hits(path_b)
    merged = hits_a_binary.merge(hits_b_binary, on="index", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError(f"No overlapping indices for {bench}")

    acc_a = hits_a_full["hit"].mean()
    acc_b = hits_b_full["hit"].mean()
    b = int(((merged["hit_a"] == 1) & (merged["hit_b"] == 0)).sum())
    c = int(((merged["hit_a"] == 0) & (merged["hit_b"] == 1)).sum())
    p_val = mcnemar_pvalue(b, c)

    result = {
        "benchmark": bench,
        "n_shared": len(merged),
        "n_a": len(hits_a_full),
        "n_b": len(hits_b_full),
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "b": b,
        "c": c,
        "p_value": p_val,
        "delta": acc_b - acc_a,
    }
    return result, merged, hits_a_full, hits_b_full


def format_result(res: Dict[str, float], alpha: float) -> str:
    sig = "YES" if res["p_value"] < alpha else "no"
    return (
        f"{res['benchmark']:<35} "
        f"shared={int(res['n_shared']):5d} "
        f"nA={int(res['n_a']):5d} "
        f"nB={int(res['n_b']):5d} "
        f"accA={res['accuracy_a']:.3f} "
        f"accB={res['accuracy_b']:.3f} "
        f"Δ={res['delta']:+.3f} "
        f"b={int(res['b']):4d} c={int(res['c']):4d} "
        f"p={res['p_value']:.4g} "
        f"sig<{alpha}? {sig}"
    )


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for run_path in (args.run_a, args.run_b):
        prepared = prepare_run_dir(run_path, cache_root, args.remote_host)
        run_dirs.append(prepared)

    if args.benchmark_filter:
        filters = [f.lower() for f in args.benchmark_filter]
    else:
        filters = None

    files_a = collect_scored_files(run_dirs[0])
    files_b = collect_scored_files(run_dirs[1])

    common_benchmarks = sorted(set(files_a) & set(files_b))
    if filters:
        common_benchmarks = [
            bench for bench in common_benchmarks if any(f in bench.lower() for f in filters)
        ]
    if not common_benchmarks:
        raise RuntimeError("No overlapping *_scored.xlsx files found between the two runs.")

    print(f"Found {len(common_benchmarks)} common benchmarks.")
    merged_frames: List[pd.DataFrame] = []
    total_n_a = 0
    total_n_b = 0
    total_hits_a = 0
    total_hits_b = 0

    for bench in common_benchmarks:
        res, merged, hits_a_full, hits_b_full = analyze_pair(bench, files_a[bench], files_b[bench])
        total_n_a += len(hits_a_full)
        total_hits_a += float(hits_a_full["hit"].sum())
        total_n_b += len(hits_b_full)
        total_hits_b += float(hits_b_full["hit"].sum())
        merged_frames.append(merged)
        print(format_result(res, args.alpha))

    overall = pd.concat(merged_frames, ignore_index=True)
    b_total = int(((overall["hit_a"] == 1) & (overall["hit_b"] == 0)).sum())
    c_total = int(((overall["hit_a"] == 0) & (overall["hit_b"] == 1)).sum())
    overall_acc_a = total_hits_a / total_n_a if total_n_a else float("nan")
    overall_acc_b = total_hits_b / total_n_b if total_n_b else float("nan")
    overall_res = {
        "benchmark": "OVERALL",
        "n_shared": len(overall),
        "n_a": total_n_a,
        "n_b": total_n_b,
        "accuracy_a": overall_acc_a,
        "accuracy_b": overall_acc_b,
        "b": b_total,
        "c": c_total,
        "p_value": mcnemar_pvalue(b_total, c_total),
        "delta": overall_acc_b - overall_acc_a,
    }
    print("-" * 120)
    print(format_result(overall_res, args.alpha))

    if not args.keep_cache:
        for run_dir in run_dirs:
            if run_dir.is_relative_to(cache_root):
                shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
