#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.0",
#   "python-dotenv>=1.0",
# ]
# ///
"""
Classify TSV-based benchmark questions into MSC domains with GPT-5-mini.

Usage example:
    python msc/categorize_tsv_questions.py --input-path path/to/data.tsv --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You are an expert classifier that assigns top-level MSC domains to math "
    "questions. Always return the requested schema before anything else."
)

MSC_PROMPT_TEMPLATE = """
You are a classifier. Your task is to assign the most relevant high-level
Mathematics Subject Classification (MSC) domain to the given question. You
must return only one domain.

Use the top-level MSC domains:
00 General and overarching topics; collections
01 History and biography
03 Mathematical logic and foundations
05 Combinatorics
06 Order, lattices, ordered algebraic structures
08 General algebraic systems
11 Number theory
12 Field theory and polynomials
13 Commutative algebra
14 Algebraic geometry
15 Linear and multilinear algebra; matrix theory
16 Associative rings and algebras
17 Nonassociative rings and algebras
18 Category theory; homological algebra
19 K-theory
20 Group theory and generalizations
22 Topological groups, Lie groups
26 Real functions
28 Measure and integration
30 Functions of a complex variable
31 Potential theory
32 Several complex variables and analytic spaces
33 Special functions
34 Ordinary differential equations
35 Partial differential equations
37 Dynamical systems and ergodic theory
39 Difference and functional equations
40 Sequences, series, summability
41 Approximations and expansions
42 Harmonic analysis on Euclidean spaces
43 Abstract harmonic analysis
44 Integral transforms, operational calculus
45 Integral equations
46 Functional analysis
47 Operator theory
49 Calculus of variations and optimal control; optimization
51 Geometry
52 Convex and discrete geometry
53 Differential geometry
54 General topology
55 Algebraic topology
57 Manifolds and cell complexes
58 Global analysis, analysis on manifolds
60 Probability theory and stochastic processes
62 Statistics
65 Numerical analysis
68 Computer science
70 Mechanics of particles and systems
74 Mechanics of deformable solids
76 Fluid mechanics
78 Optics, electromagnetic theory
80 Classical thermodynamics, heat transfer
81 Quantum theory
82 Statistical mechanics, structure of matter
83 Relativity and gravitational theory
85 Astronomy and astrophysics
86 Geophysics
90 Operations research, mathematical programming
91 Game theory, economics, social and behavioral sciences
92 Biology and other natural sciences
93 Systems theory; control
94 Information and communication, circuits
97 Mathematics education
98 None of above

Given a question, identify which MSC domain it belongs to. Consider the core
mathematical idea involved.

Return your answer in this exact format:
MSC code: <MSC code>
Domain: <Domain name>

Question:
{question}
""".strip()


@dataclass
class QuestionRecord:
    record_id: str
    tsv_index: int
    question: str
    shuffled_rank: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Categorize TSV questions with GPT-5-mini."
    )
    parser.add_argument(
        "--input-path",
        default="msc/LiveXiv_TQA_v4_local.tsv",
        help="Path to the TSV file containing questions (expects a 'question' column).",
    )
    parser.add_argument(
        "--output-path",
        default="msc/msc_labels_from_tsv.jsonl",
        help="JSONL file that accumulates MSC labels (created if missing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample size after shuffling (useful for debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for deterministic shuffling before applying the limit.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Number of concurrent API calls to run.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Chat completion model to query.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retry budget per sample before giving up.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay (seconds) for exponential backoff between retries.",
    )
    return parser.parse_args()


def load_questions(tsv_path: Path) -> List[QuestionRecord]:
    records: List[QuestionRecord] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            record_id = row.get("index") or str(row_idx)
            records.append(
                QuestionRecord(
                    record_id=str(record_id),
                    tsv_index=row_idx,
                    question=question,
                )
            )
    return records


def load_existing(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    processed: Dict[str, Dict[str, Any]] = {}
    with output_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                print(
                    f"[warn] Skipping malformed JSON at {output_path}:{line_num}",
                )
                continue
            record_id = str(payload.get("record_id") or "")
            if not record_id:
                continue
            processed[record_id] = payload
    return processed


def parse_msc_response(text: str) -> tuple[str, str]:
    msc_code = ""
    domain = ""
    for line in text.splitlines():
        trimmed = line.strip()
        lower = trimmed.lower()
        if lower.startswith("msc code"):
            payload = trimmed.split(":", 1)[1].strip()
            msc_code = payload.split()[0]
        elif lower.startswith("domain"):
            domain = trimmed.split(":", 1)[1].strip()
    return msc_code, domain


async def classify_question(
    client: AsyncOpenAI,
    record: QuestionRecord,
    model: str,
    max_retries: int,
    retry_delay: float,
) -> Dict[str, Any]:
    """
    Execute the chat completion call with retries and return the parsed payload.
    """
    user_prompt = MSC_PROMPT_TEMPLATE.format(question=record.question)
    attempt = 0
    while True:
        attempt += 1
        start = perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency = perf_counter() - start
            raw_content = response.choices[0].message.content.strip()
            msc_code, domain = parse_msc_response(raw_content)
            return {
                "record_id": record.record_id,
                "tsv_index": record.tsv_index,
                "shuffled_rank": record.shuffled_rank,
                "question": record.question,
                "msc_code": msc_code,
                "domain": domain,
                "raw_response": raw_content,
                "model": model,
                "latency_sec": latency,
            }
        except Exception as exc:  # broad exception is intentional for retries
            if attempt > max_retries:
                raise exc
            wait_time = retry_delay * (2 ** (attempt - 1))
            print(
                f"[warn] {record.record_id} attempt {attempt} failed ({exc}); "
                f"retrying in {wait_time:.1f}s"
            )
            await asyncio.sleep(wait_time)


async def process_records(args: argparse.Namespace) -> None:
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    records = load_questions(input_path)
    if not records:
        print(f"No questions found under {input_path}")
        return

    rng = random.Random(args.seed)
    rng.shuffle(records)

    if args.limit is not None:
        records = records[: args.limit]

    for rank, record in enumerate(records):
        record.shuffled_rank = rank

    existing = load_existing(output_path)
    pending = [record for record in records if record.record_id not in existing]

    print(
        f"Loaded {len(records)} questions (limit={args.limit}). "
        f"{len(existing)} existing labels, {len(pending)} to process."
    )

    if not pending:
        print("Nothing to do. All selected questions already labeled.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_workers)

    async def worker(record: QuestionRecord) -> Dict[str, Any]:
        async with semaphore:
            payload = await classify_question(
                client=client,
                record=record,
                model=args.model,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            return payload

    tasks = [asyncio.create_task(worker(record)) for record in pending]

    completed = 0
    failures = 0
    with output_path.open("a", encoding="utf-8") as sink:
        for coro in asyncio.as_completed(tasks):
            try:
                payload = await coro
            except Exception as exc:
                failures += 1
                print(f"[error] Task failed permanently: {exc}")
                continue

            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sink.flush()

            completed += 1
            rid = payload["record_id"]
            msc_code = payload["msc_code"] or "?"
            domain = payload["domain"] or "?"
            print(
                f"[{completed}/{len(pending)}] {rid} "
                f"-> MSC {msc_code} | {domain}"
            )

    if failures:
        print(f"Completed {completed} samples with {failures} failures.")
    else:
        print(f"Completed all {completed} samples successfully.")


def main() -> None:
    # Load API keys and friends automatically when a .env file is present.
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)

    args = parse_args()
    try:
        asyncio.run(process_records(args))
    except KeyboardInterrupt:
        print("Interrupted by user. Partial progress preserved.")


if __name__ == "__main__":
    main()
