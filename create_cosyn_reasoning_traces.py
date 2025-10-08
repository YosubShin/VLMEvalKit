#!/usr/bin/env python3
"""
Augment the `yosubshin/cosyn` dataset by generating a reasoning trace for each sample
using a GPT-4o-compatible vision API. The script downloads the dataset from the Hub,
invokes the API with (image, problem) and a fixed instruction prompt, and appends a
new `reasoning_trace` string field. It can save locally and optionally push to the Hub.

Usage:
  python create_cosyn_reasoning_traces.py \
    --repo-in yosubshin/cosyn \
    --repo-out yosubshin/cosyn-with-traces \
    --split train \
    --model gpt-4o-2024-08-06 \
    --push

Environment:
  OPENAI_API_KEY (or Azure equivalent if using Azure mode)
"""

import argparse
import os
import tempfile
from typing import Optional

from datasets import load_dataset, DatasetDict, Features, Value, Image as HFImage

from vlmeval.api.gpt import OpenAIWrapper


DEFAULT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_wrapper(model: str, temperature: float, max_tokens: int, img_size: int, img_detail: str,
                  use_azure: bool = False) -> OpenAIWrapper:
    return OpenAIWrapper(
        model=model,
        retry=5,
        wait=5,
        key=None,  # from env
        verbose=False,
        system_prompt=None,
        temperature=temperature,
        timeout=60,
        api_base=None,
        max_tokens=max_tokens,
        img_size=img_size,
        img_detail=img_detail,
        use_azure=use_azure,
    )


def _save_pil_to_tmp_path(image_pil) -> str:
    # Persist PIL image to a temporary file; wrapper expects a file path
    if hasattr(image_pil, 'mode') and image_pil.mode in ('RGBA', 'P', 'LA'):
        image_pil = image_pil.convert('RGB')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image_pil.save(tmp.name, format='JPEG')
    return tmp.name


def make_messages(image_pil, problem: str, instruction: str):
    # Interleaved: image then text prompt
    # Compatible with vlmeval BaseAPI message format
    image_path = _save_pil_to_tmp_path(image_pil)
    content = [
        {"type": "image", "value": image_path},
        {"type": "text", "value": f"Problem: {problem}\n\nInstruction: {instruction}"},
    ]
    return content


def generate_trace(wrapper: OpenAIWrapper, image_pil, problem: str, instruction: str) -> str:
    content = make_messages(image_pil, problem, instruction)
    answer = wrapper.generate(content, max_tokens=2048)
    return answer


def process_split(ds, wrapper: OpenAIWrapper, instruction: str, image_col: str, problem_col: str,
                  output_col: str, limit: Optional[int] = None):
    # ds: datasets.Dataset
    num_rows = len(ds) if limit is None else min(limit, len(ds))

    def _map_fn(example, idx):
        # Dataset image type is datasets.Image, which yields a PIL.Image on read
        image_pil = example[image_col]
        problem = example.get(problem_col, "")
        try:
            trace = generate_trace(wrapper, image_pil, problem, instruction)
        except Exception as e:
            trace = f"[ERROR] {type(e).__name__}: {e}"
        example[output_col] = trace
        return example

    # Use map with with_indices to pass idx if needed (e.g., rate limiting, logging)
    processed = ds.select(range(num_rows)).map(_map_fn, with_indices=True)
    return processed


def main():
    parser = argparse.ArgumentParser(description='Generate reasoning traces for yosubshin/cosyn and append as a new column.')
    parser.add_argument('--repo-in', default='yosubshin/cosyn', help='Input HF dataset repo id')
    parser.add_argument('--repo-out', default='yosubshin/cosyn-with-traces', help='Output HF dataset repo id')
    parser.add_argument('--split', default='train', help='Dataset split to process')
    parser.add_argument('--image-col', default='image', help='Name of image column')
    parser.add_argument('--problem-col', default='problem', help='Name of problem/question column')
    parser.add_argument('--output-col', default='reasoning_trace', help='Name of new output column')
    parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION, help='Instruction appended to the prompt')
    parser.add_argument('--model', default='gpt-4o-2024-08-06', help='OpenAI vision model name')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--img-detail', default='low', choices=['low', 'high'])
    parser.add_argument('--azure', action='store_true', help='Use Azure OpenAI endpoint via env')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for a quick run')
    parser.add_argument('--push', action='store_true', help='Push to Hub after processing')
    args = parser.parse_args()

    print(f"Loading dataset {args.repo_in} (split={args.split})...")
    ds_dict = load_dataset(args.repo_in)
    if args.split not in ds_dict:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds_dict.keys())}")

    # Build API wrapper
    wrapper = build_wrapper(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        img_size=args.img_size,
        img_detail=args.img_detail,
        use_azure=args.azure,
    )

    print("Generating reasoning traces...")
    processed_split = process_split(
        ds_dict[args.split],
        wrapper=wrapper,
        instruction=args.instruction,
        image_col=args.image_col,
        problem_col=args.problem_col,
        output_col=args.output_col,
        limit=args.limit,
    )

    # Preserve other splits as-is (unprocessed) if present, only replace the working split
    out_dd = DatasetDict({k: v for k, v in ds_dict.items()})
    out_dd[args.split] = processed_split

    # Save locally first
    local_dir = f"./{args.repo_out.split('/')[-1]}_local"
    print(f"Saving locally to {local_dir}...")
    out_dd.save_to_disk(local_dir)
    print("Saved.")

    # Optional push to hub
    if args.push:
        print(f"Pushing to Hub: {args.repo_out}")
        out_dd.push_to_hub(args.repo_out, private=False, commit_message="Add reasoning_trace generated by GPT-4o")
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_out}")


if __name__ == '__main__':
    main()


