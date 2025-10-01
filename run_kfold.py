#!/usr/bin/env python3
"""
K-fold inference script for VLMEvalKit.

This script runs inference k times for each prompt to assess question difficulty
and reliability. Each prompt gets k different responses which are evaluated
individually, then aggregated with a verdict_sum.

Usage:
    python run_kfold.py --data WaltonMultimodalReasoning --model qwen2_vl --k 8
"""

import os
import sys
import torch
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
import os.path as osp
from uuid import uuid4

# VLMEvalKit imports
from vlmeval import *
from vlmeval.dataset import build_dataset
from vlmeval.tools import LOAD_DATASET, abbr2full, logger
from vlmeval.utils import TSVDataset, track_progress_rich, dataset_URLs
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer
from vlmeval.tools import DATASET_TYPE, abbr2full, LOAD_DATASET
from vlmeval.config import supported_VLM
from vlmeval.utils.arguments import build_parser
from vlmeval.smp import *


def parse_args():
    """Parse command line arguments for k-fold inference."""
    parser = build_parser()

    # Add k-fold specific arguments
    parser.add_argument('--k', type=int, default=8,
                        help='Number of times to run inference per prompt (default: 8)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for sampling variation (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p for nucleus sampling (default: 0.9)')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed for reproducibility (actual seed = base + k_iteration)')

    args = parser.parse_args()

    # Validate arguments
    if args.k < 2:
        logger.error("K must be at least 2 for meaningful k-fold inference")
        sys.exit(1)

    return args


def build_model(model_name, **kwargs):
    """Build and return the model for inference."""
    if model_name in supported_VLM:
        model_cls = supported_VLM[model_name]
        model = model_cls(**kwargs)

        # Ensure model has temperature settings
        if hasattr(model, 'temperature'):
            model.temperature = kwargs.get('temperature', 0.7)
        if hasattr(model, 'top_p'):
            model.top_p = kwargs.get('top_p', 0.9)

        return model
    else:
        logger.error(f"Model {model_name} not supported")
        raise ValueError(f"Model {model_name} not supported")


def infer_kfold(model, dataset, k=8, temperature=0.7, top_p=0.9, seed_base=42,
                work_dir='./outputs', verbose=False):
    """
    Run k-fold inference on a dataset.

    Args:
        model: The VLM model to use
        dataset: The dataset to evaluate
        k: Number of inference iterations per prompt
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        seed_base: Base seed for reproducibility
        work_dir: Directory to save outputs
        verbose: Whether to print verbose output

    Returns:
        dict: Results with k predictions per index
    """
    dataset_name = dataset.dataset_name
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else str(model)

    logger.info(f"Starting k-fold inference with k={k}")
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    logger.info(f"Temperature: {temperature}, Top-p: {top_p}")

    # Prepare output file
    os.makedirs(work_dir, exist_ok=True)
    output_file = osp.join(work_dir, f'{model_name}_{dataset_name}_k{k}.pkl')

    # Load existing results if any (for resumption)
    if osp.exists(output_file):
        logger.info(f"Loading existing results from {output_file}")
        results = load(output_file)
    else:
        results = {}

    # Get dataset data
    data = dataset.data
    total_items = len(data)

    # Progress bar for overall completion
    pbar = tqdm(total=total_items, desc=f'K-fold Inference (k={k})')

    for idx_num, row in data.iterrows():
        index = row['index']

        # Skip if already completed
        if index in results and len(results[index]['predictions']) == k:
            pbar.update(1)
            continue

        # Initialize result structure for this index
        if index not in results:
            results[index] = {
                'question': row.get('question', ''),
                'answer': row.get('answer', ''),
                'predictions': [],
                'temperatures': [],
                'seeds': []
            }

        # Build prompt once
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            prompt_struct = model.build_prompt(row, dataset=dataset_name)
        else:
            prompt_struct = dataset.build_prompt(row)

        # Generate k predictions
        existing_preds = len(results[index]['predictions'])
        for k_iter in range(existing_preds, k):
            # Set seed for reproducibility
            current_seed = seed_base + k_iter
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)
            torch.manual_seed(current_seed)

            # Temporarily set model parameters if possible
            original_temp = getattr(model, 'temperature', None)
            original_top_p = getattr(model, 'top_p', None)

            try:
                if hasattr(model, 'temperature'):
                    model.temperature = temperature
                if hasattr(model, 'top_p'):
                    model.top_p = top_p

                # Generate response
                response = model.generate(message=prompt_struct, dataset=dataset_name)

                # Store result
                results[index]['predictions'].append(response)
                results[index]['temperatures'].append(temperature)
                results[index]['seeds'].append(current_seed)

                if verbose:
                    print(f"Index {index}, Iteration {k_iter+1}/{k}: {response[:100]}...")

            finally:
                # Restore original parameters
                if original_temp is not None and hasattr(model, 'temperature'):
                    model.temperature = original_temp
                if original_top_p is not None and hasattr(model, 'top_p'):
                    model.top_p = original_top_p

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save progress periodically
        if (pbar.n + 1) % 10 == 0:
            dump(results, output_file)

        pbar.update(1)

    pbar.close()

    # Final save
    dump(results, output_file)
    logger.info(f"K-fold inference complete. Results saved to {output_file}")

    return results


def convert_to_dataframe(kfold_results, k):
    """
    Convert k-fold results to a DataFrame format suitable for evaluation.

    Args:
        kfold_results: Dictionary with k predictions per index
        k: Number of predictions per index

    Returns:
        pd.DataFrame: DataFrame with separate columns for each prediction
    """
    rows = []

    for index, data in kfold_results.items():
        row = {
            'index': index,
            'question': data['question'],
            'answer': data['answer']
        }

        # Add each prediction as a separate column
        for i in range(k):
            if i < len(data['predictions']):
                row[f'prediction_{i+1}'] = data['predictions'][i]
            else:
                row[f'prediction_{i+1}'] = ''

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by index for consistency
    df = df.sort_values('index').reset_index(drop=True)

    return df


def evaluate_kfold(dataset, df_predictions, k, work_dir='./outputs', **judge_kwargs):
    """
    Evaluate k-fold predictions using the dataset's judge.

    Args:
        dataset: The dataset object with evaluate method
        df_predictions: DataFrame with k predictions per row
        k: Number of predictions
        work_dir: Working directory for temporary files
        judge_kwargs: Arguments for the judge model

    Returns:
        pd.DataFrame: DataFrame with verdicts for each prediction
    """
    dataset_name = dataset.dataset_name
    logger.info(f"Evaluating k-fold predictions for {dataset_name}")

    results = df_predictions.copy()

    # Evaluate each prediction column separately
    for i in range(k):
        pred_col = f'prediction_{i+1}'
        verdict_col = f'verdict_{i+1}'

        logger.info(f"Evaluating prediction {i+1}/{k}")

        # Create temporary dataframe for evaluation
        temp_df = df_predictions[['index', 'question', 'answer']].copy()
        temp_df['prediction'] = df_predictions[pred_col]

        # Save to temporary file
        temp_file = osp.join(work_dir, f'temp_eval_{uuid4()}.xlsx')
        temp_df.to_excel(temp_file, index=False)

        try:
            # Use dataset's evaluate method
            eval_result = dataset.evaluate(temp_file, **judge_kwargs)

            # Extract verdicts
            if isinstance(eval_result, pd.DataFrame):
                # Merge verdict into results
                if 'verdict' in eval_result.columns:
                    results[verdict_col] = eval_result['verdict'].values
                else:
                    logger.warning(f"No verdict column found for prediction {i+1}")
                    results[verdict_col] = 0
            else:
                logger.warning(f"Unexpected evaluation result format for prediction {i+1}")
                results[verdict_col] = 0

        finally:
            # Clean up temp file
            if osp.exists(temp_file):
                os.remove(temp_file)

    # Calculate verdict_sum (total correct out of k)
    verdict_columns = [f'verdict_{i+1}' for i in range(k)]
    results['verdict_sum'] = results[verdict_columns].sum(axis=1)

    # Add statistics columns
    results['verdict_mean'] = results['verdict_sum'] / k
    results['difficulty'] = pd.cut(results['verdict_mean'],
                                   bins=[0, 0.25, 0.75, 1.0],
                                   labels=['hard', 'medium', 'easy'])

    logger.info(f"Evaluation complete. Verdict distribution:")
    logger.info(f"{results['verdict_sum'].value_counts().sort_index()}")

    return results


def main():
    """Main function for k-fold inference."""
    args = parse_args()

    # Disable warnings if requested
    if args.no_warning:
        warnings.filterwarnings('ignore')

    # Setup work directory
    work_dir = args.work_dir if args.work_dir else './outputs'
    os.makedirs(work_dir, exist_ok=True)

    # Get full model and dataset names
    model_name = abbr2full(args.model)
    dataset_names = args.data

    # Convert single dataset to list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    logger.info(f"Starting k-fold inference with k={args.k}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Datasets: {dataset_names}")

    # Build model
    model_kwargs = {}
    if args.verbose:
        model_kwargs['verbose'] = True
    if args.temperature:
        model_kwargs['temperature'] = args.temperature
    if args.top_p:
        model_kwargs['top_p'] = args.top_p

    model = build_model(model_name, **model_kwargs)

    # Process each dataset
    for dataset_name in dataset_names:
        logger.info(f"\nProcessing dataset: {dataset_name}")

        # Build dataset
        dataset = build_dataset(dataset_name)

        # Run k-fold inference
        kfold_results = infer_kfold(
            model=model,
            dataset=dataset,
            k=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            seed_base=args.seed_base,
            work_dir=work_dir,
            verbose=args.verbose
        )

        # Convert to DataFrame
        df_predictions = convert_to_dataframe(kfold_results, args.k)

        # Save predictions
        pred_file = osp.join(work_dir, f'{model_name}_{dataset_name}_k{args.k}_predictions.xlsx')
        df_predictions.to_excel(pred_file, index=False)
        logger.info(f"Predictions saved to {pred_file}")

        # Evaluate if judge is available
        if hasattr(dataset, 'evaluate'):
            judge_kwargs = {
                'model': args.judge if args.judge else 'gpt-4o-mini',
                'nproc': args.nproc,
                'verbose': args.verbose
            }

            df_evaluated = evaluate_kfold(
                dataset=dataset,
                df_predictions=df_predictions,
                k=args.k,
                work_dir=work_dir,
                **judge_kwargs
            )

            # Save evaluated results
            eval_file = osp.join(work_dir, f'{model_name}_{dataset_name}_k{args.k}_evaluated.xlsx')
            df_evaluated.to_excel(eval_file, index=False)
            logger.info(f"Evaluated results saved to {eval_file}")

            # Print summary statistics
            logger.info("\n" + "="*60)
            logger.info("K-FOLD EVALUATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total questions: {len(df_evaluated)}")
            logger.info(f"K value: {args.k}")
            logger.info("\nDifficulty distribution:")
            logger.info(df_evaluated['difficulty'].value_counts())
            logger.info("\nVerdict sum distribution:")
            for i in range(args.k + 1):
                count = (df_evaluated['verdict_sum'] == i).sum()
                pct = count / len(df_evaluated) * 100
                logger.info(f"  {i}/{args.k} correct: {count} ({pct:.1f}%)")

    logger.info("\n✅ K-fold inference and evaluation complete!")


if __name__ == '__main__':
    main()