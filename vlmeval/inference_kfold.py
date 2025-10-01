"""
K-fold inference module for VLMEvalKit.

This module provides functions for running inference multiple times per prompt
to assess question difficulty and model consistency.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
from typing import Dict, List, Any, Optional, Tuple

from vlmeval.smp import load, dump
from vlmeval.tools import logger


def generate_with_variation(model, prompt_struct, dataset_name,
                           temperature=0.7, top_p=0.9, seed=None):
    """
    Generate a single response with controlled variation.

    Args:
        model: The VLM model
        prompt_struct: The structured prompt
        dataset_name: Name of the dataset
        temperature: Temperature for sampling (0-1)
        top_p: Top-p for nucleus sampling (0-1)
        seed: Random seed for reproducibility

    Returns:
        str: Generated response
    """
    # Set seed if provided
    if seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Store original parameters
    original_params = {}
    if hasattr(model, 'temperature'):
        original_params['temperature'] = model.temperature
    if hasattr(model, 'top_p'):
        original_params['top_p'] = model.top_p
    if hasattr(model, 'do_sample'):
        original_params['do_sample'] = model.do_sample

    try:
        # Set variation parameters
        if hasattr(model, 'temperature'):
            model.temperature = temperature
        if hasattr(model, 'top_p'):
            model.top_p = top_p
        if hasattr(model, 'do_sample'):
            model.do_sample = True if temperature > 0 else False

        # For API models (GPT, Claude, etc.), pass parameters directly
        generate_kwargs = {}
        if hasattr(model, 'is_api') and model.is_api:
            generate_kwargs['temperature'] = temperature
            generate_kwargs['top_p'] = top_p

        # Generate response
        response = model.generate(
            message=prompt_struct,
            dataset=dataset_name,
            **generate_kwargs
        )

        return response

    finally:
        # Restore original parameters
        for param, value in original_params.items():
            setattr(model, param, value)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def infer_data_kfold(model, dataset, k=8, temperature=0.7, top_p=0.9,
                    seed_base=42, work_dir='./outputs', verbose=False,
                    resume=True) -> Dict[str, Any]:
    """
    Perform k-fold inference on a dataset.

    Args:
        model: VLM model instance
        dataset: Dataset instance
        k: Number of inferences per prompt
        temperature: Temperature for sampling variation
        top_p: Top-p for nucleus sampling
        seed_base: Base seed for reproducibility
        work_dir: Working directory for outputs
        verbose: Whether to print detailed progress
        resume: Whether to resume from previous results

    Returns:
        Dict containing k predictions per index with metadata
    """
    dataset_name = dataset.dataset_name
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else str(model)

    # Setup output file
    os.makedirs(work_dir, exist_ok=True)
    checkpoint_file = osp.join(work_dir, f'{model_name}_{dataset_name}_k{k}_checkpoint.pkl')

    # Load existing results if resuming
    results = {}
    if resume and osp.exists(checkpoint_file):
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        results = load(checkpoint_file)

    # Get dataset data
    data = dataset.data
    total_items = len(data)

    # Progress tracking
    completed_items = sum(1 for r in results.values()
                          if len(r.get('predictions', [])) == k)
    pbar = tqdm(total=total_items, initial=completed_items,
                desc=f'K-fold inference (k={k}, T={temperature})')

    for idx_num, row in data.iterrows():
        index = row['index']

        # Skip if already completed
        if index in results and len(results[index].get('predictions', [])) == k:
            continue

        # Initialize result structure
        if index not in results:
            results[index] = {
                'index': index,
                'question': row.get('question', ''),
                'answer': row.get('answer', ''),
                'predictions': [],
                'metadata': {
                    'temperatures': [],
                    'seeds': [],
                    'top_p_values': []
                }
            }

        # Build prompt once for all k iterations
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            prompt_struct = model.build_prompt(row, dataset=dataset_name)
        else:
            prompt_struct = dataset.build_prompt(row)

        # Generate k predictions
        existing_preds = len(results[index]['predictions'])
        for k_iter in range(existing_preds, k):
            seed = seed_base + k_iter

            # Generate with variation
            response = generate_with_variation(
                model=model,
                prompt_struct=prompt_struct,
                dataset_name=dataset_name,
                temperature=temperature,
                top_p=top_p,
                seed=seed
            )

            # Store result and metadata
            results[index]['predictions'].append(response)
            results[index]['metadata']['temperatures'].append(temperature)
            results[index]['metadata']['seeds'].append(seed)
            results[index]['metadata']['top_p_values'].append(top_p)

            if verbose and k_iter == 0:
                logger.info(f"Index {index} - First response: {response[:100]}...")

        pbar.update(1)

        # Save checkpoint periodically
        if pbar.n % 10 == 0:
            dump(results, checkpoint_file)

    pbar.close()

    # Final save
    dump(results, checkpoint_file)
    logger.info(f"K-fold inference complete. Saved to {checkpoint_file}")

    return results


def kfold_results_to_dataframe(results: Dict[str, Any], k: int) -> pd.DataFrame:
    """
    Convert k-fold results dictionary to DataFrame format.

    Args:
        results: Dictionary with k predictions per index
        k: Number of predictions expected

    Returns:
        DataFrame with separate columns for each prediction
    """
    rows = []

    for index, data in results.items():
        row = {
            'index': data.get('index', index),
            'question': data.get('question', ''),
            'answer': data.get('answer', '')
        }

        # Add prediction columns
        predictions = data.get('predictions', [])
        for i in range(k):
            if i < len(predictions):
                row[f'prediction_{i+1}'] = predictions[i]
            else:
                row[f'prediction_{i+1}'] = ''

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values('index').reset_index(drop=True)


def evaluate_kfold_predictions(dataset, df_predictions: pd.DataFrame, k: int,
                              judge_model='gpt-4o-mini', nproc=4,
                              work_dir='./outputs', verbose=False) -> pd.DataFrame:
    """
    Evaluate k-fold predictions using the dataset's judge.

    Args:
        dataset: Dataset instance with evaluate method
        df_predictions: DataFrame with k predictions per row
        k: Number of predictions
        judge_model: Judge model name
        nproc: Number of parallel processes for evaluation
        work_dir: Working directory
        verbose: Whether to print progress

    Returns:
        DataFrame with verdicts and verdict_sum
    """
    import tempfile
    from uuid import uuid4

    results = df_predictions.copy()
    dataset_name = dataset.dataset_name

    # Check if dataset supports evaluation
    if not hasattr(dataset, 'evaluate'):
        logger.warning(f"Dataset {dataset_name} does not support evaluation")
        # Add dummy verdicts
        for i in range(k):
            results[f'verdict_{i+1}'] = 0
        results['verdict_sum'] = 0
        return results

    # Evaluate each prediction column
    for i in range(k):
        pred_col = f'prediction_{i+1}'
        verdict_col = f'verdict_{i+1}'

        if verbose:
            logger.info(f"Evaluating prediction {i+1}/{k}")

        # Create temp dataframe for evaluation
        temp_df = df_predictions[['index', 'question', 'answer']].copy()
        temp_df['prediction'] = df_predictions[pred_col]

        # Save to temp file
        temp_file = osp.join(work_dir, f'temp_eval_{uuid4().hex[:8]}.xlsx')
        temp_df.to_excel(temp_file, index=False)

        try:
            # Evaluate using dataset's method
            judge_kwargs = {
                'model': judge_model,
                'nproc': nproc,
                'verbose': verbose
            }
            eval_result = dataset.evaluate(temp_file, **judge_kwargs)

            # Extract verdicts
            if isinstance(eval_result, pd.DataFrame):
                if 'verdict' in eval_result.columns:
                    # Align verdicts with original dataframe
                    verdict_map = dict(zip(eval_result['index'], eval_result['verdict']))
                    results[verdict_col] = results['index'].map(verdict_map).fillna(0)
                else:
                    # Try to extract from other columns (accuracy, score, etc.)
                    score_cols = [c for c in eval_result.columns
                                 if 'score' in c.lower() or 'acc' in c.lower()]
                    if score_cols:
                        results[verdict_col] = eval_result[score_cols[0]].values
                    else:
                        results[verdict_col] = 0
            else:
                results[verdict_col] = 0

        except Exception as e:
            logger.error(f"Error evaluating prediction {i+1}: {e}")
            results[verdict_col] = 0

        finally:
            # Clean up temp file
            if osp.exists(temp_file):
                os.remove(temp_file)

    # Calculate aggregate metrics
    verdict_cols = [f'verdict_{i+1}' for i in range(k)]
    results['verdict_sum'] = results[verdict_cols].sum(axis=1)
    results['verdict_mean'] = results['verdict_sum'] / k
    results['verdict_std'] = results[verdict_cols].std(axis=1)

    # Classify difficulty
    results['difficulty'] = pd.cut(
        results['verdict_mean'],
        bins=[-0.001, 0.25, 0.75, 1.001],
        labels=['hard', 'medium', 'easy']
    )

    # Add consistency metric (low std = high consistency)
    results['consistency'] = 1 - results['verdict_std']

    return results


def analyze_kfold_results(df_results: pd.DataFrame, k: int) -> Dict[str, Any]:
    """
    Analyze k-fold evaluation results.

    Args:
        df_results: DataFrame with k-fold evaluation results
        k: Number of predictions per question

    Returns:
        Dictionary with analysis metrics
    """
    analysis = {
        'total_questions': len(df_results),
        'k_value': k,
        'verdict_sum_distribution': {},
        'difficulty_distribution': {},
        'consistency_stats': {},
        'problematic_questions': []
    }

    # Verdict sum distribution
    for i in range(k + 1):
        count = (df_results['verdict_sum'] == i).sum()
        pct = count / len(df_results) * 100
        analysis['verdict_sum_distribution'][f'{i}/{k}'] = {
            'count': int(count),
            'percentage': round(pct, 2)
        }

    # Difficulty distribution
    if 'difficulty' in df_results.columns:
        diff_counts = df_results['difficulty'].value_counts()
        for diff_level in ['easy', 'medium', 'hard']:
            if diff_level in diff_counts.index:
                count = diff_counts[diff_level]
                pct = count / len(df_results) * 100
                analysis['difficulty_distribution'][diff_level] = {
                    'count': int(count),
                    'percentage': round(pct, 2)
                }

    # Consistency statistics
    if 'consistency' in df_results.columns:
        analysis['consistency_stats'] = {
            'mean': round(df_results['consistency'].mean(), 3),
            'std': round(df_results['consistency'].std(), 3),
            'min': round(df_results['consistency'].min(), 3),
            'max': round(df_results['consistency'].max(), 3)
        }

    # Identify problematic questions (high variance, extreme difficulty)
    if 'verdict_std' in df_results.columns:
        # Questions with high variance (inconsistent)
        high_variance = df_results[df_results['verdict_std'] > 0.4]
        for _, row in high_variance.iterrows():
            analysis['problematic_questions'].append({
                'index': row['index'],
                'reason': 'high_variance',
                'verdict_sum': int(row['verdict_sum']),
                'verdict_std': round(row['verdict_std'], 3)
            })

    return analysis


def print_kfold_analysis(analysis: Dict[str, Any]):
    """
    Print formatted analysis of k-fold results.

    Args:
        analysis: Dictionary from analyze_kfold_results
    """
    print("\n" + "="*60)
    print("K-FOLD ANALYSIS REPORT")
    print("="*60)

    print(f"\nTotal Questions: {analysis['total_questions']}")
    print(f"K Value: {analysis['k_value']}")

    print("\n📊 Verdict Sum Distribution:")
    for key, stats in analysis['verdict_sum_distribution'].items():
        bar_length = int(stats['percentage'] / 2)
        bar = '█' * bar_length
        print(f"  {key:>5} correct: {stats['count']:4d} ({stats['percentage']:5.1f}%) {bar}")

    if analysis['difficulty_distribution']:
        print("\n📈 Difficulty Distribution:")
        for level, stats in analysis['difficulty_distribution'].items():
            print(f"  {level:8s}: {stats['count']:4d} ({stats['percentage']:5.1f}%)")

    if analysis['consistency_stats']:
        print("\n🎯 Consistency Statistics:")
        for metric, value in analysis['consistency_stats'].items():
            print(f"  {metric:5s}: {value}")

    if analysis['problematic_questions']:
        print(f"\n⚠️  Problematic Questions: {len(analysis['problematic_questions'])}")
        for q in analysis['problematic_questions'][:5]:  # Show first 5
            print(f"  Index {q['index']}: {q['reason']} "
                  f"(sum={q['verdict_sum']}, std={q.get('verdict_std', 'N/A')})")

    print("\n" + "="*60)