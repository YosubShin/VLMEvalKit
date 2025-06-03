#!/usr/bin/env python3
"""
WandB Logger for VLMEvalKit
Log VLMEvalKit evaluation results to Weights & Biases for experiment tracking.

Usage:
    # Log results from a specific run
    python scripts/wandb_logger.py --model GPT4o --dataset MMBench_DEV_EN --work-dir ./outputs

    # Log all existing results in work directory
    python scripts/wandb_logger.py --log-all --work-dir ./outputs

    # Run evaluation and log to WandB in one command
    python scripts/wandb_logger.py --run-and-log --model GPT4o --dataset MMBench_DEV_EN
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Add parent directory to path to import vlmeval modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    import wandb
except ImportError:
    print("ERROR: wandb is not installed. Please install it with: pip install wandb")
    sys.exit(1)

from vlmeval.smp import load, get_logger
from vlmeval.config import supported_VLM
from vlmeval.dataset import SUPPORTED_DATASETS


logger = get_logger('WandB Logger')


def extract_metrics_from_result_file(result_file: str) -> Dict[str, Any]:
    """Extract metrics from VLMEvalKit result files."""
    metrics = {}
    
    try:
        if result_file.endswith('.xlsx'):
            df = pd.read_excel(result_file)
        elif result_file.endswith('.csv'):
            df = pd.read_csv(result_file)
        elif result_file.endswith('.json'):
            with open(result_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                metrics['raw_results'] = data
                return metrics
        elif result_file.endswith('.pkl'):
            data = load(result_file)
            if isinstance(data, dict):
                return data
            else:
                metrics['raw_results'] = str(data)
                return metrics
        else:
            logger.warning(f"Unsupported file format: {result_file}")
            return metrics
            
        # Extract common metrics from DataFrame
        if 'prediction' in df.columns and 'answer' in df.columns:
            # Calculate accuracy if we have predictions and ground truth
            correct = (df['prediction'] == df['answer']).sum()
            total = len(df)
            metrics['accuracy'] = correct / total if total > 0 else 0.0
            metrics['correct_predictions'] = int(correct)
            metrics['total_samples'] = int(total)
            
        # Extract any numerical columns as metrics
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col not in ['index', 'line_id']:  # Skip identifier columns
                metrics[f'{col}_mean'] = float(df[col].mean())
                metrics[f'{col}_std'] = float(df[col].std())
                
        # Add dataset statistics
        metrics['dataset_size'] = len(df)
        
        # Extract unique categories if present
        if 'category' in df.columns:
            categories = df['category'].value_counts().to_dict()
            metrics['categories'] = {str(k): int(v) for k, v in categories.items()}
            
    except Exception as e:
        logger.error(f"Error extracting metrics from {result_file}: {e}")
        metrics['extraction_error'] = str(e)
        
    return metrics


def find_evaluation_files(work_dir: str, model_name: str, dataset_name: str) -> List[str]:
    """Find all evaluation result files for a model-dataset combination."""
    model_dir = Path(work_dir) / model_name
    if not model_dir.exists():
        return []
        
    result_files = []
    patterns = [
        f"{model_name}_{dataset_name}.xlsx",
        f"{model_name}_{dataset_name}.csv", 
        f"{model_name}_{dataset_name}.json",
        f"{model_name}_{dataset_name}_*.xlsx",
        f"{model_name}_{dataset_name}_*.csv",
        f"{model_name}_{dataset_name}_*.json",
    ]
    
    for pattern in patterns:
        result_files.extend(model_dir.glob(pattern))
        
    return [str(f) for f in result_files]


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Extract model configuration if available."""
    config = {}
    
    if model_name in supported_VLM:
        model_partial = supported_VLM[model_name]
        if hasattr(model_partial, 'keywords'):
            config.update(model_partial.keywords)
        if hasattr(model_partial, 'func'):
            config['model_class'] = model_partial.func.__name__
            
    return config


def log_to_wandb(
    model_name: str, 
    dataset_name: str, 
    result_files: List[str],
    project: str = "vlmeval-benchmark",
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> str:
    """Log evaluation results to WandB."""
    
    # Prepare WandB configuration
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "framework": "VLMEvalKit",
        "result_files": [os.path.basename(f) for f in result_files]
    }
    
    # Add model-specific configuration
    model_config = get_model_config(model_name)
    if model_config:
        config["model_config"] = model_config
        
    # Add dataset info if available
    if dataset_name in SUPPORTED_DATASETS:
        config["dataset_supported"] = True
    
    # Initialize WandB run
    run = wandb.init(
        project=project,
        name=f"{model_name}_{dataset_name}",
        config=config,
        tags=tags or [model_name, dataset_name],
        notes=notes
    )
    
    # Log metrics from each result file
    all_metrics = {}
    for result_file in result_files:
        file_metrics = extract_metrics_from_result_file(result_file)
        file_suffix = Path(result_file).stem.replace(f"{model_name}_{dataset_name}", "").lstrip("_")
        
        if file_suffix:
            # Prefix metrics with file suffix if multiple files
            prefixed_metrics = {f"{file_suffix}_{k}": v for k, v in file_metrics.items()}
        else:
            prefixed_metrics = file_metrics
            
        all_metrics.update(prefixed_metrics)
        
        # Upload result file as artifact
        artifact = wandb.Artifact(
            name=f"results_{os.path.basename(result_file)}", 
            type="evaluation_results"
        )
        artifact.add_file(result_file)
        run.log_artifact(artifact)
    
    # Log all metrics
    wandb.log(all_metrics)
    
    run_url = run.url
    wandb.finish()
    
    return run_url


def run_evaluation_and_log(
    model_name: str,
    dataset_name: str, 
    work_dir: str = "./outputs",
    project: str = "vlmeval-benchmark",
    additional_args: List[str] = None
) -> str:
    """Run VLMEvalKit evaluation and log results to WandB."""
    
    logger.info(f"Running evaluation for {model_name} on {dataset_name}")
    
    # Prepare run command
    cmd = [
        sys.executable, "run.py",
        "--model", model_name,
        "--data", dataset_name,
        "--work-dir", work_dir
    ]
    
    if additional_args:
        cmd.extend(additional_args)
        
    # Run evaluation with real-time output
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True, 
            bufsize=1,  # Line buffered
            universal_newlines=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Stream output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Print to console immediately
                print(output.strip())
                output_lines.append(output.strip())
        
        # Wait for process to complete
        return_code = process.poll()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output='\n'.join(output_lines))
            
        logger.info("Evaluation completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with return code {e.returncode}")
        if hasattr(e, 'output') and e.output:
            logger.error(f"Command output: {e.output}")
        raise
    
    # Find and log result files
    result_files = find_evaluation_files(work_dir, model_name, dataset_name)
    if not result_files:
        logger.warning(f"No result files found for {model_name}_{dataset_name}")
        return None
        
    logger.info(f"Found {len(result_files)} result files: {result_files}")
    
    # Log to WandB
    run_url = log_to_wandb(
        model_name=model_name,
        dataset_name=dataset_name, 
        result_files=result_files,
        project=project,
        notes=f"Automated run via wandb_logger.py"
    )
    
    logger.info(f"Results logged to WandB: {run_url}")
    return run_url


def log_all_existing_results(work_dir: str, project: str = "vlmeval-benchmark"):
    """Log all existing evaluation results in work directory to WandB."""
    
    work_path = Path(work_dir)
    if not work_path.exists():
        logger.error(f"Work directory does not exist: {work_dir}")
        return
        
    logged_count = 0
    
    for model_dir in work_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Find all result files in model directory
        result_files = list(model_dir.glob("*.xlsx")) + list(model_dir.glob("*.csv")) + list(model_dir.glob("*.json"))
        
        # Group by dataset
        datasets = set()
        for result_file in result_files:
            filename = result_file.stem
            if filename.startswith(f"{model_name}_"):
                dataset_part = filename[len(f"{model_name}_"):]
                # Remove suffix like _acc, _score etc
                dataset_name = dataset_part.split('_')[0]
                datasets.add(dataset_name)
                
        for dataset_name in datasets:
            dataset_files = find_evaluation_files(work_dir, model_name, dataset_name)
            if dataset_files:
                logger.info(f"Logging {model_name} x {dataset_name}")
                try:
                    run_url = log_to_wandb(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        result_files=dataset_files, 
                        project=project,
                        notes="Batch upload of existing results"
                    )
                    logger.info(f"Logged: {run_url}")
                    logged_count += 1
                except Exception as e:
                    logger.error(f"Failed to log {model_name} x {dataset_name}: {e}")
                    
    logger.info(f"Successfully logged {logged_count} evaluations to WandB")


def main():
    parser = argparse.ArgumentParser(description="Log VLMEvalKit results to WandB")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, help="Dataset name") 
    parser.add_argument("--work-dir", type=str, default="./outputs", help="Work directory")
    parser.add_argument("--project", type=str, default="vlmeval-benchmark", help="WandB project name")
    parser.add_argument("--run-and-log", action="store_true", help="Run evaluation and log results")
    parser.add_argument("--log-all", action="store_true", help="Log all existing results")
    parser.add_argument("--tags", type=str, nargs="+", help="Additional tags for WandB run")
    parser.add_argument("--notes", type=str, help="Notes for WandB run")
    
    # Pass through additional arguments to run.py
    parser.add_argument("--run-args", type=str, nargs=argparse.REMAINDER, help="Additional arguments for run.py")
    
    args = parser.parse_args()
    
    # Initialize WandB if not already done
    if not wandb.api.api_key:
        logger.info("WandB not configured. Please run 'wandb login' first.")
        return
        
    if args.log_all:
        log_all_existing_results(args.work_dir, args.project)
        
    elif args.run_and_log:
        if not args.model or not args.dataset:
            logger.error("--model and --dataset are required for --run-and-log")
            return
            
        run_evaluation_and_log(
            model_name=args.model,
            dataset_name=args.dataset,
            work_dir=args.work_dir,
            project=args.project,
            additional_args=args.run_args
        )
        
    elif args.model and args.dataset:
        # Log existing results for specific model/dataset
        result_files = find_evaluation_files(args.work_dir, args.model, args.dataset)
        if not result_files:
            logger.error(f"No result files found for {args.model} x {args.dataset}")
            return
            
        run_url = log_to_wandb(
            model_name=args.model,
            dataset_name=args.dataset,
            result_files=result_files,
            project=args.project,
            tags=args.tags,
            notes=args.notes
        )
        logger.info(f"Results logged to WandB: {run_url}")
        
    else:
        logger.error("Please specify either --log-all or provide --model and --dataset")
        parser.print_help()


if __name__ == "__main__":
    main()