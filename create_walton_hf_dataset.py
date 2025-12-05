#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.19",
#   "huggingface_hub>=0.23",
#   "pandas>=2.1",
#   "Pillow>=10.3",
# ]
# ///
"""
Create HuggingFace dataset from TSV file for WaltonMultimodalColdStart-hard.
"""

from datasets import Dataset, DatasetDict, Features, Value, Image
import pandas as pd
from PIL import Image as PILImage
import base64
from io import BytesIO
import argparse
import os


def create_walton_dataset_from_tsv(tsv_path, push_to_hub=False, repo_name="yosubshin/WaltonMultimodalColdStart-hard",
                                  split_type="train_only",
                                  input_question_col="question", input_answer_col="answer",
                                  output_question_col="problem", output_answer_col="solution"):
    """
    Create a HuggingFace dataset from TSV file with base64 images.

    Args:
        tsv_path: Path to TSV file with columns: index, image, question, answer
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        repo_name: Repository name on HuggingFace Hub
        split_type: Type of split - "train_only", "train_test", or "train_val_test"
        input_question_col: Name of the question column in input TSV (default: "question")
        input_answer_col: Name of the answer column in input TSV (default: "answer")
        output_question_col: Name of the question column in output dataset (default: "problem")
        output_answer_col: Name of the answer column in output dataset (default: "solution")
    """
    print(f"Reading TSV from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(df)} rows from TSV")

    # Process images from base64 strings
    print("Processing images...")
    images = []
    valid_indices = []

    for idx, row in df.iterrows():
        base64_str = row['image']
        if pd.notna(base64_str) and isinstance(base64_str, str):
            try:
                # Decode base64 to PIL Image
                img_data = base64.b64decode(base64_str)
                img = PILImage.open(BytesIO(img_data))
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing image at index {row['index']}: {e}")
                # Skip rows with invalid images
                continue
        else:
            print(f"Skipping row {row['index']}: No image data")
            continue

    print(f"Successfully processed {len(images)} images")

    # Filter dataframe to only include valid rows
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # Check if input columns exist
    if input_question_col not in df_valid.columns:
        raise ValueError(f"Input question column '{input_question_col}' not found in TSV. Available columns: {list(df_valid.columns)}")
    if input_answer_col not in df_valid.columns:
        raise ValueError(f"Input answer column '{input_answer_col}' not found in TSV. Available columns: {list(df_valid.columns)}")

    # Extract questions and answers using configurable column names
    questions = df_valid[input_question_col].tolist()
    answers = df_valid[input_answer_col].tolist()

    # Define the dataset features (schema) with configurable output column names
    features = Features({
        'image': Image(),  # HuggingFace will handle PIL images
        output_question_col: Value('string'),
        output_answer_col: Value('string')
    })

    # Create the dataset
    print(f"Creating HuggingFace dataset with columns: image, {output_question_col}, {output_answer_col}")
    dataset_dict = {
        'image': images,
        output_question_col: questions,
        output_answer_col: answers
    }

    dataset = Dataset.from_dict(dataset_dict, features=features)
    print(f"Created dataset with {len(dataset)} examples")

    # Create splits based on split_type parameter
    if split_type == "train_only":
        # Just a single train split
        print("Creating train-only dataset...")
        dataset_dict = DatasetDict({
            'train': dataset
        })
        print(f"Train set: {len(dataset_dict['train'])} examples")

    elif split_type == "train_test":
        # Create train/test split (50/50 by default)
        print("Creating train/test split...")
        dataset_split = dataset.train_test_split(test_size=0.5, seed=42)
        dataset_dict = DatasetDict({
            'train': dataset_split['train'],
            'test': dataset_split['test']
        })
        print(f"Train set: {len(dataset_dict['train'])} examples")
        print(f"Test set: {len(dataset_dict['test'])} examples")

    elif split_type == "train_val_test":
        # Create train/validation/test split (80/10/10)
        print("Creating train/validation/test split...")
        train_val = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_val['test'].train_test_split(test_size=0.5, seed=42)
        dataset_dict = DatasetDict({
            'train': train_val['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })
        print(f"Train set: {len(dataset_dict['train'])} examples")
        print(f"Validation set: {len(dataset_dict['validation'])} examples")
        print(f"Test set: {len(dataset_dict['test'])} examples")

    else:
        raise ValueError(f"Invalid split_type: {split_type}. Must be 'train_only', 'train_test', or 'train_val_test'")

    # Save locally first
    local_save_path = f"./{repo_name.split('/')[-1]}_local"
    print(f"Saving dataset locally to {local_save_path}...")
    dataset_dict.save_to_disk(local_save_path)
    print("Dataset saved locally")

    # Push to HuggingFace Hub if requested
    if push_to_hub:
        print(f"Pushing dataset to HuggingFace Hub: {repo_name}")
        print("Make sure you're logged in with: huggingface-cli login")

        try:
            dataset_dict.push_to_hub(
                repo_name,
                private=False,  # Set to True if you want a private dataset
                commit_message="Initial upload of WaltonMultimodalColdStart-hard dataset"
            )
            print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")
        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Make sure you have logged in with 'huggingface-cli login' and have write access to the repository")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description='Create WaltonMultimodalColdStart-hard HuggingFace dataset from TSV'
    )
    parser.add_argument('tsv_file', help='Path to input TSV file')
    parser.add_argument('--push', action='store_true',
                        help='Push dataset to HuggingFace Hub')
    parser.add_argument('--repo-name', default='yosubshin/WaltonMultimodalColdStart-hard',
                        help='Repository name on HuggingFace Hub')
    parser.add_argument('--split-type', default='train_only',
                        choices=['train_only', 'train_test', 'train_val_test'],
                        help='Type of dataset split (default: train_only)')

    # Column name configuration
    parser.add_argument('--input-question-col', default='question',
                        help='Name of the question column in input TSV (default: question)')
    parser.add_argument('--input-answer-col', default='answer',
                        help='Name of the answer column in input TSV (default: answer)')
    parser.add_argument('--output-question-col', default='problem',
                        help='Name of the question column in output dataset (default: problem)')
    parser.add_argument('--output-answer-col', default='solution',
                        help='Name of the answer column in output dataset (default: solution)')

    args = parser.parse_args()

    # Create the dataset
    dataset = create_walton_dataset_from_tsv(
        args.tsv_file,
        push_to_hub=args.push,
        repo_name=args.repo_name,
        split_type=args.split_type,
        input_question_col=args.input_question_col,
        input_answer_col=args.input_answer_col,
        output_question_col=args.output_question_col,
        output_answer_col=args.output_answer_col
    )

    # Print some examples
    print("\n" + "="*50)
    print("Sample from train set:")
    sample = dataset['train'][0]

    # Use the configured output column names
    question_col = args.output_question_col
    answer_col = args.output_answer_col

    if question_col in sample:
        print(f"{question_col} (first 200 chars): {str(sample[question_col])[:200]}...")
    if answer_col in sample:
        print(f"{answer_col} (first 200 chars): {str(sample[answer_col])[:200]}...")
    print(f"Image type: {type(sample['image'])}")
    print(f"Image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")


if __name__ == "__main__":
    main()
