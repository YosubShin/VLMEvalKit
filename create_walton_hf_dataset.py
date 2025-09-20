#!/usr/bin/env python3
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


def create_walton_dataset_from_tsv(tsv_path, push_to_hub=False, repo_name="yosubshin/WaltonMultimodalColdStart-hard"):
    """
    Create a HuggingFace dataset from TSV file with base64 images.

    Args:
        tsv_path: Path to TSV file with columns: index, image, question, answer
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        repo_name: Repository name on HuggingFace Hub
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

    # Extract problems and solutions from question and answer columns
    # The 'question' column contains the problem statement
    # The 'answer' column contains the solution
    problems = df_valid['question'].tolist()
    solutions = df_valid['answer'].tolist()

    # Define the dataset features (schema)
    features = Features({
        'image': Image(),  # HuggingFace will handle PIL images
        'problem': Value('string'),
        'solution': Value('string')
    })

    # Create the dataset
    print("Creating HuggingFace dataset...")
    dataset_dict = {
        'image': images,
        'problem': problems,
        'solution': solutions
    }

    dataset = Dataset.from_dict(dataset_dict, features=features)
    print(f"Created dataset with {len(dataset)} examples")

    # Create train/test split (90/10 by default)
    print("Creating train/test split...")
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

    # Rename 'test' to 'validation' if you prefer
    dataset_dict = DatasetDict({
        'train': dataset_split['train'],
        'test': dataset_split['test']
    })

    print(f"Train set: {len(dataset_dict['train'])} examples")
    print(f"Test set: {len(dataset_dict['test'])} examples")

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

    args = parser.parse_args()

    # Create the dataset
    dataset = create_walton_dataset_from_tsv(
        args.tsv_file,
        push_to_hub=args.push,
        repo_name=args.repo_name
    )

    # Print some examples
    print("\n" + "="*50)
    print("Sample from train set:")
    sample = dataset['train'][0]
    print(f"Problem (first 200 chars): {sample['problem'][:200]}...")
    print(f"Solution (first 200 chars): {sample['solution'][:200]}...")
    print(f"Image type: {type(sample['image'])}")
    print(f"Image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")


if __name__ == "__main__":
    main()