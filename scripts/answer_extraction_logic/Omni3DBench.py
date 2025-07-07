"""
Answer extraction and scoring logic for Omni3DBench benchmark.

This module contains the complete answer extraction and scoring pipeline for the 
Omni3DBench benchmark, which evaluates 3D spatial reasoning with different answer types
including yes/no questions, counting, multiple choice, and numerical answers with
Mean Relative Accuracy (MRA) scoring.
"""

from collections import defaultdict
import pandas as pd

# Standard prompt for Omni3DBench
OMNI3DBENCH_PROMPT = """
I will ask you a question based on an image. Answer with either true/false, one word or number, place your answer between <ans></ans> tags. Only include your answer. Question: 
"""


def extract_answer(prediction):
    """
    Extract answer from Omni3DBench prediction text using <ans></ans> tags.
    
    Args:
        prediction: The model's prediction text
        
    Returns:
        Extracted answer string or original prediction if no tags found
    """
    if '<ans>' in prediction:
        return prediction.split('<ans>')[1].split('</ans>')[0]
    else:
        return prediction


def Omni3DBench_acc(data):
    """
    Calculate accuracy metrics for Omni3DBench with different answer types.
    
    This function implements the scoring methodology for Omni3DBench which includes:
    - Yes/No questions: Binary classification accuracy
    - Multiple choice: Exact string matching
    - Numeric (count): Exact integer matching
    - Numeric (other): Mean Relative Accuracy (MRA) with multiple thresholds
    
    Args:
        data: DataFrame containing columns:
            - answer_type: 'str', 'int', or 'float'
            - answer: Ground truth answer
            - prediction: Model prediction
            
    Returns:
        DataFrame with accuracy results for each answer type
    """
    # MRA thresholds for numerical evaluation (from VSI-Bench paper)
    mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    correct_at_threshold = {key: 0 for key in mra_thresholds}
    
    # Initialize counters for different answer types
    yn_correct = 0
    yn_n = 0
    num_ct_n = 0
    num_ct_correct = 0
    multi_correct = 0
    multi_n = 0
    num_other_n = 0

    for i in range(len(data)):
        row = data.iloc[i]
        ans_type = row['answer_type']
        gt = row['answer']
        pred = extract_answer(row['prediction'])

        # Numeric (count) - integer answers
        if ans_type == "int":
            num_ct_n += 1
            try:
                pred = int(pred)
            except:
                continue
            gt = int(gt)
            if gt == pred:
                num_ct_correct += 1
                
        elif ans_type == "str":
            # Yes/No questions
            if gt in ["yes", "no"]:
                yn_n += 1
                try:
                    if gt in pred.lower():
                        yn_correct += 1
                    elif gt == "yes" and "true" in pred.lower():
                        yn_correct += 1
                    elif gt == "no" and "false" in pred.lower():
                        yn_correct += 1
                except:
                    continue
            # Multi-choice questions
            else:
                multi_n += 1
                try:
                    if gt == pred.lower():
                        multi_correct += 1
                except:
                    continue
                    
        elif ans_type == "float":
            # Numeric (other) - calculated with Mean Relative Accuracy (MRA)
            # This follows the methodology from VSI-Bench (https://arxiv.org/abs/2412.14171)
            num_other_n += 1
            for threshold in mra_thresholds:
                try:
                    pred = float(pred)
                except:
                    continue
                gt = float(gt)
                if abs(gt - pred) / gt < threshold:
                    correct_at_threshold[threshold] += 1

    # Compute accuracy for each category
    yn_acc = yn_correct / yn_n if yn_n != 0 else None
    multi_acc = multi_correct / multi_n if multi_n != 0 else None
    num_ct_acc = num_ct_correct / num_ct_n if num_ct_n != 0 else None
    num_other_mra = 0

    # Calculate Mean Relative Accuracy for numerical answers
    if num_other_n != 0:
        for threshold in mra_thresholds:
            correct_at_threshold[threshold] /= num_other_n
            num_other_mra += correct_at_threshold[threshold]

        num_other_mra = num_other_mra / len(mra_thresholds)
    else:
        num_other_mra = None

    # Build results DataFrame
    res = defaultdict(list)
    res['Yes/No Accuracy'].append(yn_acc)
    res['Multiple Choice Accuracy'].append(multi_acc)
    res['Numeric (count) Accuracy'].append(num_ct_acc)
    res['Numeric (other) Mean Relative Accuracy'].append(num_other_mra)
    res = pd.DataFrame(res)
    return res


def extract_answer_omni3dbench(prediction_text):
    """
    Extract answer from Omni3DBench prediction text.
    
    Args:
        prediction_text: The model's prediction text
        
    Returns:
        Extracted answer string or None if no answer found
    """
    extracted = extract_answer(prediction_text)
    if extracted == prediction_text:
        # No <ans></ans> tags found, try to extract from common patterns
        prediction_lower = prediction_text.lower().strip()
        
        # Try to find yes/no answers
        if 'yes' in prediction_lower and 'no' not in prediction_lower:
            return 'yes'
        elif 'no' in prediction_lower and 'yes' not in prediction_lower:
            return 'no'
        elif 'true' in prediction_lower and 'false' not in prediction_lower:
            return 'yes'
        elif 'false' in prediction_lower and 'true' not in prediction_lower:
            return 'no'
        
        # Try to extract numbers
        import re
        numbers = re.findall(r'-?\d+\.?\d*', prediction_text)
        if numbers:
            return numbers[0]
        
        # Return the cleaned text
        return prediction_text.strip()
    
    return extracted


def score_omni3dbench(predicted_answer, ground_truth_answer, answer_type):
    """
    Score Omni3DBench prediction against ground truth based on answer type.
    
    Args:
        predicted_answer: The predicted answer string
        ground_truth_answer: The ground truth answer
        answer_type: Type of answer ('str', 'int', 'float')
        
    Returns:
        Dictionary containing scoring results
    """
    if not predicted_answer or not ground_truth_answer:
        return {
            'correct': False,
            'score': 0.0,
            'explanation': 'Missing prediction or ground truth',
            'extracted_answer': predicted_answer,
            'answer_type': answer_type
        }
    
    try:
        if answer_type == "int":
            # Integer counting questions
            pred_int = int(predicted_answer)
            gt_int = int(ground_truth_answer)
            is_correct = pred_int == gt_int
            
            return {
                'correct': is_correct,
                'score': 1.0 if is_correct else 0.0,
                'explanation': 'Exact integer match',
                'extracted_answer': predicted_answer,
                'answer_type': answer_type
            }
            
        elif answer_type == "str":
            # String-based questions (yes/no or multiple choice)
            pred_lower = predicted_answer.lower().strip()
            gt_lower = str(ground_truth_answer).lower().strip()
            
            # Handle yes/no questions with variations
            if gt_lower in ["yes", "no"]:
                is_correct = False
                if gt_lower in pred_lower:
                    is_correct = True
                elif gt_lower == "yes" and "true" in pred_lower:
                    is_correct = True
                elif gt_lower == "no" and "false" in pred_lower:
                    is_correct = True
                
                return {
                    'correct': is_correct,
                    'score': 1.0 if is_correct else 0.0,
                    'explanation': 'Yes/No classification',
                    'extracted_answer': predicted_answer,
                    'answer_type': answer_type
                }
            else:
                # Multiple choice questions
                is_correct = gt_lower == pred_lower
                
                return {
                    'correct': is_correct,
                    'score': 1.0 if is_correct else 0.0,
                    'explanation': 'Exact string match',
                    'extracted_answer': predicted_answer,
                    'answer_type': answer_type
                }
                
        elif answer_type == "float":
            # Numerical questions with Mean Relative Accuracy
            pred_float = float(predicted_answer)
            gt_float = float(ground_truth_answer)
            
            # Calculate MRA scores at different thresholds
            mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
            threshold_scores = {}
            
            for threshold in mra_thresholds:
                if abs(gt_float - pred_float) / gt_float < threshold:
                    threshold_scores[threshold] = 1.0
                else:
                    threshold_scores[threshold] = 0.0
            
            # Calculate overall MRA score
            mra_score = sum(threshold_scores.values()) / len(mra_thresholds)
            
            return {
                'correct': mra_score > 0,
                'score': mra_score,
                'explanation': f'Mean Relative Accuracy: {mra_score:.3f}',
                'extracted_answer': predicted_answer,
                'answer_type': answer_type,
                'threshold_scores': threshold_scores
            }
            
    except (ValueError, TypeError) as e:
        return {
            'correct': False,
            'score': 0.0,
            'explanation': f'Type conversion error: {e}',
            'extracted_answer': predicted_answer,
            'answer_type': answer_type
        }
    
    # Fallback case
    return {
        'correct': False,
        'score': 0.0,
        'explanation': 'Unknown scoring method',
        'extracted_answer': predicted_answer,
        'answer_type': answer_type
    }


def build_omni3d_prompt(question):
    """
    Build the Omni3D-specific prompt for questions.
    
    Args:
        question: The question text
        
    Returns:
        Formatted prompt string
    """
    return OMNI3DBENCH_PROMPT + question


def process_omni3d_batch(data):
    """
    Process a batch of Omni3D items.
    
    Args:
        data: DataFrame containing the items to process
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for i in range(len(data)):
        item = data.iloc[i]
        
        # Extract answer
        extracted_answer = extract_answer_omni3dbench(item['prediction'])
        
        # Score the result
        score_result = score_omni3dbench(
            extracted_answer, 
            item['answer'], 
            item['answer_type']
        )
        
        results.append({
            'index': item.get('index', i),
            'extracted_answer': extracted_answer,
            'ground_truth': item['answer'],
            'answer_type': item['answer_type'],
            'correct': score_result['correct'],
            'score': score_result['score'],
            'explanation': score_result['explanation']
        })
    
    return pd.DataFrame(results)


def calculate_omni3d_summary_metrics(results_df):
    """
    Calculate summary metrics for Omni3D evaluation.
    
    Args:
        results_df: DataFrame with individual evaluation results
        
    Returns:
        Dictionary with summary metrics
    """
    if len(results_df) == 0:
        return {
            'overall_accuracy': 0.0,
            'total_items': 0,
            'by_answer_type': {}
        }
    
    # Overall metrics
    total_items = len(results_df)
    overall_correct = results_df['correct'].sum()
    overall_accuracy = overall_correct / total_items
    
    # Metrics by answer type
    by_answer_type = {}
    for answer_type in results_df['answer_type'].unique():
        type_data = results_df[results_df['answer_type'] == answer_type]
        type_total = len(type_data)
        type_correct = type_data['correct'].sum()
        type_accuracy = type_correct / type_total if type_total > 0 else 0.0
        
        by_answer_type[answer_type] = {
            'accuracy': round(type_accuracy * 100, 2),
            'total': type_total,
            'correct': type_correct
        }
    
    return {
        'overall_accuracy': round(overall_accuracy * 100, 2),
        'total_items': total_items,
        'correct_items': overall_correct,
        'by_answer_type': by_answer_type
    }