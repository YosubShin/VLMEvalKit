"""
Answer extraction and scoring logic for VMCBench benchmark (both DEV and TEST).

This module contains the complete answer extraction and scoring pipeline for the 
VMCBench benchmark, which evaluates vision-language models across multiple datasets
with multiple choice questions. The benchmark includes both VMCBench_DEV and VMCBench_TEST
which use the same evaluation methodology.
"""

import pandas as pd
import numpy as np
import random
import string
from collections import defaultdict


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    
    Args:
        response: The model's response string
        all_choices: List of valid choice letters (e.g., ['A', 'B', 'C', 'D'])
        index2ans: Dictionary mapping choice letters to answer text
        
    Returns:
        The predicted choice letter
    """
    response = str(response)
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response or f'{choice}. ' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def get_mc_score(row, use_parse=True):
    """
    Get multiple choice score for a single row.
    
    Args:
        row: Pandas Series containing the row data
        use_parse: Whether to use parsing (True) or direct output (False)
        
    Returns:
        Integer score (1 for correct, 0 for incorrect)
    """
    if use_parse:
        if pd.isna(row.get("A")):
            return 0
        
        response = row["prediction"]
        all_choices = []
        for i in range(9):  # Support up to 9 choices
            choice_letter = chr(65 + i)  # A, B, C, D, E, F, G, H, I
            if choice_letter in row and not pd.isna(row[choice_letter]):
                all_choices.append(choice_letter)
        
        index2ans = {index: row[index] for index in all_choices}
        pred_index = parse_multi_choice_response(response, all_choices, index2ans)
    else:
        pred_index = row["output"]
    
    return int(pred_index == row["answer"])


def report_vmc_acc(data):
    """
    Report VMC accuracy with category-wise breakdown.
    
    Args:
        data: DataFrame containing evaluation results with 'category' and 'hit' columns
        
    Returns:
        DataFrame with accuracy results by category and overall metrics
    """
    # Define dataset groupings
    general_datasets = ["SEEDBench", "MMStar", "A-OKVQA", "VizWiz", "MMVet", "VQAv2", "OKVQA"]
    reason_datasets = ["MMMU", "MathVista", "ScienceQA", "RealWorldQA", "GQA", "MathVision"]
    ocr_datasets = ["TextVQA", "OCRVQA"]
    doc_datasets = ["AI2D", "ChartQA", "DocVQA", "InfoVQA", "TableVQABench"]
    
    results = {}
    
    # Calculate accuracy for each category
    for category in data['category'].unique():
        results[category] = data[data['category'] == category]['hit'].mean()
    
    results = pd.DataFrame(results, index=[0])
    
    # Calculate overall and grouped metrics
    results["Overall"] = data['hit'].mean()
    
    # Calculate grouped accuracies
    available_general = [d for d in general_datasets if d in results.columns]
    available_reason = [d for d in reason_datasets if d in results.columns]
    available_ocr = [d for d in ocr_datasets if d in results.columns]
    available_doc = [d for d in doc_datasets if d in results.columns]
    
    if available_general:
        results['General'] = results[available_general].mean(axis=1)
    else:
        results['General'] = 0.0
        
    if available_reason:
        results['Reasoning'] = results[available_reason].mean(axis=1)
    else:
        results['Reasoning'] = 0.0
        
    if available_ocr:
        results['OCR'] = results[available_ocr].mean(axis=1)
    else:
        results['OCR'] = 0.0
        
    if available_doc:
        results['Doc & Chart'] = results[available_doc].mean(axis=1)
    else:
        results['Doc & Chart'] = 0.0
    
    # Convert to percentages
    for key in results:
        results[key] = round(results[key] * 100, 2)
    
    # Reorder columns
    summary_cols = ['Overall', 'General', 'Reasoning', 'OCR', 'Doc & Chart']
    detail_cols = available_general + available_reason + available_ocr + available_doc
    
    final_cols = []
    for col in summary_cols:
        if col in results.columns:
            final_cols.append(col)
    for col in detail_cols:
        if col in results.columns and col not in final_cols:
            final_cols.append(col)
    
    results = results[final_cols]
    return results


def build_choices(item):
    """
    Build dictionary of choices from item data.
    
    Args:
        item: Dictionary containing choice data (A, B, C, D, etc.)
        
    Returns:
        Dictionary mapping choice letters to choice text
    """
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def build_option_str(choices):
    """
    Build option string from choices dictionary.
    
    Args:
        choices: Dictionary mapping choice letters to choice text
        
    Returns:
        Formatted option string
    """
    option_str = ""
    for key, value in choices.items():
        option_str += f"{key}. {value}\n"
    return option_str.strip()


def can_infer_option(answer, choices):
    """
    Check if we can infer the option from the answer text.
    
    Args:
        answer: The answer text to check
        choices: Dictionary of available choices
        
    Returns:
        The inferred option letter or None
    """
    answer = str(answer).strip()
    
    # Check for explicit patterns
    import re
    patterns = [
        r'(?:answer|choice|option)(?:\s+is)?\s*:?\s*([A-Z])',
        r'(?:select|choose)(?:\s+option)?\s*([A-Z])',
        r'(?:correct|right)(?:\s+answer)?\s*(?:is)?\s*([A-Z])',
        r'([A-Z])\s*(?:is|would be)?\s*(?:the)?\s*(?:correct|right|answer)',
        r'option\s*([A-Z])',
        r'choice\s*([A-Z])',
        r'\b([A-Z])\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for match in matches:
            if match.upper() in choices:
                return match.upper()
    
    # Check for content-based matching
    answer_lower = answer.lower()
    for choice_key, choice_text in choices.items():
        if choice_text.lower() in answer_lower or answer_lower in choice_text.lower():
            return choice_key
    
    return None


def extract_answer_vmcbench(prediction_text, choices=None):
    """
    Extract answer from VMCBench prediction text.
    
    Args:
        prediction_text: The model's prediction text
        choices: Optional dictionary of choices for content-based matching
        
    Returns:
        Extracted answer option letter or None if no answer found
    """
    if choices:
        return can_infer_option(prediction_text, choices)
    
    # Fallback to pattern matching without choices
    import re
    patterns = [
        r'(?:answer|choice|option)(?:\s+is)?\s*:?\s*([A-Z])',
        r'(?:select|choose)(?:\s+option)?\s*([A-Z])',
        r'(?:correct|right)(?:\s+answer)?\s*(?:is)?\s*([A-Z])',
        r'([A-Z])\s*(?:is|would be)?\s*(?:the)?\s*(?:correct|right|answer)',
        r'option\s*([A-Z])',
        r'choice\s*([A-Z])',
        r'\b([A-Z])\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, prediction_text, re.IGNORECASE)
        for match in matches:
            if match.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                return match.upper()
    
    return None


def score_vmcbench(predicted_answer, ground_truth_answer, choices=None):
    """
    Score VMCBench prediction against ground truth.
    
    Args:
        predicted_answer: The predicted answer option letter
        ground_truth_answer: The ground truth answer option letter
        choices: Optional dictionary of choices for content-based matching
        
    Returns:
        Dictionary containing scoring results
    """
    if not predicted_answer or not ground_truth_answer:
        return {
            'correct': False,
            'score': 0.0,
            'explanation': 'Missing prediction or ground truth',
            'extracted_answer': predicted_answer
        }
    
    # Direct option letter match
    if predicted_answer.upper() == ground_truth_answer.upper():
        return {
            'correct': True,
            'score': 1.0,
            'explanation': 'Exact option match',
            'extracted_answer': predicted_answer
        }
    
    # Content-based matching if choices are provided
    if choices and predicted_answer in choices and ground_truth_answer in choices:
        pred_content = choices[predicted_answer].lower().strip()
        gt_content = choices[ground_truth_answer].lower().strip()
        
        # Simple content similarity check
        if pred_content == gt_content:
            return {
                'correct': True,
                'score': 1.0,
                'explanation': 'Content match despite different option letters',
                'extracted_answer': predicted_answer
            }
    
    return {
        'correct': False,
        'score': 0.0,
        'explanation': 'No match found',
        'extracted_answer': predicted_answer
    }


def eval_vmc_item(item, use_parse=True):
    """
    Evaluate a single VMC item.
    
    Args:
        item: Dictionary or Series containing the item data
        use_parse: Whether to use parsing or direct output
        
    Returns:
        Dictionary containing evaluation results
    """
    if isinstance(item, pd.Series):
        score = get_mc_score(item, use_parse=use_parse)
    else:
        # Convert dict to Series-like object
        item_series = pd.Series(item)
        score = get_mc_score(item_series, use_parse=use_parse)
    
    return {
        'hit': score,
        'log': f'Prediction: {item.get("prediction", "")}, Answer: {item.get("answer", "")}'
    }


def process_vmc_batch(data, use_parse=True):
    """
    Process a batch of VMC items.
    
    Args:
        data: DataFrame containing the items to process
        use_parse: Whether to use parsing or direct output
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for i in range(len(data)):
        item = data.iloc[i]
        
        # Extract answer
        choices = build_choices(item)
        extracted_answer = extract_answer_vmcbench(item['prediction'], choices)
        
        # Score the result
        score = get_mc_score(item, use_parse=use_parse)
        
        results.append({
            'index': item.get('index', i),
            'category': item.get('category', 'Unknown'),
            'extracted_answer': extracted_answer,
            'ground_truth': item.get('answer', ''),
            'hit': score,
            'prediction': item.get('prediction', '')
        })
    
    return pd.DataFrame(results)


def calculate_vmc_accuracy(data):
    """
    Calculate accuracy for VMC dataset with detailed breakdowns.
    
    Args:
        data: DataFrame containing evaluation results
        
    Returns:
        Dictionary containing accuracy metrics
    """
    if len(data) == 0:
        return {
            'overall_accuracy': 0.0,
            'total_items': 0,
            'correct_items': 0,
            'by_category': {}
        }
    
    total_items = len(data)
    correct_items = data['hit'].sum()
    overall_accuracy = correct_items / total_items
    
    # Category-wise accuracy
    by_category = {}
    if 'category' in data.columns:
        for category in data['category'].unique():
            cat_data = data[data['category'] == category]
            cat_total = len(cat_data)
            cat_correct = cat_data['hit'].sum()
            cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0.0
            
            by_category[category] = {
                'accuracy': round(cat_accuracy * 100, 2),
                'total': cat_total,
                'correct': cat_correct
            }
    
    return {
        'overall_accuracy': round(overall_accuracy * 100, 2),
        'total_items': total_items,
        'correct_items': correct_items,
        'by_category': by_category
    }


def build_vmc_prompt(question, choices):
    """
    Build VMC-specific prompt for questions.
    
    Args:
        question: The question text
        choices: Dictionary of choices
        
    Returns:
        Formatted prompt string
    """
    prompt = question + "\n"
    for key, value in choices.items():
        prompt += f"{key}. {value}\n"
    prompt += "Answer with the option's letter from the given choices directly."
    return prompt


# Aliases for VMCBench_DEV and VMCBench_TEST (they use the same logic)
def extract_answer_vmcbench_dev(prediction_text, choices=None):
    """Extract answer for VMCBench_DEV (alias for extract_answer_vmcbench)."""
    return extract_answer_vmcbench(prediction_text, choices)


def extract_answer_vmcbench_test(prediction_text, choices=None):
    """Extract answer for VMCBench_TEST (alias for extract_answer_vmcbench)."""
    return extract_answer_vmcbench(prediction_text, choices)


def score_vmcbench_dev(predicted_answer, ground_truth_answer, choices=None):
    """Score VMCBench_DEV (alias for score_vmcbench)."""
    return score_vmcbench(predicted_answer, ground_truth_answer, choices)


def score_vmcbench_test(predicted_answer, ground_truth_answer, choices=None):
    """Score VMCBench_TEST (alias for score_vmcbench)."""
    return score_vmcbench(predicted_answer, ground_truth_answer, choices)