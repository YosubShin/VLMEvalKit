"""
Answer extraction and scoring logic for LiveXivVQA benchmark.

This module contains the complete answer extraction and scoring pipeline for the 
LiveXivVQA benchmark, which evaluates visual question answering on academic 
paper figures with multiple choice questions.
"""

import pandas as pd
import numpy as np
import re
import string
import random
from collections import defaultdict


def build_livexiv_prompt(line, prefix=None):
    """
    Build the LiveXiv-specific prompt for VQA questions.
    
    Args:
        line: Dictionary containing question data
        prefix: Optional prefix string
        
    Returns:
        Formatted prompt string
    """
    question = line['question']
    options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
    
    for key, item in options.items():
        question += f'\n{key}: {item}'
    
    prompt = f"{question}\nAnswer with the option's letter from the given choices directly."
    
    if prefix:
        prompt = f"{prefix} {prompt}"
    
    return prompt


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
    
    # Check for explicit patterns like "Answer: A" or "The answer is B"
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


def build_gpt_prompt(question, options, prediction):
    """
    Build GPT prompt for answer matching when direct extraction fails.
    
    Args:
        question: The original question
        options: The formatted options string
        prediction: The model's prediction
        
    Returns:
        The GPT prompt string
    """
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


def extract_answer_from_item(model, item, dataset_name=None):
    """
    Extract answer from LiveXivVQA item using multiple strategies.
    
    Args:
        model: The judge model for GPT-based matching
        item: Dictionary containing question, options, and prediction
        dataset_name: Name of the dataset (should be 'LiveXivVQA')
        
    Returns:
        Dictionary with 'opt' (predicted option) and 'log' (extraction log)
    """
    choices = build_choices(item)
    option_str = build_option_str(choices)
    
    # Try direct inference first
    ret = can_infer_option(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=f'Direct inference: {item["prediction"]}')
    
    # If no model provided, return random choice
    if model is None:
        options = list(choices.keys()) + ['Z']
        return dict(opt=random.choice(options), log='No model provided, random choice')
    
    # Use GPT-based matching
    prompt = build_gpt_prompt(item['question'], option_str, item['prediction'])
    retry = 3
    
    while retry:
        try:
            ans = model.generate(prompt)
            if 'Failed to obtain answer via API' in ans:
                print('GPT API failed to answer.')
            else:
                ret = can_infer_option(ans, choices)
                if ret:
                    return dict(opt=ret, log=f'GPT-based matching: {ans}')
                else:
                    print(f'GPT output includes 0 / > 1 letter among candidates {set(choices)} and Z: {ans}')
        except Exception as e:
            print(f'Error in GPT-based matching: {e}')
        
        retry -= 1
    
    # Final fallback: random choice
    options = list(choices.keys()) + ['Z']
    return dict(opt=random.choice(options), log='Failed to predict, random choice')


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


def eval_livexiv_item(model, item, dataset_name=None):
    """
    Evaluate a single LiveXiv item.
    
    Args:
        model: The judge model for evaluation
        item: Dictionary containing the item data
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing evaluation results
    """
    res = extract_answer_from_item(model, item, dataset_name=dataset_name)
    opt, match_log = res['opt'], res['log']
    
    if opt == item['GT']:
        return dict(hit=1, log=f'Match Log: {match_log}')
    else:
        return dict(hit=0, log=f'Match Log: {match_log}')


def calculate_livexiv_accuracy(data):
    """
    Calculate accuracy for LiveXiv dataset.
    
    Args:
        data: DataFrame containing evaluation results
        
    Returns:
        Dictionary containing accuracy metrics
    """
    total = len(data)
    if total == 0:
        return {'accuracy': 0.0, 'total': 0, 'correct': 0}
    
    correct = sum(data['hit'])
    accuracy = correct / total
    
    results = {
        'accuracy': round(accuracy * 100, 2),
        'total': total,
        'correct': correct
    }
    
    # Add category-wise accuracy if categories are available
    if 'category' in data.columns:
        category_results = {}
        for category in data['category'].unique():
            cat_data = data[data['category'] == category]
            cat_total = len(cat_data)
            cat_correct = sum(cat_data['hit'])
            cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
            category_results[category] = {
                'accuracy': round(cat_accuracy * 100, 2),
                'total': cat_total,
                'correct': cat_correct
            }
        results['categories'] = category_results
    
    return results


def extract_answer_livexivvqa(prediction_text):
    """
    Extract answer from LiveXivVQA prediction text.
    
    Args:
        prediction_text: The model's prediction text
        
    Returns:
        Extracted answer option letter or None if no answer found
    """
    # This is a simplified version - in practice, you'd need the choices
    # to use the full extraction logic
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


def score_livexivvqa(predicted_answer, ground_truth_answer, choices=None):
    """
    Score LiveXivVQA prediction against ground truth.
    
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
        pred_content = choices[predicted_answer].lower()
        gt_content = choices[ground_truth_answer].lower()
        
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


def process_livexiv_batch(data, model=None):
    """
    Process a batch of LiveXiv items.
    
    Args:
        data: DataFrame containing the items to process
        model: Optional model for GPT-based matching
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for i in range(len(data)):
        item = data.iloc[i].to_dict()
        
        # Extract answer
        extraction_result = extract_answer_from_item(model, item, dataset_name='LiveXivVQA')
        
        # Score the result
        score = get_mc_score(data.iloc[i])
        
        results.append({
            'index': item.get('index', i),
            'predicted_option': extraction_result['opt'],
            'ground_truth': item.get('answer', item.get('GT', '')),
            'score': score,
            'extraction_log': extraction_result['log']
        })
    
    return pd.DataFrame(results)