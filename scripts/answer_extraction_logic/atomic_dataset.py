"""
Answer extraction and scoring logic for atomic_dataset benchmark.

This module contains the complete answer extraction and scoring pipeline for the 
atomic_dataset benchmark, which is part of the Physics_yale dataset collection.
The benchmark evaluates physics problems with LaTeX boxed answer format.
"""

import logging
import re
import timeout_decorator
from sympy import simplify, expand, trigsimp
from sympy.parsing.latex import parse_latex
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

FAIL_MSG = 'Failed to obtain answer via API.'

# Judge system and user prompts for LLM-based equivalence checking
Judge_SYS_PROMPT = "You are an assistant that compares LaTeX expressions for equivalence."

Judge_USER_PROMPT = "Compare the following LaTeX expressions and check if the numerical parts are equivalent in meaning.\n\nExpression 1:\n{expr1}\n\nExpression 2:\n{expr2}\n\nReturn True if they are equivalent, otherwise return False. Focus on mathematical content."


def build_physic_prompt(line):
    """
    Build the physics expert prompt for atomic_dataset questions.
    
    Args:
        line: Dictionary containing question data with 'question' key
        
    Returns:
        List of message dictionaries for the model
    """
    prompt_text = (
        "You are a physics expert assistant. Solve the following question step-by-step.\n\n"
        "At the VERY END of your answer, output ONLY the FINAL ANSWER in this format:\n\n"
        "\\[\n\\boxed{{your_final_answer_here}}\n\\]\n\n"
        "You MUST put the final answer in the `\\boxed{}` environment.\n"
        "This applies even if the answer is a text explanation like \"The singlet state is lower in energy.\"\n"
        "Do NOT include multiple boxes.\n"
        "Do NOT include \\boxed anywhere else in your reasoning.\n"
        "The box must appear on the last line of the response.\n\n"
        "WARNING: DO NOT forget to include \boxed{} with the final answer. Responses without it will be considered INVALID.\n\n"
        "Example:\n\n"
        "Question: What is the energy difference between n=2 and n=1 in hydrogen?\n"
        "Answer:\nThe energy levels are E_n = -13.6 / n² (in eV).\n"
        "E_2 = -13.6 / 4 = -3.4 eV\n"
        "E_1 = -13.6 eV\n"
        "ΔE = 13.6 - 3.4 = 10.2 eV\n"
        "\\[\n\\boxed{10.2\\ \\text{eV}}\n\\]\n\n"
        "Question: Which energy state is lower in hydrogen molecule?\n"
        "Answer:\nBased on spin multiplicity, the singlet state lies lower in energy than the triplet.\n"
        "\\[\n\\boxed{The singlet state is lower in energy}\n\\]\n\n"
        f"Question: {line['question']}\nAnswer:"
    )
    return [{"type": "text", "value": prompt_text}]


def extract_all_boxed_content(latex_response, latex_wrap=r'\\boxed{([^{}]*|{.*?})}'):
    """
    Extract all boxed content from LaTeX response.
    
    Args:
        latex_response: The model's response containing LaTeX
        latex_wrap: Regex pattern for boxed content
        
    Returns:
        List of extracted boxed content strings
    """
    pattern = re.compile(
        r'\\boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}'
        r'|\\\\\[boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}\\\\\]',
        re.DOTALL
    )
    matches = pattern.findall(latex_response)
    if not matches:
        return []
    return [match.strip() for sublist in matches for match in sublist if match.strip()]


def extract_final_answer(latex_response):
    """
    Extract the final answer from LaTeX boxed format.
    
    Args:
        latex_response: The model's response containing LaTeX
        
    Returns:
        Extracted answer string or original response if no boxed content found
    """
    match = re.search(r'\\boxed{(.*?)}|\\\\\[boxed{(.*?)}\\\\\]', latex_response)
    if match:
        return next(group for group in match.groups() if group).strip()
    return latex_response


def extract_final_answer_list(last_answer):
    """
    Extract final answer that may contain a list of items.
    
    Args:
        last_answer: The answer string to extract from
        
    Returns:
        List of extracted answer items
    """
    matches = re.findall(r'\\boxed{\\\[(.*?)\\\]}|\\\\\[boxed{\\\[(.*?)\\\]}\\\\\]', last_answer)
    if matches:
        return [item.strip() for sublist in matches for item in sublist if item for item in item.split(',')]
    return [extract_final_answer(last_answer)]


def extract_final_answer_allform(latex_response, answer_type=None, latex_wrap=r'\\boxed{(.*?)}'):
    """
    Extract final answer in all supported formats.
    
    Args:
        latex_response: The model's response containing LaTeX
        answer_type: Type of answer expected ('list' or None)
        latex_wrap: Regex pattern for boxed content
        
    Returns:
        List of extracted answers
    """
    boxed_content = extract_all_boxed_content(latex_response, latex_wrap)
    if not boxed_content:
        return []

    if answer_type == 'list':
        return [extract_final_answer_list(item) for item in boxed_content]
    return [extract_final_answer(item) for item in boxed_content]


def _extract_core_eq(expr: str) -> str:
    """
    Extract the core equation from a LaTeX expression.
    
    Args:
        expr: LaTeX expression string
        
    Returns:
        Core equation string
    """
    if "\\implies" in expr:
        expr = expr.split("\\implies")[-1].strip()
    if "=" in expr:
        expr = expr.split("=")[-1].strip()
    return expr


def _preprocess_latex(string: str) -> str:
    """
    Preprocess LaTeX string for equivalence checking.
    
    Args:
        string: LaTeX string to preprocess
        
    Returns:
        Preprocessed string
    """
    if not string:
        return ""
    string = re.sub(r"_\{.*?\}", "", string)
    string = re.sub(r"_\\?\w", "", string)
    string = string.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*")
    return string


@timeout_decorator.timeout(10, use_signals=False)
def _standardize_expr(expr):
    """
    Standardize mathematical expression using SymPy.
    
    Args:
        expr: Mathematical expression to standardize
        
    Returns:
        Standardized expression
    """
    return simplify(expand(trigsimp(expr)))


def is_equiv(model, expr1: str, expr2: str, verbose: bool = False, capture_judge_responses: bool = False) -> dict:
    """
    Check if two mathematical expressions are equivalent.
    
    Args:
        model: The judge model for LLM-based comparison
        expr1: First expression to compare
        expr2: Second expression to compare
        verbose: Whether to print verbose output
        capture_judge_responses: Whether to capture judge responses
        
    Returns:
        Dictionary containing equivalence check results
    """
    result_data = {
        "input_expressions": {"expr1": expr1, "expr2": expr2},
        "preprocessed_expressions": {},
        "sympy_result": None,
        "llm_result": None,
        "final_result": None,
        "error": None,
        "llm_comparison_result": None,
    }
    try:
        if "\text" in expr1 or "\text" in expr2:
            model.sys_prompt = Judge_SYS_PROMPT
            user_prompt = Judge_USER_PROMPT.format(expr1=expr1, expr2=expr2)
            generate_result = model.generate(user_prompt)
            
            # Capture judge response if enabled
            if capture_judge_responses:
                result_data["judge_response_1"] = {
                    "prompt": user_prompt,
                    "response": generate_result,
                    "timestamp": __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
                    "reason": "text_based_comparison"
                }
            
            if generate_result and "true" in generate_result.lower():
                result_data["llm_result"] = 1
            else:
                result_data["llm_result"] = 0
            result_data["final_result"] = result_data["llm_result"]
            return result_data

        expr1_processed = _preprocess_latex(expr1)
        expr2_processed = _preprocess_latex(expr2)
        expr1_core = _extract_core_eq(expr1_processed)
        expr2_core = _extract_core_eq(expr2_processed)

        try:
            expr1_sympy = _standardize_expr(parse_latex(expr1_core))
            expr2_sympy = _standardize_expr(parse_latex(expr2_core))
            result_data["preprocessed_expressions"] = {
                "expr1": str(expr1_sympy),
                "expr2": str(expr2_sympy)
            }

            sympy_result = simplify(expr1_sympy - expr2_sympy) == 0 or expr1_sympy.equals(expr2_sympy)
        except Exception as e:
            result_data["error"] = str(e)
            sympy_result = None

        result_data["sympy_result"] = sympy_result

        if sympy_result:
            result_data["final_result"] = True
        else:
            model.sys_prompt = Judge_SYS_PROMPT
            user_prompt = Judge_USER_PROMPT.format(expr1=expr1, expr2=expr2)
            generate_result = model.generate(user_prompt)
            
            # Capture judge response if enabled
            if capture_judge_responses:
                result_data["judge_response_2"] = {
                    "prompt": user_prompt,
                    "response": generate_result,
                    "timestamp": __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
                    "reason": "sympy_failed_fallback_to_llm",
                    "sympy_result": sympy_result,
                    "sympy_error": result_data.get("error")
                }
            
            if generate_result and "true" in generate_result.lower():
                result_data["llm_result"] = 1
            else:
                result_data["llm_result"] = 0
            result_data["final_result"] = result_data["llm_result"]

    except Exception as e:
        result_data["error"] = str(e)

    return result_data


def PHYSIC_auxeval(model, line, capture_judge_responses=False):
    """
    Auxiliary evaluation function for physics problems.
    
    Args:
        model: The judge model for equivalence checking
        line: Dictionary containing prediction and ground truth answer
        capture_judge_responses: Whether to capture judge responses
        
    Returns:
        Dictionary containing evaluation results
    """
    equiv_data = {}
    try:
        response = line['prediction']
        if not response or not isinstance(response, str):
            equiv_data['LOG'] = 'Invalid response format, returning False.'
            return dict(log=equiv_data, res=False)

        pred_boxed = extract_final_answer_allform(response)
        gt = line['answer'].strip()

        flat_preds = [item.strip() for group in pred_boxed for item in (group if isinstance(group, list) else [group])]

        if gt in flat_preds:
            equiv_data['LOG'] = 'GT found in prediction, returning True.'
            return dict(log=equiv_data, res=True)

        for pred in flat_preds:
            equiv_data = is_equiv(model, pred, gt, capture_judge_responses=capture_judge_responses)
            if equiv_data['llm_result']:
                equiv_data['LOG'] = 'Equivalence found, returning True.'
                return dict(log=equiv_data, res=True)

        equiv_data['LOG'] = 'No equivalence found, returning False.'
        return dict(log=equiv_data, res=False)
    except Exception as e:
        logging.warning(f'post_check error: {e}')
        equiv_data['LOG'] = f'Exception occurred: {e}'
        return dict(log=equiv_data, res=False)


def PHYSIC_acc(result_file):
    """
    Calculate accuracy for physics problems.
    
    Args:
        result_file: Path to result file or DataFrame
        
    Returns:
        DataFrame with accuracy results by category
    """
    # This would need to be adapted to load the data properly
    # For now, assuming data is already loaded
    if isinstance(result_file, str):
        # Load from file path
        data = pd.read_csv(result_file)  # Adjust based on actual file format
    else:
        data = result_file
    
    tot = defaultdict(int)
    hit = defaultdict(int)
    lt = len(data)

    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item.get('category', 'Overall')

        tot['Overall'] += 1
        tot[cate] += 1

        if item.get('res'):
            hit['Overall'] += 1
            hit[cate] += 1

        pred_raw = item.get("res", "")
        gt = item.get("answer", "").strip()
        pred_boxed = extract_final_answer_allform(str(pred_raw))
        flat_pred = [ans.strip() for group in pred_boxed for ans in (group if isinstance(group, list) else [group])]

    res = defaultdict(list)
    for k in tot:
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] else 0.0)

    return pd.DataFrame(res).sort_values('Subject', ignore_index=True)


def extract_answer_atomic_dataset(prediction_text):
    """
    Extract answer from atomic_dataset prediction text.
    
    Args:
        prediction_text: The model's prediction text
        
    Returns:
        Extracted answer string or None if no answer found
    """
    extracted_answers = extract_final_answer_allform(prediction_text)
    if extracted_answers:
        return extracted_answers[0]  # Return the first extracted answer
    return None


def score_atomic_dataset(model, predicted_answer, ground_truth_answer, capture_judge_responses=False):
    """
    Score atomic_dataset prediction against ground truth.
    
    Args:
        model: The judge model for equivalence checking
        predicted_answer: The predicted answer string
        ground_truth_answer: The ground truth answer string
        capture_judge_responses: Whether to capture judge responses
        
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
    
    # Direct string match
    if predicted_answer.strip() == ground_truth_answer.strip():
        return {
            'correct': True,
            'score': 1.0,
            'explanation': 'Exact string match',
            'extracted_answer': predicted_answer
        }
    
    # Equivalence check using LLM
    equiv_result = is_equiv(model, predicted_answer, ground_truth_answer, capture_judge_responses=capture_judge_responses)
    
    is_correct = bool(equiv_result.get('final_result', False))
    
    return {
        'correct': is_correct,
        'score': 1.0 if is_correct else 0.0,
        'explanation': 'LLM-based equivalence check' if is_correct else 'No equivalence found',
        'extracted_answer': predicted_answer,
        'equiv_details': equiv_result
    }