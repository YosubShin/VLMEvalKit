"""
Answer extraction and scoring logic for VMCBench_DEV benchmark.

This module provides the specific interface for VMCBench_DEV evaluation,
importing the shared logic from VMCBench.py since both DEV and TEST
versions use identical evaluation methodologies.
"""

from .VMCBench import (
    extract_answer_vmcbench as extract_answer_vmcbench_dev,
    score_vmcbench as score_vmcbench_dev,
    parse_multi_choice_response,
    get_mc_score,
    report_vmc_acc,
    build_choices,
    build_option_str,
    can_infer_option,
    eval_vmc_item,
    process_vmc_batch,
    calculate_vmc_accuracy,
    build_vmc_prompt
)

# Re-export all functions with VMCBench_DEV naming for clarity
__all__ = [
    'extract_answer_vmcbench_dev',
    'score_vmcbench_dev', 
    'parse_multi_choice_response',
    'get_mc_score',
    'report_vmc_acc',
    'build_choices',
    'build_option_str',
    'can_infer_option',
    'eval_vmc_item',
    'process_vmc_batch',
    'calculate_vmc_accuracy',
    'build_vmc_prompt'
]


def extract_answer(prediction_text, choices=None):
    """
    Extract answer from VMCBench_DEV prediction text.
    
    Args:
        prediction_text: The model's prediction text
        choices: Optional dictionary of choices for content-based matching
        
    Returns:
        Extracted answer option letter or None if no answer found
    """
    return extract_answer_vmcbench_dev(prediction_text, choices)


def score(predicted_answer, ground_truth_answer, choices=None):
    """
    Score VMCBench_DEV prediction against ground truth.
    
    Args:
        predicted_answer: The predicted answer option letter
        ground_truth_answer: The ground truth answer option letter
        choices: Optional dictionary of choices for content-based matching
        
    Returns:
        Dictionary containing scoring results
    """
    return score_vmcbench_dev(predicted_answer, ground_truth_answer, choices)