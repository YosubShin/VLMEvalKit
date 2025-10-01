import string
import copy as cp
import os
from ..smp import *
import re


def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    # First, try to extract from common answer formats using regex
    # Check for XML-style tags: <answer>D</answer>
    xml_match = re.search(r'<answer>\s*([A-Z])\s*</answer>', answer, re.IGNORECASE)
    if xml_match and xml_match.group(1).upper() in choices:
        return xml_match.group(1).upper()

    # Check for LaTeX boxed format: \boxed{D}
    latex_match = re.search(r'\\boxed\s*\{\s*([A-Z])\s*\}', answer, re.IGNORECASE)
    if latex_match and latex_match.group(1).upper() in choices:
        return latex_match.group(1).upper()

    # Check for various answer patterns
    patterns = [
        # Original pattern: "Answer: D", "The answer is D", "The correct answer is D"
        r'(?:(?:The|the)?\s*(?:correct|best|final)?\s*answer\s*(?:is|:)\s*)([A-Z])(?:\s|$|\.|\,)',
        # "Final Answer: Z" with optional asterisks
        r'(?:Final\s+Answer\s*:\s*\**)([A-Z])(?:\**)(?:\s|$|\.|\,)',
        # "**Final Answer: Z**"
        r'\*\*Final\s+Answer\s*:\s*([A-Z])\*\*',
        # "Final output: B" or "✅ Final output: **A**"
        r'(?:✅\s*)?Final\s+output\s*:\s*\**([A-Z])\**(?:\s|$|\.|\,)',
        # "### Final Output: Z"
        r'#{1,3}\s*Final\s+Output\s*:\s*\**([A-Z])\**(?:\s|$|\.|\,)',
        # "Output: A"
        r'(?:^|\s)Output\s*:\s*\**([A-Z])\**(?:\s|$|\.|\,)',
        # Standalone bold answer at end: "**Z**"
        r'\*\*([A-Z])\*\*\s*$',
        # Bold answer anywhere: "**A**" (less strict)
        r'\*\*([A-Z])\*\*'
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.MULTILINE)
        if match and match.group(1).upper() in choices:
            return match.group(1).upper()

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    # Extended character list to include quotes and angle brackets
    chars = '.()[],:;!*#{}"\'<>'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                logger = get_logger('Evaluation')
                logger.info(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_sequence(answer, choices=None):
    answer_upper = answer.upper()

    sequence_match = re.search(r'\b([A-D]{4})\b', answer_upper)
    if sequence_match:
        candidate = sequence_match.group(1)
        if len(set(candidate)) == 4:
            return candidate

    order_patterns = [
        r'(?:first|1st|首先|第一步).*?([A-D])',
        r'(?:second|2nd|其次|第二步).*?([A-D])',
        r'(?:third|3rd|再次|第三步).*?([A-D])',
        r'(?:fourth|4th|最后|第四步).*?([A-D])'
    ]

    sequence = []
    for pattern in order_patterns:
        match = re.search(pattern, answer_upper, re.IGNORECASE)
        if match:
            option = match.group(1).upper()
            if option not in sequence:
                sequence.append(option)

    if len(sequence) == 4:
        return ''.join(sequence)

    step_pattern = (
        r'(?:step\s*[\d一二三四]+|'
        r'步骤\s*[\d一二三四]+|'
        r'第\s*[\d一二三四]\s*步)'
        r'.*?([A-D])'
    )
    step_matches = re.findall(step_pattern, answer_upper, re.IGNORECASE)
    if len(step_matches) >= 4:
        unique = []
        for m in step_matches[:4]:
            if m.upper() not in unique:
                unique.append(m.upper())
        if len(unique) == 4:
            return ''.join(unique)

    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def can_infer_lego(answer, question_type, choices):
    answer = str(answer)
    if question_type == 'sort':
        copt = can_infer_sequence(answer, choices)
    else:  # multiple-choice
        copt = can_infer_option(answer, choices)  # option
    return copt if copt else can_infer_text(answer, choices)
