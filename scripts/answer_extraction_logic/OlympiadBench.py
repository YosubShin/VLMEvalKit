"""
Answer extraction and scoring logic for OlympiadBench benchmark.

This module contains the complete answer extraction and scoring pipeline for the 
OlympiadBench benchmark, which evaluates mathematical reasoning on competition-level 
problems with sophisticated mathematical equivalence checking.
"""

import re
import json
from math import isclose
from decimal import Decimal, getcontext
from fractions import Fraction
import sys
import math
import timeout_decorator
import logging
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

try:
    import sympy as sp
    from sympy import simplify, Eq, sympify, evalf, Pow
    from sympy.parsing.latex import parse_latex
    import antlr4
except ImportError:
    logging.warning('sympy or antlr4 is not installed, please install it for OlympiadBench evaluation.')

# Answer type mappings for multilingual support
chinese_answer_type_dict = {
    'Numerical': '数值',
    'Expression': '表达式',
    'Equation': '方程',
    'Interval': '区间'
}

english_answer_type_dict = {
    'Numerical': 'a numerical value',
    'Expression': 'an expression',
    'Equation': 'an equation',
    'Interval': 'an interval'
}

# Set precision for decimal operations
sys.set_int_max_str_digits(1000000)
getcontext().prec = 50


def get_single_answer_type_text(answer_type, is_chinese):
    """
    Get the text description for a single answer type.
    
    Args:
        answer_type: The type of answer expected
        is_chinese: Whether to return Chinese text
        
    Returns:
        Text description of the answer type
    """
    if '-' in answer_type:  # No need now
        answer_type = answer_type[:answer_type.find('-')]
    for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    raise ValueError(f'Error parsing answer type {answer_type}!')


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    """
    Get the full text description for answer type(s).
    
    Args:
        answer_type: The type(s) of answer expected
        is_chinese: Whether to return Chinese text
        multiple_answer: Whether multiple answers are expected
        
    Returns:
        Full text description for the prompt
    """
    # 'Tuple' has various meanings in different context, such as position or values of a series of variable,
    # so it may lead to confusion to directly use 'tuple' in the prompt.
    if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
        full_answer_text = ''
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type, is_chinese)
            if is_chinese:
                full_answer_text = f'，答案类型为{answer_text}'
            else:
                full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if ',' not in answer_type:  # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type, is_chinese)
                if is_chinese:
                    full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                else:
                    full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
            else:
                answer_types = answer_type.split(',')
                answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    if is_chinese:
                        full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                    else:
                        full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                else:
                    if is_chinese:
                        answer_text = '、'.join(answer_types)
                        full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
                    else:
                        answer_text = ', '.join(answer_types)
                        full_answer_text = (
                            f'The problem has multiple answers, with the answers in order being {answer_text}. '
                        )
    return full_answer_text


def make_input(prompt, question_content):
    """
    Create input for the model by combining prompt and question.
    
    Args:
        prompt: The base prompt
        question_content: The question text
        
    Returns:
        Combined input string
    """
    input_text = prompt + '\n' + question_content
    return input_text


class MathJudger:
    """
    Mathematical equivalence judge for OlympiadBench problems.
    Handles complex mathematical expressions, equations, intervals, and numerical values.
    """
    
    def __init__(self):
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    def split_by_comma(self, expr: str):
        """
        Split expression by comma while respecting parentheses.
        
        Args:
            expr: Expression to split
            
        Returns:
            List of split expressions
        """
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char == "(" or char == "[":
                in_bracket_num += 1
            elif char == ")" or char == "]":
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        """
        Transform plus-minus signs into separate expressions.
        
        Args:
            expr_list: List of expressions
            
        Returns:
            Expanded list with plus-minus variations
        """
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-8):
        """
        Judge if two expressions are mathematically equivalent.
        
        Args:
            expression1: First expression (ground truth)
            expression2: Second expression (prediction)
            precision: Numerical precision for comparison
            
        Returns:
            Boolean indicating equivalence
        """
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            return True

        # Remove Chinese characters
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # Design precision list
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                try:
                    if self.is_equal(item1, item2):
                        temp_list1.remove(item1)
                        temp_list2.remove(item2)
                        precision.remove(self.precision)
                        break
                except Exception as err:
                    logging.warning(f'{type(err)}: {err}')
                    continue
            else:
                # If we didn't break from the inner loop, no match was found
                return False

        # If all elements are matched and removed, the lists can be paired
        return True

    def is_interval(self, expr):
        """Check if expression is an interval notation."""
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    @timeout_decorator.timeout(30)
    def is_equal(self, expression1, expression2):
        """
        Check if two expressions are equal using multiple methods.
        
        Args:
            expression1: First expression
            expression2: Second expression
            
        Returns:
            Boolean indicating equality
        """
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            return True

        # Check if both are intervals
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    return True
            except:
                return False

        # Check numerical equality
        try:
            if self.numerical_equal(expression1, expression2):
                return True
        except:
            pass

        # Check expression equality
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                return True
        except:
            pass

        # Check equation equality
        try:
            if self.equation_equal(expression1, expression2):
                return True
        except:
            pass

        return False

    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        """
        Check if two numerical expressions are equal within precision.
        
        Args:
            expression1: First expression (ground truth)
            expression2: Second expression (prediction)
            include_percentage: Whether to consider percentage variations
            
        Returns:
            Boolean indicating numerical equality
        """
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):
        """
        Check if two mathematical expressions are equivalent.
        
        Args:
            exp1: First expression
            exp2: Second expression
            
        Returns:
            Boolean indicating expression equality
        """
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        exp_too_long = len(exp1) > 300 or len(exp2) > 300

        # Convert to SymPy format
        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
                    not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(
                            "These two number can not be calculated by current computer for: "
                            f"\"{str(expr1_sym)}\" and \"{str(expr2_sym)}\""
                        )
                        return False
                    if exp_too_long:
                        print(f'Expression {exp1} or {exp2} is too long to compute. ')
                        return False

                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:
                    return False
            elif exp_too_long:
                print(f'Expression {exp1} or {exp2} is too long to compute. ')
                return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)
                    num_value = simplified_expr.evalf()
                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):
        """
        Check if two equations are equivalent.
        
        Args:
            expression1: First equation
            expression2: Second equation
            
        Returns:
            Boolean indicating equation equality
        """
        def simplify_equation(latex_eq):
            # Split equation into left and right sides
            lhs, rhs = latex_eq.split('=')

            # Parse LaTeX expressions
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            # Create equation object
            equation = Eq(lhs_expr, rhs_expr)

            # Simplify equation: move right side to left
            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        # If the quotient of the two equations is a non-zero integer, they are equivalent
        if (division_result_1.is_Integer and division_result_1 != 0) or (
                division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        """
        Check if two interval expressions are equal.
        
        Args:
            expression1: First interval
            expression2: Second interval
            
        Returns:
            Boolean indicating interval equality
        """
        def compare_two_interval(inter1, inter2):
            # First compare brackets
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            # Split interval parts
            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):
        """
        Preprocess expressions by extracting boxed content and cleaning.
        
        Args:
            expression1: First expression
            expression2: Second expression
            
        Returns:
            Tuple of preprocessed expressions
        """
        def extract_boxed_content(latex_str):
            # Find all \\boxed{...} structures
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                # Search from after \\boxed{ until finding matching closing bracket
                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    # Extract content inside \\boxed{}
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    # If brackets don't match, return error
                    raise ValueError("Mismatched braces in LaTeX string.")

            # If no \\boxed{} found, extract formulas from last line
            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def special_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            # Replace special characters
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = special_symbol_replace(exp1), special_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):
        """
        Check if the power expression can be computed.
        
        Args:
            expr: SymPy expression to check
            
        Returns:
            Boolean indicating if computation is feasible
        """
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()

            if base.is_number and exp.is_number:
                MAX_EXP = 1000  # Maximum exponent threshold

                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True

    def sympy_sub_pi(self, expression_sympy):
        """Substitute pi symbol with numerical approximation."""
        return expression_sympy.subs(self.pi, math.pi)


def extract_answer(is_chinese, model_output, is_deepseek=False):
    """
    Extract answer from model output for OlympiadBench.
    
    Args:
        is_chinese: Whether the output is in Chinese
        model_output: The model's output text
        is_deepseek: Whether using DeepSeek model format
        
    Returns:
        Extracted answer string
    """
    if str(model_output) == 'nan':
        model_output = 'nan'

    if is_deepseek:
        if is_chinese:
            matches = re.findall('## 解题答案(.*)', model_output)
        else:
            matches = re.findall('The answer is: (.*)', model_output)

        if matches:
            model_answer = matches[-1].strip()
            return model_answer
        else:
            return model_output

    if is_chinese:
        matches = re.findall('所以最终答案是(.*)', model_output)
    else:
        matches = re.findall('So the final answer is (.*)', model_output)

    if matches:
        model_answer = matches[-1].strip()
        return model_answer
    else:
        return model_output


def calculate_olympiad_accuracy(result_file):
    """
    Calculate accuracy for OlympiadBench results.
    
    Args:
        result_file: Path to result file or DataFrame
        
    Returns:
        DataFrame with accuracy results
    """
    # This would need to be adapted based on the actual data format
    if isinstance(result_file, str):
        data = pd.read_csv(result_file)  # Adjust based on actual file format
    else:
        data = result_file
    
    total = len(data)
    if total == 0:
        return pd.DataFrame({'accuracy': [0.0], 'total': [0], 'correct': [0]})
    
    correct = sum(data.get('correct', data.get('hit', [])))
    accuracy = correct / total
    
    return pd.DataFrame({
        'accuracy': [round(accuracy * 100, 2)],
        'total': [total],
        'correct': [correct]
    })


def extract_answer_olympiadbench(prediction_text):
    """
    Extract answer from OlympiadBench prediction text.
    
    Args:
        prediction_text: The model's prediction text
        
    Returns:
        Extracted answer string or original text if no specific pattern found
    """
    return extract_answer(is_chinese=False, model_output=prediction_text, is_deepseek=False)


def score_olympiadbench(predicted_answer, ground_truth_answer, precision=1e-8):
    """
    Score OlympiadBench prediction against ground truth using mathematical equivalence.
    
    Args:
        predicted_answer: The predicted answer string
        ground_truth_answer: The ground truth answer string
        precision: Numerical precision for comparison
        
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
    
    # Initialize math judger
    judger = MathJudger()
    
    try:
        # Use mathematical equivalence checking
        is_correct = judger.judge(ground_truth_answer, predicted_answer, precision)
        
        return {
            'correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'explanation': 'Mathematical equivalence check',
            'extracted_answer': predicted_answer
        }
    except Exception as e:
        logging.warning(f'Error in mathematical equivalence check: {e}')
        
        # Fallback to string comparison
        is_correct = predicted_answer.strip() == ground_truth_answer.strip()
        
        return {
            'correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'explanation': f'String comparison (math check failed: {e})',
            'extracted_answer': predicted_answer
        }