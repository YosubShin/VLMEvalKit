#!/usr/bin/env python3
"""
VMCBench Standalone Scorer

A standalone scoring system for VMCBench benchmarks that implements a 4-stage
answer matching pipeline incorporating strategies from multiple VLMEvalKit benchmarks.

Usage:
    python scripts/vmcbench_standalone_scorer.py \
        --benchmarks VMCBench_DEV VMCBench_TEST \
        --input-dir results/full/Qwen2.5-VL-7B-Instruct \
        --llm-backend openai \
        --model gpt-4o-mini \
        --verbose
"""

import argparse
import json
import logging
import os
import re
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Mathematical libraries
try:
    import sympy as sp
    from sympy import simplify, expand, trigsimp, sympify, Eq
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: SymPy not available. Mathematical equivalence checking will be limited.")

# LLM backend libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class VMCBenchScorer:
    """
    Main scorer class that implements the 4-stage answer matching pipeline.
    """
    
    VALID_BENCHMARKS = [
        'VMCBench_DEV', 'VMCBench_TEST', 'atomic_dataset', 'LiveXivVQA', 
        'OlympiadBench', 'Omni3DBench'
    ]
    
    def __init__(self, benchmarks: List[str], input_dir: str, output_dir: Optional[str] = None,
                 llm_backend: str = 'openai', model: str = 'gpt-4o-mini', 
                 api_key: Optional[str] = None, verbose: bool = False, max_samples: Optional[int] = None, resume: bool = False):
        """
        Initialize the VMCBench scorer.
        
        Args:
            benchmarks: List of benchmark names to process
            input_dir: Directory containing input XLSX files
            output_dir: Directory for output files (defaults to input_dir)
            llm_backend: LLM backend ('openai' or 'anthropic')
            model: Model name for LLM judge
            api_key: API key for LLM service
            verbose: Enable verbose logging
            max_samples: Maximum number of samples to process per benchmark (for testing)
            resume: Resume from existing results file by skipping processed samples
        """
        self.benchmarks = benchmarks
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.verbose = verbose
        self.max_samples = max_samples
        self.resume = resume
        
        # Validate benchmarks
        for benchmark in benchmarks:
            if benchmark not in self.VALID_BENCHMARKS:
                print(f"Warning: {benchmark} not in validated benchmark list")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize LLM judge
        self.llm_judge = self._init_llm_judge(llm_backend, model, api_key)
        
        self.logger.info(f"Initialized VMCBench scorer for benchmarks: {benchmarks}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_llm_judge(self, backend: str, model: str, api_key: Optional[str]):
        """Initialize LLM judge backend."""
        if backend == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available. Install with: pip install openai")
            return OpenAIJudge(model=model, api_key=api_key)
        elif backend == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not available. Install with: pip install anthropic")
            return AnthropicJudge(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")
    
    def find_benchmark_files(self) -> Dict[str, Path]:
        """
        Find XLSX files matching the benchmark naming pattern.
        
        Returns:
            Dictionary mapping benchmark names to file paths
        """
        found_files = {}
        
        for benchmark in self.benchmarks:
            # Pattern: {model_name}_{benchmark_name}.xlsx
            pattern = f"*_{benchmark}.xlsx"
            matches = list(self.input_dir.glob(pattern))
            
            if matches:
                if len(matches) > 1:
                    self.logger.warning(f"Multiple files found for {benchmark}: {matches}")
                    self.logger.warning(f"Using: {matches[0]}")
                found_files[benchmark] = matches[0]
                self.logger.info(f"Found file for {benchmark}: {matches[0]}")
            else:
                self.logger.error(f"No file found for benchmark {benchmark} with pattern {pattern}")
        
        return found_files
    
    def load_benchmark_data(self, file_path: Path, benchmark_name: str) -> pd.DataFrame:
        """
        Load benchmark data from XLSX file.
        
        Args:
            file_path: Path to XLSX file
            benchmark_name: Name of the benchmark being processed
            
        Returns:
            DataFrame with benchmark data
        """
        try:
            df = pd.read_excel(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Validate required columns
            required_cols = ['answer', 'prediction']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Ensure we have a unique identifier column
            if 'index' not in df.columns and df.index.name != 'index':
                df = df.reset_index(drop=False)
                if 'index' not in df.columns:
                    df['index'] = range(len(df))
            
            # Handle resume mode
            if self.resume:
                df = self._filter_unprocessed_samples(df, file_path, benchmark_name)
            
            # Convert to string and handle NaN values
            df['answer'] = df['answer'].astype(str).fillna('')
            df['prediction'] = df['prediction'].astype(str).fillna('')
            
            # Apply sampling if max_samples is specified
            if self.max_samples and len(df) > self.max_samples:
                df = df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
                self.logger.info(f"Sampled {len(df)} rows from original dataset")
            
            self.logger.info(f"Available columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def _extract_choices(self, row: pd.Series) -> Dict[str, str]:
        """
        Extract multiple choice options from row if available.
        
        Args:
            row: Pandas Series containing row data
            
        Returns:
            Dictionary mapping choice letters to choice text
        """
        choices = {}
        for i in range(9):  # Support up to 9 choices (A-Z)
            choice_letter = chr(65 + i)  # A, B, C, D, ...
            if choice_letter in row and pd.notna(row[choice_letter]):
                choices[choice_letter] = str(row[choice_letter])
        return choices
    
    def _filter_unprocessed_samples(self, df: pd.DataFrame, file_path: Path, benchmark_name: str) -> pd.DataFrame:
        """
        Filter out samples that have already been processed when resuming.
        
        Args:
            df: Input DataFrame
            file_path: Path to the input file
            benchmark_name: Name of the benchmark
            
        Returns:
            DataFrame with only unprocessed samples
        """
        output_path = self.output_dir / f"{file_path.stem}_scored.xlsx"
        
        if not output_path.exists():
            raise FileNotFoundError(
                f"Resume mode enabled but no existing results file found at: {output_path}. "
                f"Remove --resume flag to create a new results file."
            )
        
        try:
            # Load existing results
            existing_df = pd.read_excel(output_path)
            self.logger.info(f"Found existing results file with {len(existing_df)} rows")
            
            # Get processed sample indices/IDs
            if 'index' in existing_df.columns:
                processed_indices = set(existing_df['index'].values)
            else:
                processed_indices = set(existing_df.index.values)
            
            # Filter out processed samples
            if 'index' in df.columns:
                mask = ~df['index'].isin(processed_indices)
            else:
                mask = ~df.index.isin(processed_indices)
            
            unprocessed_df = df[mask].reset_index(drop=True)
            
            self.logger.info(
                f"Resume mode: Found {len(existing_df)} processed samples, "
                f"{len(unprocessed_df)} samples remaining to process"
            )
            
            return unprocessed_df
            
        except Exception as e:
            raise RuntimeError(
                f"Error reading existing results file {output_path}: {e}. "
                f"File may be corrupted or in wrong format."
            )
    
    def _append_results_to_existing(self, new_results: pd.DataFrame, output_path: Path):
        """
        Append new results to existing results file.
        
        Args:
            new_results: New DataFrame with results to append
            output_path: Path to the existing results file
        """
        try:
            existing_df = pd.read_excel(output_path)
            combined_df = pd.concat([existing_df, new_results], ignore_index=True)
            combined_df.to_excel(output_path, index=False)
            self.logger.info(f"Appended {len(new_results)} new results to existing file")
        except Exception as e:
            self.logger.error(f"Error appending results: {e}")
            raise
    
    def process_benchmark(self, benchmark_name: str, file_path: Path):
        """
        Process a single benchmark file through the 4-stage pipeline.
        
        Args:
            benchmark_name: Name of the benchmark
            file_path: Path to the benchmark file
        """
        self.logger.info(f"Processing benchmark: {benchmark_name}")
        
        # Load data
        df = self.load_benchmark_data(file_path, benchmark_name)
        
        # Check if there are any samples to process
        if len(df) == 0:
            self.logger.info(f"No samples to process for {benchmark_name} (all already processed)")
            return
        
        # Apply 4-stage pipeline
        df_scored = self.apply_four_stage_pipeline(df)
        
        # Save results
        output_path = self.output_dir / f"{file_path.stem}_scored.xlsx"
        
        if self.resume and output_path.exists():
            # Append to existing results
            self._append_results_to_existing(df_scored, output_path)
        else:
            # Create new results file
            df_scored.to_excel(output_path, index=False)
            self.logger.info(f"Saved scored results to: {output_path}")
        
        # Print summary statistics
        self._print_summary_stats(df_scored, benchmark_name)
    
    def stage1_simple_match(self, prediction: str, answer: str) -> Tuple[str, bool, str]:
        """
        Stage 1: Simple exact matching strategies only.
        
        Applies only the most conservative exact matching strategies:
        1. Exact match: prediction == answer (after stripping)
        2. Case-insensitive match: prediction.lower() == answer.lower()
        3. Whitespace-normalized match: handles extra/missing whitespace
        
        Note: Moved single character extraction and boolean pattern matching to Stage 2
        to consolidate with similar strategies and reduce duplication.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        # Clean inputs
        pred_clean = prediction.strip()
        answer_clean = answer.strip()
        
        # Strategy 1: Exact match
        if pred_clean == answer_clean:
            return answer_clean, True, "Exact match"
        
        # Strategy 2: Case-insensitive match
        if pred_clean.lower() == answer_clean.lower():
            return answer_clean, True, "Case-insensitive match"
        
        # Strategy 3: Extra whitespace removal
        pred_no_space = re.sub(r'\s+', ' ', pred_clean).strip()
        answer_no_space = re.sub(r'\s+', ' ', answer_clean).strip()
        if pred_no_space.lower() == answer_no_space.lower():
            return answer_clean, True, "Whitespace-normalized match"
        
        return pred_clean, False, "No simple match found"
    
    def stage2_complex_match(self, prediction: str, answer: str, 
                           choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Stage 2: Unified complex extraction strategies with comprehensive match collection.
        
        Behavior:
        - Searches from END of response to prioritize final answers over intermediate work
        - Collects ALL matches from ALL strategies for full visibility
        - Uses priority order: LaTeX > Math > Tags > SymPy > MCQ* > Language > Boolean*
        - Applies heuristics to reduce false positives:
          * MCQ and Boolean detectors only run when ground truth is short (<15 chars)
          * Numeric detector removed completely (too many false positives)
          * SymPy equivalence prioritized above simple pattern matching
        - Returns first successful match but reports all findings in error message
        - Includes match positions in text for debugging
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Dictionary of multiple choice options (optional)
            
        Returns:
            Tuple of (extracted_answer, success, detailed_match_report)
        """
        # Define core strategies (always run)
        core_strategies = [
            ("LaTeX Boxed", lambda: self._extract_latex_boxed_with_positions(prediction)),
            ("Math Expressions", lambda: self._extract_math_expressions_with_positions(prediction)),
            ("Structured Tags", lambda: self._extract_structured_tags_with_positions(prediction)),
            ("SymPy Equivalence", lambda: self._check_mathematical_equivalence_with_positions(prediction, answer)),
            ("Natural Language", lambda: self._extract_natural_language_with_positions(prediction)),
        ]
        
        # Define conditional strategies (only run when ground truth is short)
        conditional_strategies = []
        if len(answer.strip()) < 15:  # Only run for short ground truth to reduce false positives
            conditional_strategies.extend([
                ("Multiple Choice", lambda: self._extract_multiple_choice_with_positions(prediction, choices_dict)),
                ("Boolean Answers", lambda: self._extract_boolean_answers_with_positions(prediction)),
            ])
        
        # Combine strategies in priority order
        strategies = core_strategies + conditional_strategies
        # Note: Numeric Answers detector removed completely due to excessive false positive risk
        
        all_matches = []
        selected_result = None
        selected_strategy = None
        
        # Collect all matches from all strategies
        for strategy_name, strategy_func in strategies:
            try:
                matches = strategy_func()
                if matches:  # matches is now a list of (content, start_pos, end_pos) tuples
                    all_matches.extend([(strategy_name, content, start_pos, end_pos) for content, start_pos, end_pos in matches])
                    
                    # Select first valid result using priority order (if not already selected)
                    if not selected_result:
                        # Get the last (rightmost) match from this strategy
                        last_match = max(matches, key=lambda x: x[1])  # max by start_pos
                        content = last_match[0]
                        if content and content.strip() != prediction.strip():
                            # Strict length-based heuristic: require exact length match for structured extraction
                            # This prevents structured tag extraction from returning full explanations instead of short answers
                            gt_len = len(answer.strip())
                            content_len = len(content.strip())
                            
                            # Apply strict length filter for structured extraction strategies
                            if strategy_name in ["Structured Tags", "Math Expressions"]:
                                # Require exact length match - if lengths don't match, reject this extraction
                                if content_len != gt_len:
                                    continue  # Skip this match, try next strategy
                            
                            selected_result = content.strip()
                            selected_strategy = strategy_name
                            
            except Exception as e:
                all_matches.append((strategy_name, f"ERROR: {str(e)}", -1, -1))
                continue
        
        # Build comprehensive match report
        if all_matches:
            # Sort matches by position (rightmost first for "end-searching" perspective)
            position_sorted = sorted([m for m in all_matches if m[2] >= 0], key=lambda x: x[2], reverse=True)
            error_matches = [m for m in all_matches if m[2] == -1]
            
            match_details = []
            for strategy, content, start, end in position_sorted:
                if start >= 0:
                    match_details.append(f"{strategy}@{start}-{end}: '{content}'")
            
            for strategy, error, _, _ in error_matches:
                match_details.append(f"{strategy}: {error}")
            
            # Add heuristic info to report
            heuristic_info = f"GT_len={len(answer.strip())}, MCQ/Bool={'enabled' if len(answer.strip()) < 15 else 'disabled'}, Strict_length_filter=enabled"
            match_report = f"Matches found: {'; '.join(match_details)} | Heuristics: {heuristic_info}"
            
            if selected_result:
                # Check if selected result matches ground truth
                if selected_result.lower() == answer.lower():
                    return answer, True, f"SUCCESS with {selected_strategy}: '{selected_result}' | {match_report}"
                else:
                    return selected_result, True, f"EXTRACTED with {selected_strategy}: '{selected_result}' | {match_report}"
            else:
                return prediction, False, f"NO VALID EXTRACTION | {match_report}"
        else:
            heuristic_info = f"GT_len={len(answer.strip())}, MCQ/Bool={'enabled' if len(answer.strip()) < 15 else 'disabled'}, Strict_length_filter=enabled"
            return prediction, False, f"No matches found across all strategies | Heuristics: {heuristic_info}"
    
    def stage3_llm_judge(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Stage 3: LLM equivalence checking.
        
        Uses the initialized LLM judge to determine semantic equivalence.
        For multiple choice questions (LiveXivVQA, VMCBench_DEV), provides choice context.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Optional dictionary mapping choice letters to their values (A->value, B->value, etc.)
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        return self.llm_judge.judge_equivalence(prediction, answer, choices_dict)
    
    # Unified extraction methods for Stage 2 with position tracking
    def _extract_latex_boxed_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract content from LaTeX \boxed{} format with positions."""
        matches = []
        
        # Method 1: Complex nested bracket parsing (most robust)
        for match in re.finditer(r'\\boxed{', prediction):
            start_index = match.end()
            end_index = start_index
            stack = 1
            
            while stack > 0 and end_index < len(prediction):
                if prediction[end_index] == '{':
                    stack += 1
                elif prediction[end_index] == '}':
                    stack -= 1
                end_index += 1
            
            if stack == 0:
                content = prediction[start_index:end_index - 1].strip()
                if content:
                    matches.append((content, match.start(), end_index))
        
        # Method 2: Simple regex fallback for basic cases (if no complex matches found)
        if not matches:
            for match in re.finditer(r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}', prediction):
                content = match.group(1).strip()
                if content:
                    matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_latex_boxed(self, prediction: str) -> Optional[str]:
        """Extract content from LaTeX \boxed{} format with nested bracket support."""
        # Method 1: Complex nested bracket parsing (most robust)
        boxed_matches = re.finditer(r'\\boxed{', prediction)
        results = []
        
        for match in boxed_matches:
            start_index = match.end()
            end_index = start_index
            stack = 1
            
            while stack > 0 and end_index < len(prediction):
                if prediction[end_index] == '{':
                    stack += 1
                elif prediction[end_index] == '}':
                    stack -= 1
                end_index += 1
            
            if stack == 0:
                content = prediction[start_index:end_index - 1]
                results.append(content.strip())
        
        if results:
            return results[-1]  # Return last (most recent) boxed content
        
        # Method 2: Simple regex fallback for basic cases
        simple_pattern = r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
        matches = re.findall(simple_pattern, prediction)
        if matches:
            return matches[-1].strip()
        
        # Backward compatibility - return last match
        matches = self._extract_latex_boxed_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_math_expressions_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract mathematical expressions with positions."""
        matches = []
        
        # Dollar-wrapped math: $expression$
        for match in re.finditer(r'\$([^$]+)\$', prediction):
            content = match.group(1).strip()
            if content:
                matches.append((content, match.start(), match.end()))
        
        # LaTeX expressions without \boxed (only if SymPy available)
        if SYMPY_AVAILABLE:
            try:
                from sympy.parsing.latex import parse_latex
                # Try to parse the whole prediction as LaTeX
                expr = parse_latex(prediction)
                expr_str = str(expr)
                if expr_str != prediction:
                    matches.append((expr_str, 0, len(prediction)))
            except Exception:
                pass
        
        return matches
    
    def _extract_math_expressions(self, prediction: str) -> Optional[str]:
        """Extract mathematical expressions from various formats."""
        # Dollar-wrapped math: $expression$
        dollar_pattern = r'\$([^$]+)\$'
        dollar_matches = re.findall(dollar_pattern, prediction)
        if dollar_matches:
            return dollar_matches[-1].strip()
        
        # LaTeX expressions without \boxed
        if SYMPY_AVAILABLE:
            try:
                from sympy.parsing.latex import parse_latex
                expr = parse_latex(prediction)
                return str(expr)
            except Exception:
                pass
        else:
            self.logger.warning("SymPy not available - LaTeX expression parsing disabled. Install with: pip install sympy")
        
        # Backward compatibility - return last match
        matches = self._extract_math_expressions_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_structured_tags_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract content from structured tags with positions."""
        matches = []
        
        # XML-style tags
        tag_patterns = [
            r'<ans>(.*?)</ans>',           # Omni3D format
            r'<answer>(.*?)</answer>',     # Generic answer tags
            r'<result>(.*?)</result>',     # Result tags
            r'<final>(.*?)</final>',       # Final answer tags
            r'\[ANSWER\](.*?)\[/ANSWER\]', # Bracket format
            r'\[ANS\](.*?)\[/ANS\]',       # Alternative bracket
        ]
        
        for pattern in tag_patterns:
            for match in re.finditer(pattern, prediction, re.IGNORECASE | re.DOTALL):
                content = match.group(1).strip()
                if content:
                    matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_structured_tags(self, prediction: str) -> Optional[str]:
        """Extract content from structured XML-style tags and brackets."""
        # XML-style tags
        tag_patterns = [
            r'<ans>(.*?)</ans>',           # Omni3D format
            r'<answer>(.*?)</answer>',     # Generic answer tags
            r'<result>(.*?)</result>',     # Result tags
            r'<final>(.*?)</final>',       # Final answer tags
            r'\[ANSWER\](.*?)\[/ANSWER\]', # Bracket format
            r'\[ANS\](.*?)\[/ANS\]',       # Alternative bracket
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Backward compatibility - return last match
        matches = self._extract_structured_tags_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_multiple_choice_with_positions(self, prediction: str, choices_dict: Optional[Dict[str, str]]) -> List[Tuple[str, int, int]]:
        """Extract multiple choice answers with positions (enhanced with Stage 1 logic)."""
        matches = []
        
        if not choices_dict:
            # Generic MCQ extraction without specific choices (enhanced with Stage 1 patterns)
            mcq_patterns = [
                r'\b([A-Z])\b',  # Isolated single letter (merged from Stage 1)
                r'(?:^|\s)([A-Z])(?:\s|$|\.|,)',  # Single letter with boundaries
                r'(?:is|are)\s+([A-Z])(?:\s|$|\.|,)',  # "The answer is A"
                r'(?:option|choice)\s+([A-Z])(?:\s|$|\.|,)',  # "Option A"
                r'([A-Z])\s*(?:is|are)\s*(?:correct|right)',  # "A is correct"
                r'\(([A-Z])\)',  # (A)
                r'([A-Z])\.',    # A.
                r'^([A-Z])\.\s',  # B. (at start of string/line)
                r'\n([A-Z])\.\s',  # B. (at start of new line)
            ]
            
            for pattern in mcq_patterns:
                for match in re.finditer(pattern, prediction, re.IGNORECASE):
                    content = match.group(1).upper().strip()
                    if content:
                        matches.append((content, match.start(), match.end()))
        else:
            # Specific choice-based extraction
            response = str(prediction)
            all_choices = list(choices_dict.keys())
            
            # Pattern 1: Bracketed options (A), (B) and formatted choices A., B. 
            for choice in all_choices:
                patterns = [
                    f'\\({choice}\\)',      # (A)
                    f'{choice}\\.\\s',      # A. (anywhere)
                    f'^{choice}\\.\\s',     # A. (at start of string)
                    f'\\n{choice}\\.\\s',    # A. (at start of line)
                ]
                for pattern in patterns:
                    for match in re.finditer(pattern, response):
                        matches.append((choice, match.start(), match.end()))
            
            # Pattern 2: Standalone letters
            for choice in all_choices:
                for match in re.finditer(f'\\s{choice}\\s', response):
                    matches.append((choice, match.start(), match.end()))
            
            # Pattern 3: Content matching (assign position as end of prediction)
            for choice, choice_text in choices_dict.items():
                if choice_text.lower() in response.lower():
                    pos = response.lower().find(choice_text.lower())
                    if pos >= 0:
                        matches.append((choice, pos, pos + len(choice_text)))
        
        return matches
    
    def _extract_multiple_choice(self, prediction: str, choices_dict: Optional[Dict[str, str]]) -> Optional[str]:
        """Extract multiple choice answers (A, B, C, D, etc.)."""
        if not choices_dict:
            # Generic MCQ extraction without specific choices
            mcq_patterns = [
                r'(?:^|\s)([A-Z])(?:\s|$|\.|,)',  # Single letter
                r'(?:is|are)\s+([A-Z])(?:\s|$|\.|,)',  # "The answer is A"
                r'(?:option|choice)\s+([A-Z])(?:\s|$|\.|,)',  # "Option A"
                r'([A-Z])\s*(?:is|are)\s*(?:correct|right)',  # "A is correct"
                r'\(([A-Z])\)',  # (A)
                r'([A-Z])\.',    # A.
            ]
            
            for pattern in mcq_patterns:
                matches = re.findall(pattern, prediction, re.IGNORECASE)
                if matches:
                    return matches[0].upper().strip()
            return None
        
        # Specific choice-based extraction
        response = str(prediction)
        all_choices = list(choices_dict.keys())
        
        # Clean response
        for char in [',', '.', '!', '?', ';', ':', "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
        
        candidates = []
        
        # Pattern 1: Bracketed options (A), (B) or A., B.
        for choice in all_choices:
            if f'({choice})' in response or f'{choice}. ' in response:
                candidates.append(choice)
        
        # Pattern 2: Standalone letters " A ", " B "
        if len(candidates) == 0:
            for choice in all_choices:
                if f' {choice} ' in response:
                    candidates.append(choice)
        
        # Pattern 3: Content matching
        if len(candidates) == 0 and len(response.split()) > 5:
            for choice, choice_text in choices_dict.items():
                if choice_text.lower() in response.lower():
                    candidates.append(choice)
        
        # Backward compatibility - return last match
        matches = self._extract_multiple_choice_with_positions(prediction, choices_dict)
        return matches[-1][0] if matches else None
    
    def _extract_natural_language_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract answers from natural language patterns with positions."""
        matches = []
        
        # Common answer introduction patterns
        patterns = [
            r'So the final answer is\s*([^.\n]+)',
            r'Therefore,?\s*the answer is\s*([^.\n]+)',
            r'The answer is\s*([^.\n]+)',
            r'Answer:\s*([^\n]+)',
            r'Final Answer:\s*([^\n]+)',
            r'Solution:\s*([^\n]+)',
            r'Result:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, prediction, re.IGNORECASE):
                content = match.group(1).strip()
                if content:
                    matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_natural_language_answers(self, prediction: str) -> Optional[str]:
        """Extract answers from natural language patterns."""
        # Common answer introduction patterns
        patterns = [
            r'So the final answer is\s*([^.\n]+)',
            r'Therefore,?\s*the answer is\s*([^.\n]+)',
            r'The answer is\s*([^.\n]+)',
            r'Answer:\s*([^\n]+)',
            r'Final Answer:\s*([^\n]+)',
            r'Solution:\s*([^\n]+)',
            r'Result:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Backward compatibility - return last match
        matches = self._extract_natural_language_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _extract_boolean_answers_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract boolean answers with positions (enhanced with Stage 1 logic)."""
        matches = []
        pred_lower = prediction.lower()
        
        # Enhanced patterns (merged from Stage 1)
        true_patterns = ['yes', 'true', 'correct', 'right']
        false_patterns = ['no', 'false', 'incorrect', 'wrong']
        
        # Find all true/yes patterns
        for pattern in true_patterns:
            for match in re.finditer(r'\b' + pattern + r'\b', pred_lower):
                matches.append(('yes', match.start(), match.end()))
        
        # Find all false/no patterns  
        for pattern in false_patterns:
            for match in re.finditer(r'\b' + pattern + r'\b', pred_lower):
                matches.append(('no', match.start(), match.end()))
        
        return matches
    
    def _extract_boolean_answers(self, prediction: str) -> Optional[str]:
        """Extract boolean/yes-no style answers."""
        pred_lower = prediction.lower()
        
        # Yes patterns
        yes_patterns = ['yes', 'true', 'correct', 'right']
        no_patterns = ['no', 'false', 'incorrect', 'wrong']
        
        # Count occurrences to handle cases like "not true"
        yes_count = sum(1 for pattern in yes_patterns if pattern in pred_lower)
        no_count = sum(1 for pattern in no_patterns if pattern in pred_lower)
        
        if yes_count > no_count and yes_count > 0:
            return 'yes'
        elif no_count > yes_count and no_count > 0:
            return 'no'
        
        # Backward compatibility - return last match
        matches = self._extract_boolean_answers_with_positions(prediction)
        if not matches:
            return None
        
        # Count yes vs no to determine final answer
        yes_count = sum(1 for content, _, _ in matches if content == 'yes')
        no_count = sum(1 for content, _, _ in matches if content == 'no')
        
        if yes_count > no_count:
            return 'yes'
        elif no_count > yes_count:
            return 'no'
        else:
            return matches[-1][0]  # If tied, return last occurrence
    
    def _extract_numeric_answers_with_positions(self, prediction: str) -> List[Tuple[str, int, int]]:
        """Extract numeric answers with positions."""
        matches = []
        
        # Look for numbers (decimals first for specificity)
        number_patterns = [
            r'-?\d+\.\d+',  # Decimals
            r'-?\d+',       # Integers
        ]
        
        for pattern in number_patterns:
            for match in re.finditer(pattern, prediction):
                content = match.group(0)
                matches.append((content, match.start(), match.end()))
        
        return matches
    
    def _extract_numeric_answers(self, prediction: str) -> Optional[str]:
        """Extract numeric answers from text."""
        # Look for numbers (integers and decimals)
        number_patterns = [
            r'-?\d+\.\d+',  # Decimals first (more specific)
            r'-?\d+',       # Integers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, prediction)
            if matches:
                return matches[-1]  # Return last number found
        
        # Backward compatibility - return last match
        matches = self._extract_numeric_answers_with_positions(prediction)
        return matches[-1][0] if matches else None
    
    def _check_mathematical_equivalence_with_positions(self, prediction: str, answer: str) -> List[Tuple[str, int, int]]:
        """Check mathematical equivalence with position info."""
        if not SYMPY_AVAILABLE:
            return []
            
        try:
            # Try to parse both as mathematical expressions
            pred_expr = sympify(prediction)
            answer_expr = sympify(answer)
            
            # Check if expressions are equivalent
            diff = simplify(pred_expr - answer_expr)
            if diff == 0:
                # Return the answer as equivalent (position spans whole prediction)
                return [(answer, 0, len(prediction))]
                
        except Exception:
            pass
        
        return []
    
    def _check_mathematical_equivalence(self, prediction: str, answer: str) -> Optional[str]:
        """Check mathematical equivalence using SymPy."""
        if not SYMPY_AVAILABLE:
            self.logger.warning("SymPy not available - mathematical equivalence checking disabled. Install with: pip install sympy")
            return None
            
        try:
            # Try to parse both as mathematical expressions
            pred_expr = sympify(prediction)
            answer_expr = sympify(answer)
            
            # Check if expressions are equivalent
            diff = simplify(pred_expr - answer_expr)
            if diff == 0:
                return answer
                
        except Exception:
            pass
        
        return None

    def apply_four_stage_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the 4-stage answer matching pipeline to all rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional scoring columns
        """
        results = []
        
        self.logger.info("Applying 4-stage pipeline...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            prediction = str(row['prediction'])
            answer = str(row['answer'])
            
            # Extract choices if available (for MCQ datasets)
            choices_dict = self._extract_choices(row)
            
            # Initialize results for this row
            row_result = {
                'stage1_match': 0,
                'stage2_match': 0, 
                'stage3_match': 0,
                'stage4_match': 0,
                'final_answer': '',
                'hit': 0,
                'stage_errors': ''
            }
            
            errors = []
            final_answer = None
            success_stage = None
            
            # Stage 1: Simple matching
            try:
                stage1_result, stage1_success, stage1_error = self.stage1_simple_match(prediction, answer)
                if stage1_success:
                    row_result['stage1_match'] = 1
                    final_answer = stage1_result
                    success_stage = 1
                errors.append(f"Stage1: {stage1_error}")
            except Exception as e:
                errors.append(f"Stage1: ERROR - {str(e)}")
                stage1_success = False
            
            # Stage 2: Complex matching (if Stage 1 failed)
            if not stage1_success:
                try:
                    stage2_result, stage2_success, stage2_error = self.stage2_complex_match(
                        prediction, answer, choices_dict
                    )
                    if stage2_success:
                        row_result['stage2_match'] = 1
                        final_answer = stage2_result
                        success_stage = 2
                    errors.append(f"Stage2: {stage2_error}")
                except Exception as e:
                    errors.append(f"Stage2: ERROR - {str(e)}")
                    stage2_success = False
            else:
                errors.append("Stage2: Skipped - Stage 1 succeeded")
                stage2_success = False
            
            # Stage 3: LLM judge (if Stages 1-2 failed)
            if not (stage1_success or stage2_success):
                try:
                    stage3_result, stage3_success, stage3_error = self.stage3_llm_judge(prediction, answer, choices_dict)
                    if stage3_success:
                        row_result['stage3_match'] = 1
                        final_answer = stage3_result
                        success_stage = 3
                    errors.append(f"Stage3: {stage3_error}")
                except Exception as e:
                    errors.append(f"Stage3: ERROR - {str(e)}")
                    stage3_success = False
            else:
                errors.append("Stage3: Skipped - Earlier stage succeeded")
                stage3_success = False
            
            # Stage 4: Fallback
            if not (stage1_success or stage2_success or stage3_success):
                final_answer = "NOMATCH"
                row_result['stage4_match'] = 1
                success_stage = 4
                errors.append("Stage4: Fallback - NOMATCH")
            else:
                errors.append("Stage4: Not needed")
            
            # Set final results
            row_result['final_answer'] = final_answer or "NOMATCH"
            
            # Determine scoring method based on answer_type column
            if 'answer_type' in row and row['answer_type'] == 'float':
                # Use MRA (Mean Relative Accuracy) scoring for float answer types
                score_result = self._calculate_mra_score(final_answer, answer)
                row_result['hit'] = score_result['score']
                row_result['mra_details'] = score_result.get('mra_details', '')
            else:
                # Use exact match scoring for all other answer types
                row_result['hit'] = 1 if final_answer == answer else 0
                
            row_result['stage_errors'] = " | ".join(errors)
            
            results.append(row_result)
        
        # Combine original data with results
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df], axis=1)
    
    def _calculate_mra_score(self, predicted_answer: str, ground_truth_answer: str) -> dict:
        """
        Calculate Mean Relative Accuracy (MRA) score for float answer types.
        
        MRA methodology from VSI-Bench paper (https://arxiv.org/abs/2412.14171):
        - Tests prediction accuracy at multiple relative error thresholds
        - MRA = average accuracy across all thresholds
        - Formula: |ground_truth - prediction| / ground_truth < threshold
        
        Args:
            predicted_answer: The predicted answer string
            ground_truth_answer: The ground truth answer string
            
        Returns:
            Dictionary containing MRA score and details
        """
        # MRA thresholds from Omni3DBench implementation
        mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        
        try:
            # Convert to float values
            pred_float = float(predicted_answer)
            gt_float = float(ground_truth_answer)
            
            # Handle division by zero
            if gt_float == 0:
                if pred_float == 0:
                    # Both are zero - perfect match
                    return {
                        'score': 1.0,
                        'mra_details': 'Perfect match: both values are zero',
                        'threshold_scores': {str(t): 1.0 for t in mra_thresholds}
                    }
                else:
                    # Ground truth is zero but prediction is not - no match
                    return {
                        'score': 0.0,
                        'mra_details': 'No match: ground truth is zero but prediction is not',
                        'threshold_scores': {str(t): 0.0 for t in mra_thresholds}
                    }
            
            # Calculate relative error: |gt - pred| / gt
            relative_error = abs(gt_float - pred_float) / abs(gt_float)
            
            # Test each threshold
            threshold_scores = {}
            for threshold in mra_thresholds:
                if relative_error < threshold:
                    threshold_scores[str(threshold)] = 1.0
                else:
                    threshold_scores[str(threshold)] = 0.0
            
            # Calculate MRA as average across all thresholds
            mra_score = sum(threshold_scores.values()) / len(mra_thresholds)
            
            return {
                'score': mra_score,
                'mra_details': f'Relative_error={relative_error:.4f}, MRA={mra_score:.3f}',
                'threshold_scores': threshold_scores,
                'relative_error': relative_error
            }
            
        except (ValueError, TypeError) as e:
            # Could not convert to float - return zero score
            return {
                'score': 0.0,
                'mra_details': f'Type conversion error: {e}',
                'threshold_scores': {str(t): 0.0 for t in mra_thresholds}
            }
    
    def _print_summary_stats(self, df: pd.DataFrame, benchmark_name: str):
        """Print summary statistics for the scored benchmark."""
        total_rows = len(df)
        hits = df['hit'].sum()
        accuracy = hits / total_rows if total_rows > 0 else 0
        
        stage_stats = {
            'Stage 1 (Simple)': df['stage1_match'].sum(),
            'Stage 2 (Complex)': df['stage2_match'].sum(), 
            'Stage 3 (LLM)': df['stage3_match'].sum(),
            'Stage 4 (Fallback)': df['stage4_match'].sum()
        }
        
        print(f"\n=== {benchmark_name} Summary ===")
        print(f"Total rows: {total_rows}")
        print(f"Score total: {hits:.3f}")  # Changed to show fractional scores for MRA
        print(f"Average score: {accuracy:.3f}")
        
        # Check if we have MRA scoring (float answer types)
        if 'answer_type' in df.columns:
            float_rows = df[df['answer_type'] == 'float']
            if len(float_rows) > 0:
                float_score = float_rows['hit'].sum()
                float_avg = float_score / len(float_rows)
                print(f"MRA scoring (float types): {len(float_rows)} rows, avg score: {float_avg:.3f}")
                
                # Show breakdown by answer type
                for answer_type in df['answer_type'].unique():
                    type_rows = df[df['answer_type'] == answer_type]
                    type_score = type_rows['hit'].sum()
                    type_avg = type_score / len(type_rows) if len(type_rows) > 0 else 0
                    print(f"  {answer_type} type: {len(type_rows)} rows, avg score: {type_avg:.3f}")
        
        print("\nStage success counts:")
        for stage, count in stage_stats.items():
            percentage = count / total_rows * 100 if total_rows > 0 else 0
            print(f"  {stage}: {count} ({percentage:.1f}%)")
    
    def _save_summary_to_file(self, summary_lines: List[str]):
        """Save summary statistics to a text file."""
        summary_path = self.output_dir / "vmcbench_scoring_summary.txt"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
            self.logger.info(f"Saved summary statistics to: {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving summary file: {e}")
    
    def process_all(self):
        """Process all benchmarks."""
        found_files = self.find_benchmark_files()
        
        if not found_files:
            self.logger.error("No benchmark files found!")
            return
        
        # Track overall statistics across all benchmarks
        all_benchmark_stats = []
        summary_lines = []
        
        # Add header to summary
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        summary_lines.extend([
            "VMCBench Standalone Scorer - Results Summary",
            f"Generated: {timestamp}",
            f"Benchmarks processed: {', '.join(self.benchmarks)}",
            f"LLM Backend: {self.llm_judge.__class__.__name__ if hasattr(self, 'llm_judge') else 'N/A'}",
            f"Resume mode: {'Enabled' if self.resume else 'Disabled'}",
            f"Max samples: {self.max_samples if self.max_samples else 'No limit'}",
            "=" * 60,
            ""
        ])
        
        for benchmark_name, file_path in found_files.items():
            try:
                # Load the results to get final statistics
                output_path = self.output_dir / f"{file_path.stem}_scored.xlsx"
                
                self.process_benchmark(benchmark_name, file_path)
                
                # Read the results file to get final stats
                if output_path.exists():
                    results_df = pd.read_excel(output_path)
                    total_rows = len(results_df)
                    hits = results_df['hit'].sum() if 'hit' in results_df.columns else 0
                    accuracy = hits / total_rows if total_rows > 0 else 0
                    
                    # Store stats for overall summary
                    all_benchmark_stats.append({
                        'benchmark': benchmark_name,
                        'total': total_rows,
                        'hits': hits,
                        'accuracy': accuracy
                    })
                    
                    # Add to summary text
                    summary_lines.extend([
                        f"=== {benchmark_name} ===",
                        f"Total samples: {total_rows}",
                        f"Correct answers: {hits}",
                        f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)",
                        ""
                    ])
                    
                    # Add stage breakdown if available
                    stage_cols = ['stage1_match', 'stage2_match', 'stage3_match', 'stage4_match']
                    if all(col in results_df.columns for col in stage_cols):
                        stage_stats = {
                            'Stage 1 (Simple)': results_df['stage1_match'].sum(),
                            'Stage 2 (Complex)': results_df['stage2_match'].sum(),
                            'Stage 3 (LLM)': results_df['stage3_match'].sum(),
                            'Stage 4 (Fallback)': results_df['stage4_match'].sum()
                        }
                        
                        summary_lines.append("Stage success breakdown:")
                        for stage, count in stage_stats.items():
                            percentage = count / total_rows * 100 if total_rows > 0 else 0
                            summary_lines.append(f"  {stage}: {count} ({percentage:.1f}%)")
                        summary_lines.append("")
                
            except Exception as e:
                self.logger.error(f"Error processing {benchmark_name}: {e}")
                summary_lines.extend([
                    f"=== {benchmark_name} (ERROR) ===",
                    f"Error: {str(e)}",
                    ""
                ])
                continue
        
        # Calculate and display overall statistics
        if all_benchmark_stats:
            total_samples = sum(stat['total'] for stat in all_benchmark_stats)
            total_hits = sum(stat['hits'] for stat in all_benchmark_stats)
            overall_accuracy = total_hits / total_samples if total_samples > 0 else 0
            
            overall_summary = [
                "=" * 60,
                "OVERALL SUMMARY",
                "=" * 60,
                f"Total benchmarks: {len(all_benchmark_stats)}",
                f"Total samples: {total_samples}",
                f"Total correct: {total_hits}",
                f"Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)",
                "",
                "Per-benchmark breakdown:"
            ]
            
            for stat in all_benchmark_stats:
                overall_summary.append(
                    f"  {stat['benchmark']}: {stat['hits']}/{stat['total']} = {stat['accuracy']:.3f} ({stat['accuracy']*100:.1f}%)"
                )
            
            summary_lines.extend(overall_summary)
            
            # Print overall summary to console
            print("\n" + "\n".join(overall_summary))
        
        # Final completion message
        completion_msg = [
            "",
            "=" * 60,
            "PROCESSING COMPLETE",
            "=" * 60,
            f"Processed {len(found_files)} benchmarks",
            f"Output directory: {self.output_dir}",
            f"Summary saved to: {self.output_dir / 'vmcbench_scoring_summary.txt'}"
        ]
        
        summary_lines.extend(completion_msg)
        print("\n" + "\n".join(completion_msg))
        
        # Save summary to file
        self._save_summary_to_file(summary_lines)


class LLMEquivalenceJudge:
    """Base class for LLM equivalence judges."""
    
    SYSTEM_PROMPT = "You are an assistant that compares responses for semantic or mathematical equivalence."
    
    USER_PROMPT_TEMPLATE = """
Compare the following model response to the ground truth answer. Extract the model's actual answer from its response, and then determine if the extracted answer is semantically or mathematically equivalent to the ground truth. Condition your choice of equivalence checking on the contents of the prompt, whether mathematical or semantic. Model predictions may contain extensive internal chains of thought or may enclose the answer in special containers such as "/boxed{}", <ans></ans>, etc.

Response 1 (Model): {prediction}
Response 2 (Ground Truth): {answer}

Return a JSON response with this exact format:
{{
    "equivalent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Focus on semantic meaning rather than exact text matching.
Consider responses equivalent if they convey the same core meaning, even if wording differs.
"""

    USER_PROMPT_WITH_CHOICES_TEMPLATE = """
Compare the following model response to the ground truth answer for a multiple choice question. Extract the model's actual answer from its response, and then determine if the extracted answer is semantically or mathematically equivalent to the ground truth. The model may reference choice letters (A, B, C, D) or the actual choice values.

Multiple Choice Options:
{choices_context}

Response 1 (Model): {prediction}
Response 2 (Ground Truth): {answer} (which corresponds to the value: {correct_choice_value})

Return a JSON response with this exact format:
{{
    "equivalent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Consider responses equivalent if:
- The model selects the correct choice letter ({answer})
- The model provides the correct choice value ({correct_choice_value})
- The model's answer is semantically equivalent to the correct choice value
- The model's reasoning leads to the correct conclusion even if not explicitly stated

Focus on semantic meaning rather than exact text matching.
"""
    
    def judge_equivalence(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Judge semantic equivalence between prediction and answer.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Optional dictionary mapping choice letters to their values
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        raise NotImplementedError
    

class OpenAIJudge(LLMEquivalenceJudge):
    """OpenAI-based equivalence judge."""
    
    def __init__(self, model: str = 'gpt-4o-mini', api_key: Optional[str] = None):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def judge_equivalence(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """Judge equivalence using OpenAI API."""
        try:
            # Choose appropriate template based on whether choices are available
            if choices_dict and answer in choices_dict:
                # Multiple choice context available
                choices_context = "\n".join([f"{letter}: {value}" for letter, value in choices_dict.items()])
                correct_choice_value = choices_dict[answer]
                
                user_prompt = self.USER_PROMPT_WITH_CHOICES_TEMPLATE.replace("{prediction}", str(prediction))
                user_prompt = user_prompt.replace("{answer}", str(answer))
                user_prompt = user_prompt.replace("{choices_context}", choices_context)
                user_prompt = user_prompt.replace("{correct_choice_value}", str(correct_choice_value))
            else:
                # Standard template for non-multiple choice or when choices not available
                user_prompt = self.USER_PROMPT_TEMPLATE.replace("{prediction}", str(prediction))
                user_prompt = user_prompt.replace("{answer}", str(answer))
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                equivalent = result.get('equivalent', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                if equivalent:
                    return answer, True, f"LLM judge: equivalent - {reasoning}"
                else:
                    return prediction, False, f"LLM judge: not equivalent - {reasoning}"
                    
            except json.JSONDecodeError:
                # Fallback: look for true/false in response
                if 'true' in response_text.lower():
                    return answer, True, "LLM judge: equivalent (fallback parsing)"
                else:
                    return prediction, False, "LLM judge: not equivalent (fallback parsing)"
                    
        except Exception as e:
            return prediction, False, f"OpenAI API error: {str(e)}"


class AnthropicJudge(LLMEquivalenceJudge):
    """Anthropic-based equivalence judge."""
    
    def __init__(self, model: str = 'claude-3-sonnet-20240229', api_key: Optional[str] = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
        
    def judge_equivalence(self, prediction: str, answer: str, choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """Judge equivalence using Anthropic API."""
        try:
            # Choose appropriate template based on whether choices are available
            if choices_dict and answer in choices_dict:
                # Multiple choice context available
                choices_context = "\n".join([f"{letter}: {value}" for letter, value in choices_dict.items()])
                correct_choice_value = choices_dict[answer]
                
                user_prompt = self.USER_PROMPT_WITH_CHOICES_TEMPLATE.replace("{prediction}", str(prediction))
                user_prompt = user_prompt.replace("{answer}", str(answer))
                user_prompt = user_prompt.replace("{choices_context}", choices_context)
                user_prompt = user_prompt.replace("{correct_choice_value}", str(correct_choice_value))
            else:
                # Standard template for non-multiple choice or when choices not available
                user_prompt = self.USER_PROMPT_TEMPLATE.replace("{prediction}", str(prediction))
                user_prompt = user_prompt.replace("{answer}", str(answer))
            
            response = self.client.messages.create(
                model=self.model,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=200
            )
            
            response_text = response.content[0].text
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                equivalent = result.get('equivalent', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                if equivalent:
                    return answer, True, f"LLM judge: equivalent - {reasoning}"
                else:
                    return prediction, False, f"LLM judge: not equivalent - {reasoning}"
                    
            except json.JSONDecodeError:
                # Fallback: look for true/false in response
                if 'true' in response_text.lower():
                    return answer, True, "LLM judge: equivalent (fallback parsing)"
                else:
                    return prediction, False, "LLM judge: not equivalent (fallback parsing)"
                    
        except Exception as e:
            return prediction, False, f"Anthropic API error: {str(e)}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VMCBench Standalone Scorer - 4-stage answer matching pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process VMCBench with OpenAI (full dataset)
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV VMCBench_TEST \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend openai --model gpt-4o-mini --verbose
      
  # Test with small sample (50 rows)
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend openai --model gpt-4o-mini \\
      --max-samples 50 --verbose
      
  # Resume from existing results (skip processed samples)
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend openai --model gpt-4o-mini \\
      --resume --verbose
      
  # Process with Anthropic backend
  python scripts/vmcbench_standalone_scorer.py \\
      --benchmarks VMCBench_DEV \\
      --input-dir results/full/Qwen2.5-VL-7B-Instruct \\
      --llm-backend anthropic --model claude-3-sonnet-20240229
        """
    )
    
    parser.add_argument(
        '--benchmarks', 
        nargs='+', 
        required=True,
        help='List of benchmark names to process (e.g., VMCBench_DEV VMCBench_TEST)'
    )
    parser.add_argument(
        '--input-dir', 
        required=True,
        help='Directory containing XLSX files'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for output files (defaults to input-dir)'
    )
    parser.add_argument(
        '--llm-backend',
        choices=['openai', 'anthropic'],
        default='openai',
        help='LLM backend for Stage 3 equivalence checking'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='Model name for LLM judge'
    )
    parser.add_argument(
        '--api-key',
        help='API key for LLM service (or use environment variables)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process per benchmark (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results file by skipping already processed samples'
    )
    
    args = parser.parse_args()
    
    # Initialize and run scorer
    scorer = VMCBenchScorer(
        benchmarks=args.benchmarks,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm_backend=args.llm_backend,
        model=args.model,
        api_key=args.api_key,
        verbose=args.verbose,
        max_samples=args.max_samples,
        resume=args.resume
    )
    
    scorer.process_all()


if __name__ == '__main__':
    main()