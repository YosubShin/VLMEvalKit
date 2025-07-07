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
        for i in range(9):  # Support up to 9 choices (A-I)
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
        Stage 1: Simple matching strategies.
        
        Includes exact matches, case-insensitive matches, whitespace handling,
        single character extraction, True/False matching, and number extraction.
        
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
        
        # Strategy 4: Single character extraction (for MCQ)
        single_char_pattern = r'\b([A-I])\b'
        pred_chars = re.findall(single_char_pattern, pred_clean.upper())
        if pred_chars and pred_chars[0] == answer_clean.upper():
            return answer_clean, True, f"Single character match: {pred_chars[0]}"
        
        # Strategy 5: True/False pattern matching
        true_patterns = ['true', 'yes', 'correct', 'right']
        false_patterns = ['false', 'no', 'incorrect', 'wrong']
        
        pred_lower = pred_clean.lower()
        answer_lower = answer_clean.lower()
        
        if answer_lower in true_patterns:
            for pattern in true_patterns:
                if pattern in pred_lower:
                    return answer_clean, True, f"True/Yes pattern match: {pattern}"
        
        if answer_lower in false_patterns:
            for pattern in false_patterns:
                if pattern in pred_lower:
                    return answer_clean, True, f"False/No pattern match: {pattern}"
        
        # Strategy 6: Number extraction
        pred_numbers = re.findall(r'-?\d+\.?\d*', pred_clean)
        answer_numbers = re.findall(r'-?\d+\.?\d*', answer_clean)
        
        if pred_numbers and answer_numbers:
            try:
                pred_num = float(pred_numbers[0])
                answer_num = float(answer_numbers[0])
                if abs(pred_num - answer_num) < 1e-6:
                    return answer_clean, True, f"Numerical match: {pred_num} ≈ {answer_num}"
            except ValueError:
                pass
        
        # Strategy 7: Substring matching (prediction contains answer or vice versa)
        if answer_clean.lower() in pred_clean.lower():
            return answer_clean, True, "Answer substring in prediction"
        
        if pred_clean.lower() in answer_clean.lower() and len(pred_clean) > 2:
            return answer_clean, True, "Prediction substring in answer"
        
        return pred_clean, False, "No simple match found"
    
    def stage2_complex_match(self, prediction: str, answer: str, 
                           choices_dict: Optional[Dict[str, str]] = None) -> Tuple[str, bool, str]:
        """
        Stage 2: Complex matching strategies from all benchmarks.
        
        Combines sophisticated extraction logic from VMCBench, Omni3DBench,
        OlympiadBench, and atomic_dataset.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            choices_dict: Dictionary of multiple choice options (optional)
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        strategies = [
            ("VMCBench MCQ", lambda: self._vmcbench_mcq_parsing(prediction, choices_dict)),
            ("Omni3D Tags", lambda: self._omni3d_extraction(prediction)),
            ("Olympiad Math", lambda: self._olympiad_math_extraction(prediction)),
            ("Atomic Physics", lambda: self._atomic_physics_extraction(prediction)),
            ("SymPy Equivalence", lambda: self._sympy_equivalence_check(prediction, answer)),
            ("LaTeX Parsing", lambda: self._latex_expression_parsing(prediction)),
            ("Structured Tags", lambda: self._structured_tag_extraction(prediction)),
            ("Pattern Extraction", lambda: self._pattern_based_extraction(prediction))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func()
                if result and result != prediction:  # Found something useful
                    # Check if result matches answer
                    if result.lower().strip() == answer.lower().strip():
                        return answer, True, f"Success with {strategy_name}: {result}"
                    # Even if not exact match, return the extracted result
                    elif len(result.strip()) > 0:
                        return result, True, f"Extracted with {strategy_name}: {result}"
            except Exception as e:
                continue  # Try next strategy
        
        return prediction, False, "All complex strategies failed"
    
    def stage3_llm_judge(self, prediction: str, answer: str) -> Tuple[str, bool, str]:
        """
        Stage 3: LLM equivalence checking.
        
        Uses the initialized LLM judge to determine semantic equivalence.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        return self.llm_judge.judge_equivalence(prediction, answer)
    
    # Helper methods for Stage 2 complex matching
    def _vmcbench_mcq_parsing(self, prediction: str, choices_dict: Optional[Dict[str, str]]) -> Optional[str]:
        """VMCBench multiple choice parsing logic."""
        if not choices_dict:
            return None
            
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
        
        # Return first candidate or None
        return candidates[0] if candidates else None
    
    def _omni3d_extraction(self, prediction: str) -> Optional[str]:
        """Omni3DBench structured extraction using <ans></ans> tags."""
        # Tag-based extraction
        if '<ans>' in prediction and '</ans>' in prediction:
            try:
                start = prediction.find('<ans>') + 5
                end = prediction.find('</ans>')
                return prediction[start:end].strip()
            except:
                pass
        
        # Fallback pattern extraction
        pred_lower = prediction.lower()
        
        # Yes/no detection
        yes_patterns = ['yes', 'true', 'correct']
        no_patterns = ['no', 'false', 'incorrect']
        
        for pattern in yes_patterns:
            if pattern in pred_lower:
                return 'yes'
        
        for pattern in no_patterns:
            if pattern in pred_lower:
                return 'no'
        
        # Number extraction
        numbers = re.findall(r'-?\d+\.?\d*', prediction)
        if numbers:
            return numbers[0]
        
        return None
    
    def _olympiad_math_extraction(self, prediction: str) -> Optional[str]:
        """OlympiadBench mathematical extraction with boxed content."""
        # Extract \\boxed{} content
        boxed_pattern = r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return matches[-1].strip()  # Return last match
        
        # Extract $...$ content
        dollar_pattern = r'\$([^$]+)\$'
        dollar_matches = re.findall(dollar_pattern, prediction)
        if dollar_matches:
            return dollar_matches[-1].strip()
        
        # "So the final answer is..." pattern
        final_answer_patterns = [
            r'So the final answer is\s*([^.]+)',
            r'Therefore,?\s*the answer is\s*([^.]+)',
            r'The answer is\s*([^.]+)'
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _atomic_physics_extraction(self, prediction: str) -> Optional[str]:
        """Atomic dataset physics extraction with complex boxed content."""
        # Complex boxed extraction with nested brackets
        def extract_boxed_content(text):
            boxed_matches = re.finditer(r'\\boxed{', text)
            results = []
            
            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1
                
                while stack > 0 and end_index < len(text):
                    if text[end_index] == '{':
                        stack += 1
                    elif text[end_index] == '}':
                        stack -= 1
                    end_index += 1
                
                if stack == 0:
                    content = text[start_index:end_index - 1]
                    results.append(content)
            
            return results
        
        boxed_content = extract_boxed_content(prediction)
        if boxed_content:
            return boxed_content[-1].strip()
        
        return None
    
    def _sympy_equivalence_check(self, prediction: str, answer: str) -> Optional[str]:
        """SymPy mathematical equivalence checking."""
        if not SYMPY_AVAILABLE:
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
    
    def _latex_expression_parsing(self, prediction: str) -> Optional[str]:
        """LaTeX expression parsing and normalization."""
        if not SYMPY_AVAILABLE:
            return None
            
        try:
            # Try to parse LaTeX with SymPy
            expr = parse_latex(prediction)
            return str(expr)
        except Exception:
            pass
        
        return None
    
    def _structured_tag_extraction(self, prediction: str) -> Optional[str]:
        """Generalized structured tag extraction."""
        # Common tag patterns
        tag_patterns = [
            r'<answer>(.*?)</answer>',
            r'<result>(.*?)</result>',
            r'<final>(.*?)</final>',
            r'\[ANSWER\](.*?)\[/ANSWER\]',
            r'Answer:\s*([^\n]+)',
            r'Final Answer:\s*([^\n]+)'
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _pattern_based_extraction(self, prediction: str) -> Optional[str]:
        """Pattern-based extraction with common answer formats."""
        # Common answer patterns
        patterns = [
            r'(?:^|\s)([A-I])(?:\s|$|\.|,)',  # Single letter choices
            r'(?:is|are)\s+([A-I])(?:\s|$|\.|,)',  # "The answer is A"
            r'(?:option|choice)\s+([A-I])(?:\s|$|\.|,)',  # "Option A"
            r'([A-I])\s*(?:is|are)\s*(?:correct|right)',  # "A is correct"
            r'(\d+(?:\.\d+)?)',  # Numbers
            r'(true|false|yes|no)'  # Boolean answers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, prediction, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
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
                    stage3_result, stage3_success, stage3_error = self.stage3_llm_judge(prediction, answer)
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
            row_result['hit'] = 1 if final_answer == answer else 0
            row_result['stage_errors'] = " | ".join(errors)
            
            results.append(row_result)
        
        # Combine original data with results
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df], axis=1)
    
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
        print(f"Correct answers (hits): {hits}")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nStage success counts:")
        for stage, count in stage_stats.items():
            percentage = count / total_rows * 100 if total_rows > 0 else 0
            print(f"  {stage}: {count} ({percentage:.1f}%)")
    
    def process_all(self):
        """Process all benchmarks."""
        found_files = self.find_benchmark_files()
        
        if not found_files:
            self.logger.error("No benchmark files found!")
            return
        
        for benchmark_name, file_path in found_files.items():
            try:
                self.process_benchmark(benchmark_name, file_path)
            except Exception as e:
                self.logger.error(f"Error processing {benchmark_name}: {e}")
                continue
        
        print(f"\n=== Processing Complete ===")
        print(f"Processed {len(found_files)} benchmarks")
        print(f"Output directory: {self.output_dir}")


class LLMEquivalenceJudge:
    """Base class for LLM equivalence judges."""
    
    SYSTEM_PROMPT = "You are an assistant that compares responses for semantic equivalence."
    
    USER_PROMPT_TEMPLATE = """
Compare the following two responses and determine if they are semantically equivalent.

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
    
    def judge_equivalence(self, prediction: str, answer: str) -> Tuple[str, bool, str]:
        """
        Judge semantic equivalence between prediction and answer.
        
        Args:
            prediction: Model's prediction
            answer: Ground truth answer
            
        Returns:
            Tuple of (extracted_answer, success, error_message)
        """
        raise NotImplementedError
    

class OpenAIJudge(LLMEquivalenceJudge):
    """OpenAI-based equivalence judge."""
    
    def __init__(self, model: str = 'gpt-4o-mini', api_key: Optional[str] = None):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def judge_equivalence(self, prediction: str, answer: str) -> Tuple[str, bool, str]:
        """Judge equivalence using OpenAI API."""
        try:
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                prediction=prediction, answer=answer
            )
            
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
        
    def judge_equivalence(self, prediction: str, answer: str) -> Tuple[str, bool, str]:
        """Judge equivalence using Anthropic API."""
        try:
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                prediction=prediction, answer=answer
            )
            
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