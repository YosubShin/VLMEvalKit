# Answer Extraction Logic for VLMEvalKit Benchmarks

This directory contains the complete answer extraction and scoring logic for key benchmarks in VLMEvalKit. Each module provides standalone implementations that can be used independently of the main VLMEvalKit framework.

## Overview

The answer extraction logic has been duplicated from the original VLMEvalKit codebase for the following benchmarks:

1. **atomic_dataset** - Physics problems with LaTeX boxed answer format
2. **LiveXivVQA** - Academic paper figure VQA with multiple choice
3. **OlympiadBench** - Mathematical competition problems with sophisticated equivalence checking
4. **Omni3DBench** - 3D spatial reasoning with multiple answer types
5. **VMCBench_DEV** - Vision-language multi-choice benchmark (development set)
6. **VMCBench_TEST** - Vision-language multi-choice benchmark (test set)

## File Structure

```
answer_extraction_logic/
├── README.md                 # This file
├── atomic_dataset.py         # Physics problems evaluation
├── LiveXivVQA.py            # Academic VQA evaluation  
├── OlympiadBench.py         # Mathematical competition problems
├── Omni3DBench.py           # 3D spatial reasoning
├── VMCBench.py              # Shared VMC evaluation logic
├── VMCBench_DEV.py          # VMC development set interface
└── VMCBench_TEST.py         # VMC test set interface
```

## Benchmark Details

### 1. atomic_dataset

**Type**: VQA (Visual Question Answering) - Physics Problems  
**Answer Format**: LaTeX boxed format (`\boxed{answer}`)  
**Evaluation Method**: LLM-based equivalence checking with SymPy fallback

**Raw Response Storage**:
- ✅ **Complete responses saved**: Yes
- **Standard location**: `prediction` field in `{model_name}_atomic_dataset.xlsx`
- **Enhanced storage**: `--save-detailed-eval` → `{model_name}_atomic_dataset_raw_responses.{format}`
- **Judge responses**: `--save-judge-responses` → `{model_name}_atomic_dataset_{judge}_judge_responses.{format}`
- **Enhanced fields**: `raw_model_response`, `processed_model_answer`, `judge_response_1/2`

**Key Features**:
- Extracts answers from `\boxed{}` LaTeX format
- Uses sophisticated mathematical equivalence checking
- Supports both numerical and text-based physics answers
- Includes judge response capturing for analysis

**Main Functions**:
- `extract_answer_atomic_dataset(prediction_text)` - Extract answer from prediction
- `score_atomic_dataset(model, predicted_answer, ground_truth_answer)` - Score with equivalence checking
- `build_physic_prompt(line)` - Build physics expert prompt
- `PHYSIC_auxeval(model, line)` - Complete evaluation pipeline

### 2. LiveXivVQA

**Type**: MCQ (Multiple Choice Questions) - Academic Paper Figures  
**Answer Format**: Single letter (A, B, C, D, etc.)  
**Evaluation Method**: Option letter matching with GPT fallback

**Raw Response Storage**:
- ✅ **Complete responses saved**: Yes
- **Standard location**: `prediction` field in `{model_name}_LiveXivVQA.xlsx`
- **Enhanced storage**: Not supported for `--save-detailed-eval` flag
- **Judge responses**: Not supported for `--save-judge-responses` flag
- **Available fields**: `prediction` (raw response), `answer` (ground truth), `category`

**Key Features**:
- Parses multiple choice responses with robust pattern matching
- Handles both explicit option selection and content-based matching
- Custom prompt building for academic figure questions
- GPT-based answer matching for ambiguous cases

**Main Functions**:
- `extract_answer_livexivvqa(prediction_text)` - Extract option letter
- `score_livexivvqa(predicted_answer, ground_truth_answer)` - Score option match
- `parse_multi_choice_response(response, all_choices, index2ans)` - Parse MCQ response
- `build_livexiv_prompt(line)` - Build LiveXiv-specific prompt

### 3. OlympiadBench

**Type**: Mathematical Competition Problems  
**Answer Format**: Various (numerical, expressions, equations, intervals)  
**Evaluation Method**: Advanced mathematical equivalence with SymPy

**Raw Response Storage**:
- ✅ **Complete responses saved**: Yes
- **Standard location**: `prediction` field in `{model_name}_OlympiadBench.xlsx`
- **Enhanced storage**: `--save-detailed-eval` → `{model_name}_OlympiadBench_raw_responses.{format}`
- **Judge responses**: Not supported (uses SymPy, no LLM judges)
- **Enhanced fields**: `raw_model_response`, `processed_model_answer` (after regex extraction)

**Key Features**:
- Sophisticated mathematical expression parsing
- Multi-type answer support (numerical, expressions, equations, intervals)
- Plus-minus sign handling (`\pm` expansion)
- Timeout protection for complex computations
- Multilingual support (English/Chinese)

**Main Functions**:
- `extract_answer_olympiadbench(prediction_text)` - Extract mathematical answer
- `score_olympiadbench(predicted_answer, ground_truth_answer)` - Mathematical equivalence scoring
- `MathJudger` class - Complete mathematical equivalence checker
- `get_answer_type_text(answer_type, is_chinese, multiple_answer)` - Answer type prompting

### 4. Omni3DBench

**Type**: 3D Spatial Reasoning  
**Answer Format**: Mixed (yes/no, numbers, multiple choice)  
**Evaluation Method**: Type-specific scoring with Mean Relative Accuracy (MRA)

**Raw Response Storage**:
- ✅ **Complete responses saved**: Yes
- **Standard location**: `prediction` field in `{model_name}_Omni3DBench.xlsx`
- **Enhanced storage**: Not supported for `--save-detailed-eval` flag
- **Judge responses**: Not supported (uses rule-based scoring, no LLM judges)
- **Available fields**: `prediction` (raw response), `answer` (ground truth), `answer_type`

**Key Features**:
- Answer extraction from `<ans></ans>` tags
- Multiple answer type support (str, int, float)
- Mean Relative Accuracy (MRA) for numerical answers
- Yes/no question handling with variations (true/false)
- Counting task exact matching

**Main Functions**:
- `extract_answer_omni3dbench(prediction_text)` - Extract from tags or patterns
- `score_omni3dbench(predicted_answer, ground_truth_answer, answer_type)` - Type-specific scoring
- `Omni3DBench_acc(data)` - Complete accuracy calculation with MRA
- `build_omni3d_prompt(question)` - Build Omni3D prompt

### 5. VMCBench (DEV & TEST)

**Type**: MCQ - Vision-Language Multi-Choice Benchmark  
**Answer Format**: Single letter (A, B, C, D, etc.)  
**Evaluation Method**: Pattern matching only (NO LLM judge) with category breakdown

**Raw Response Storage**:
- ✅ **Complete responses saved**: Yes
- **Standard location**: `prediction` field in `{model_name}_VMCBench_DEV.xlsx` / `{model_name}_VMCBench_TEST.xlsx`
- **Enhanced storage**: `--save-detailed-eval` → `{model_name}_VMCBench_{DEV|TEST}_raw_responses.{format}`
- **Judge responses**: Not supported (uses pattern matching only, no LLM judges)
- **Enhanced fields**: `raw_model_response`, `available_choices`, `choice_options`, `category`

**Key Features**:
- Shared logic between DEV and TEST versions
- Category-wise accuracy reporting
- Robust multiple choice parsing
- Content-based fallback matching
- Detailed accuracy breakdowns by dataset categories

**Main Functions**:
- `extract_answer_vmcbench(prediction_text, choices)` - Extract MCQ answer
- `score_vmcbench(predicted_answer, ground_truth_answer)` - Score MCQ match
- `report_vmc_acc(data)` - Category-wise accuracy reporting
- `get_mc_score(row)` - Single item scoring

## Usage Examples

### Basic Answer Extraction

```python
# atomic_dataset
from answer_extraction_logic.atomic_dataset import extract_answer_atomic_dataset
answer = extract_answer_atomic_dataset("The answer is \\boxed{42}")

# LiveXivVQA  
from answer_extraction_logic.LiveXivVQA import extract_answer_livexivvqa
answer = extract_answer_livexivvqa("The correct answer is B")

# OlympiadBench
from answer_extraction_logic.OlympiadBench import extract_answer_olympiadbench
answer = extract_answer_olympiadbench("So the final answer is 3.14159")

# Omni3DBench
from answer_extraction_logic.Omni3DBench import extract_answer_omni3dbench
answer = extract_answer_omni3dbench("The answer is <ans>yes</ans>")

# VMCBench
from answer_extraction_logic.VMCBench import extract_answer_vmcbench
choices = {"A": "cat", "B": "dog", "C": "bird"}
answer = extract_answer_vmcbench("I choose option A", choices)
```

### Complete Scoring

```python
# atomic_dataset (requires model for equivalence checking)
from answer_extraction_logic.atomic_dataset import score_atomic_dataset
result = score_atomic_dataset(model, "42", "6*7", capture_judge_responses=True)

# LiveXivVQA
from answer_extraction_logic.LiveXivVQA import score_livexivvqa
result = score_livexivvqa("B", "B")

# OlympiadBench
from answer_extraction_logic.OlympiadBench import score_olympiadbench
result = score_olympiadbench("3.14159", "\\pi", precision=1e-3)

# Omni3DBench
from answer_extraction_logic.Omni3DBench import score_omni3dbench
result = score_omni3dbench("yes", "yes", "str")

# VMCBench
from answer_extraction_logic.VMCBench import score_vmcbench
result = score_vmcbench("A", "A")
```

### Batch Processing

```python
# Process multiple items
from answer_extraction_logic.LiveXivVQA import process_livexiv_batch
from answer_extraction_logic.Omni3DBench import process_omni3d_batch
from answer_extraction_logic.VMCBench import process_vmc_batch

# Each returns a DataFrame with evaluation results
livexiv_results = process_livexiv_batch(livexiv_data, model)
omni3d_results = process_omni3d_batch(omni3d_data)
vmc_results = process_vmc_batch(vmc_data)
```

## Dependencies

### Core Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `re` - Regular expressions for pattern matching

### Specialized Dependencies

**atomic_dataset & OlympiadBench**:
- `sympy` - Mathematical expression parsing and equivalence
- `timeout_decorator` - Computation timeout protection
- `antlr4` - LaTeX parsing support

**OlympiadBench specific**:
- `decimal` - High precision decimal arithmetic
- `fractions` - Fraction arithmetic

## Error Handling

All modules include comprehensive error handling:

- **Timeout Protection**: Mathematical computations are protected with timeouts
- **Type Conversion**: Robust handling of type conversions with fallbacks
- **Missing Data**: Graceful handling of missing predictions or ground truth
- **Malformed Input**: Pattern matching with multiple fallback strategies

## Evaluation Methodologies

### Mathematical Equivalence (atomic_dataset, OlympiadBench)
1. **Direct String Match**: Exact string comparison
2. **SymPy Equivalence**: Mathematical expression equivalence
3. **LLM Judge**: Fallback to language model for complex cases
4. **Numerical Tolerance**: Precision-based numerical comparison

### Multiple Choice (LiveXivVQA, VMCBench)
1. **Pattern Matching**: Extract option letters from various formats
2. **Content Matching**: Match based on choice content
3. **GPT Fallback**: Use language model for ambiguous cases
4. **Random Fallback**: Last resort random selection

### Multi-Type (Omni3DBench)
1. **Type-Specific Logic**: Different evaluation per answer type
2. **Tag Extraction**: Parse structured answer tags
3. **MRA Scoring**: Mean Relative Accuracy for numerical answers
4. **Binary Classification**: Yes/no question handling

## Detailed Evaluation Pipelines

This section provides step-by-step accounts of the evaluation pipeline for each benchmark, from raw LLM prediction to final score and judgment.

### Atomic Evaluation Pipeline

**Input**: Raw LLM prediction text (physics problem response)
**Output**: Binary score (0 or 1) + detailed equivalence log

```
Step 1: Answer Extraction
├─ Input: "The energy difference is E₂ - E₁ = -3.4 - (-13.6) = 10.2 eV. \\boxed{10.2\\ \\text{eV}}"
├─ Function: extract_final_answer_allform()
├─ Process: 
│  ├─ Find all \\boxed{...} patterns using regex
│  ├─ Extract nested content handling braces
│  ├─ If no \\boxed found → extract $...$ patterns from last line
│  └─ Return: ["10.2\\ \\text{eV}"]
└─ Branch A: Content found → Continue
   Branch B: No content found → Return []

Step 2: Direct String Matching
├─ Input: extracted=["10.2\\ \\text{eV}"], ground_truth="10.2 eV"
├─ Function: Check if ground_truth in extracted answers
├─ Process: exact string comparison after strip()
└─ Branch A: Match found → Return True (score=1)
   Branch B: No match → Continue to Step 3

Step 3: Mathematical Equivalence Check
├─ Input: pred="10.2\\ \\text{eV}", gt="10.2 eV"
├─ Function: is_equiv(model, pred, gt)
├─ Process:
│  ├─ Check for \\text presence → Use LLM judge
│  ├─ If no \\text → SymPy mathematical comparison
│  ├─ Preprocess LaTeX: remove decorative symbols
│  ├─ Parse with SymPy: sympify(parse_latex(expr))
│  ├─ Check equivalence: simplify(expr1 - expr2) == 0
│  └─ Timeout protection: @timeout_decorator.timeout(30)
├─ Branch A: SymPy equivalence found → Return True
├─ Branch B: SymPy fails → Use LLM judge
│  ├─ Judge prompt: "Compare expressions for equivalence"
│  ├─ LLM response: "True" or "False"
│  └─ Parse: "true" in response.lower()
└─ Branch C: All methods fail → Return False

Step 4: Final Scoring
├─ Input: equivalence_result=True/False
├─ Output: {"correct": bool, "score": 1.0/0.0, "explanation": str}
├─ Log: Complete trace of extraction and equivalence checking
└─ Judge responses: Optionally captured for analysis
```

### LiveXivVQA Evaluation Pipeline

**Input**: Raw LLM prediction text (academic figure VQA response)
**Output**: Binary score (0 or 1) + option letter

```
Step 1: Direct Pattern Matching
├─ Input: "Looking at the graph, the trend clearly shows... The answer is B."
├─ Function: can_infer_option()
├─ Process: Test multiple regex patterns
│  ├─ Pattern 1: "(?:answer|choice|option).*([A-Z])" → Finds "B"
│  ├─ Pattern 2: "([A-Z]).*(?:correct|right)" → Backup pattern
│  ├─ Pattern 3: "\\b([A-Z])\\b" → Standalone letters
│  └─ Validate: Check if found letter in available choices
└─ Branch A: Valid option found → Return "B"
   Branch B: No pattern match → Continue to Step 2

Step 2: Content-Based Matching
├─ Input: prediction_text, choices={"A": "Linear increase", "B": "Exponential decay"}
├─ Function: Content substring matching
├─ Process:
│  ├─ For each choice: check if choice_text in prediction.lower()
│  ├─ Reverse check: check if prediction in choice_text.lower()
│  └─ Return first match found
└─ Branch A: Content match found → Return option letter
   Branch B: No content match → Continue to Step 3

Step 3: Random Fallback
├─ Input: All extraction methods failed
├─ Process: random.choice(available_options + ["Z"])
└─ Output: Random option letter + failure log

Step 4: Final Scoring
├─ Input: predicted_option="B", ground_truth="B"
├─ Function: Direct comparison (predicted == ground_truth)
├─ Output: {"correct": True, "score": 1.0, "extracted_answer": "B"}
└─ Note: VMCBench uses NO LLM judge, only pattern matching + random fallback
```

### OlympiadBench Evaluation Pipeline

**Input**: Raw LLM prediction text (mathematical competition response)
**Output**: Binary score (0 or 1) + mathematical equivalence details

```
Step 1: Answer Extraction
├─ Input: "The solution involves... So the final answer is x² + 2x + 1"
├─ Function: extract_answer(is_chinese=False, model_output, is_deepseek=False)
├─ Process:
│  ├─ Check for "So the final answer is (.*)" pattern
│  ├─ If found: extract match.strip()
│  ├─ If not found: return entire model_output
│  └─ Handle special model formats (DeepSeek, Chinese)
└─ Output: "x² + 2x + 1"

Step 2: Mathematical Preprocessing
├─ Input: pred="x² + 2x + 1", gt="(x+1)²"
├─ Function: MathJudger.preprocess()
├─ Process:
│  ├─ Extract boxed content if present
│  ├─ Remove special symbols: \\left, \\right, %, etc.
│  ├─ Clean decorative LaTeX: \\mathrm{}, \\mathbf{}
│  ├─ Strip unwanted characters: $,.:;^_=+`!@#$%^&*~
│  └─ Handle \\in statements for set notation
└─ Output: cleaned expressions

Step 3: Equivalence Testing (Multiple Methods)
├─ Function: MathJudger.judge()
├─ Branch A: Direct String Match
│  ├─ Compare cleaned expressions exactly
│  └─ If equal → Return True
├─ Branch B: Interval Comparison (if both are intervals)
│  ├─ Check bracket types: () vs []
│  ├─ Split by commas and \\cup
│  ├─ Compare endpoints using expression_equal()
│  └─ If equivalent → Return True
├─ Branch C: Numerical Comparison
│  ├─ Convert to float() if possible
│  ├─ Test with percentage variations: [val/100, val, val*100]
│  ├─ Check: abs(pred - gt) ≤ precision
│  └─ If within tolerance → Return True
├─ Branch D: Expression Equivalence
│  ├─ Parse with SymPy: sympify(parse_latex(expr))
│  ├─ Standardize: simplify(expand(trigsimp(expr)))
│  ├─ Test: simplify(expr1 - expr2) == 0
│  ├─ Handle variables vs constants separately
│  ├─ Timeout protection: 30 seconds
│  └─ If equivalent → Return True
└─ Branch E: Equation Equivalence
   ├─ Split by "=" and move to standard form
   ├─ Create: Eq(lhs, rhs) → simplify(lhs - rhs)
   ├─ Test quotients: expr1/expr2 and expr2/expr1
   ├─ Check if quotient is non-zero integer
   └─ If equivalent → Return True

Step 4: Error Handling & Fallbacks
├─ Timeout exceeded → Return False + timeout error
├─ SymPy parsing error → Return False + parse error  
├─ Computation overflow → Return False + overflow error
├─ Unknown symbols → Return False + symbol error
└─ All methods tested → Return final result

Step 5: Final Scoring
├─ Input: equivalence_result=True/False + detailed logs
├─ Output: {"correct": bool, "score": 1.0/0.0, "explanation": "method_used"}
├─ Additional: Complete error trace and method attempted
└─ Performance: Timing information for complex expressions
```

### Omni3DBench Evaluation Pipeline

**Input**: Raw LLM prediction text (3D spatial reasoning response)
**Output**: Type-specific score + detailed breakdown

```
Step 1: Answer Extraction
├─ Input: "The object is closer to the camera. <ans>yes</ans>"
├─ Function: extract_answer()
├─ Process:
│  ├─ Look for <ans>...</ans> tags
│  ├─ Extract content between tags
│  ├─ If no tags found → return full prediction
│  └─ Strip whitespace from extracted content
├─ Branch A: Tags found → Return "yes"
└─ Branch B: No tags → Continue with pattern matching

Step 2: Fallback Pattern Extraction (if no tags)
├─ Input: "The answer is yes, the object is closer"
├─ Process:
│  ├─ Yes/No detection: check for "yes"/"no", "true"/"false"
│  ├─ Number extraction: regex r'-?\\d+\\.?\\d*'
│  ├─ Priority: yes/no > numbers > full text
│  └─ Return best match found
└─ Output: Extracted answer

Step 3: Type-Specific Scoring
├─ Input: answer="yes", ground_truth="yes", answer_type="str"
├─ Function: score_omni3dbench()
├─ Branch A: Integer Type (answer_type="int")
│  ├─ Convert: int(predicted), int(ground_truth)
│  ├─ Compare: exact equality check
│  └─ Score: 1.0 if equal, 0.0 if not
├─ Branch B: String Type (answer_type="str")
│  ├─ Sub-branch B1: Yes/No Questions
│  │  ├─ Normalize: lower() and strip()
│  │  ├─ Direct match: gt_lower in pred_lower
│  │  ├─ Variation match: "yes"→"true", "no"→"false"
│  │  └─ Score: 1.0 if match, 0.0 if not
│  └─ Sub-branch B2: Multiple Choice
│     ├─ Exact string comparison: gt_lower == pred_lower
│     └─ Score: 1.0 if equal, 0.0 if not
└─ Branch C: Float Type (answer_type="float")
   ├─ Convert: float(predicted), float(ground_truth)
   ├─ MRA Calculation:
   │  ├─ Thresholds: [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
   │  ├─ For each threshold: check if |gt-pred|/gt < threshold
   │  ├─ Threshold score: 1.0 if within, 0.0 if not
   │  └─ Final MRA: average of all threshold scores
   └─ Score: MRA value (0.0 to 1.0)

Step 4: Batch Processing (Omni3DBench_acc)
├─ Input: DataFrame with multiple predictions
├─ Process: Aggregate by answer type
│  ├─ Yes/No: count correct / total yes/no questions
│  ├─ Integer: count exact matches / total integer questions  
│  ├─ Multiple choice: count exact matches / total MC questions
│  ├─ Float: calculate average MRA across all float questions
│  └─ Handle division by zero for empty categories
└─ Output: Category-wise accuracy DataFrame

Step 5: Final Results
├─ Individual: {"correct": bool, "score": float, "answer_type": str}
├─ Batch: {"Yes/No Accuracy": float, "Numeric (count) Accuracy": float, ...}
├─ Additional: Threshold breakdown for MRA scores
└─ Error logs: Type conversion failures, missing data
```

### VMCBench Evaluation Pipeline

**Input**: Raw LLM prediction text (multi-category MCQ response)
**Output**: Binary score + category-wise accuracy breakdown

```
Step 1: Choice Building
├─ Input: item with A, B, C, D columns + prediction
├─ Function: build_choices()
├─ Process:
│  ├─ Scan columns A-Z for non-null values
│  ├─ Build mapping: {"A": "choice_text_A", "B": "choice_text_B", ...}
│  └─ Create index2ans for content matching
└─ Output: choices dict + all_choices list

Step 2: Multi-Choice Response Parsing
├─ Input: "Based on the image... The correct answer is (B)"
├─ Function: parse_multi_choice_response()
├─ Process:
│  ├─ Clean response: strip punctuation
│  ├─ Add spaces: " " + response + " " (avoid partial matches)
│  ├─ Branch A: Bracketed Options
│  │  ├─ Search: "(A)", "(B)", "A.", "B." patterns
│  │  ├─ Collect all candidates found
│  │  └─ If found → Set ans_with_brack=True
│  ├─ Branch B: Standalone Letters (if Branch A empty)
│  │  ├─ Search: " A ", " B " patterns
│  │  └─ Collect letter candidates
│  ├─ Branch C: Content Matching (if A & B empty)
│  │  ├─ Check: choice_text.lower() in response.lower()
│  │  ├─ For each choice content found
│  │  └─ Add corresponding letter to candidates
│  └─ Branch D: Random Selection (if all empty)
│     └─ Return: random.choice(all_choices)
├─ Disambiguation (if multiple candidates):
│  ├─ Find position of each candidate in response
│  ├─ Use rightmost (last) occurrence
│  └─ Return: candidate with max position index
└─ Output: Single option letter

Step 3: Individual Scoring
├─ Input: predicted="B", ground_truth="B"
├─ Function: get_mc_score()
├─ Process: Direct comparison (predicted == ground_truth)
└─ Output: 1 if equal, 0 if not

Step 4: Category Aggregation
├─ Input: DataFrame with category column + hit scores
├─ Function: report_vmc_acc()
├─ Process:
│  ├─ Group by category: calculate mean(hit) for each
│  ├─ Dataset groupings:
│  │  ├─ General: SEEDBench, MMStar, A-OKVQA, VizWiz, MMVet, VQAv2, OKVQA
│  │  ├─ Reasoning: MMMU, MathVista, ScienceQA, RealWorldQA, GQA, MathVision
│  │  ├─ OCR: TextVQA, OCRVQA  
│  │  └─ Doc & Chart: AI2D, ChartQA, DocVQA, InfoVQA, TableVQABench
│  ├─ Calculate group averages: mean across datasets in each group
│  ├─ Overall accuracy: mean across all items
│  └─ Convert to percentages: * 100 and round to 2 decimals
└─ Output: Comprehensive accuracy DataFrame

Step 5: Error Handling & Fallbacks
├─ Missing choices → Return 0 score
├─ Empty prediction → Random selection + log
├─ Invalid characters → Pattern matching with fallbacks
└─ Final fallback → Random choice + detailed error log
   Note: VMCBench does NOT use LLM models for answer extraction

Step 6: Final Results
├─ Individual: {"hit": 0/1, "log": "extraction_details"}
├─ Batch: DataFrame with Overall/General/Reasoning/OCR/Doc accuracies
├─ Category detail: Accuracy for each specific dataset
└─ Performance: Detailed logs for debugging failed extractions
```

## Performance Considerations

- **Caching**: Results can be cached to avoid re-computation
- **Batch Processing**: All modules support batch evaluation
- **Timeout Limits**: Mathematical computations have configurable timeouts
- **Memory Efficiency**: Streaming evaluation for large datasets

## Contributing

When adding new benchmarks:

1. Follow the established pattern of `extract_answer_[benchmark](prediction_text)` and `score_[benchmark](...)`
2. Include comprehensive docstrings with parameter and return type documentation
3. Add error handling for common failure cases
4. Include usage examples in the module docstring
5. Update this README with the new benchmark details

## Raw Response Storage Summary

All benchmarks save complete, unedited raw LLM responses in the **`prediction`** field of the main result files. Enhanced storage options are available for select benchmarks:

### Standard Storage (All Benchmarks)
- **Location**: `./outputs/{model_name}/T{date}_G{commit}/`
- **File format**: `{model_name}_{dataset_name}.xlsx`
- **Key field**: `prediction` - Contains complete, unprocessed raw LLM response
- **Usage**: Always available for all evaluations

### Enhanced Storage (Select Benchmarks)
- **Supported**: `atomic_dataset`, `OlympiadBench`, `VMCBench_DEV`, `VMCBench_TEST`
- **Activation**: `--save-detailed-eval` flag
- **File format**: `{model_name}_{dataset_name}_raw_responses.{format}`
- **Enhanced fields**: `raw_model_response`, `processed_model_answer`, `available_choices`, etc.
- **Format options**: JSON, CSV, XLSX (via `--response-format`)

### Judge Response Storage (Physics Benchmarks Only)
- **Supported**: `atomic_dataset` and other Yale Physics datasets
- **Activation**: `--save-judge-responses` flag
- **File format**: `{model_name}_{dataset_name}_{judge_model}_judge_responses.{format}`
- **Judge fields**: `judge_response_1/2` with complete LLM judge reasoning
- **Usage**: For analyzing LLM-based equivalence checking

### File Organization Structure
```
./outputs/{model_name}/T{date}_G{commit}/
├── {model_name}_{dataset}.xlsx                    # Main results (prediction field)
├── {model_name}_{dataset}_raw_responses.json      # Enhanced raw responses
├── {model_name}_{dataset}_{judge}_judge_responses.json  # Judge interactions
└── {model_name}_{dataset}_acc.csv                 # Evaluation metrics
```

### Usage Examples
```bash
# Standard storage (prediction field in main .xlsx)
python run.py --model GPT4o --data LiveXivVQA

# Enhanced raw response storage (supported benchmarks)
python run.py --model GPT4o --data VMCBench_DEV --save-detailed-eval --response-format json

# Judge response storage (physics benchmarks only)
python run.py --model GPT4o --data atomic_dataset --save-judge-responses

# Both enhanced and judge responses
python run.py --model GPT4o --data atomic_dataset --save-detailed-eval --save-judge-responses
```

## References

- **VSI-Bench**: Mean Relative Accuracy methodology (https://arxiv.org/abs/2412.14171)
- **VLMEvalKit**: Original evaluation framework
- **SymPy**: Mathematical computation library
- **ANTLR**: Parser generator for LaTeX expressions