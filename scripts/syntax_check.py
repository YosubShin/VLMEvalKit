#!/usr/bin/env python3
"""
Syntax validation for VLLM batch processing implementation.

This script validates the syntax and basic structure of our implementation
without requiring external dependencies.
"""

import ast
import sys
from pathlib import Path


def check_python_syntax(file_path):
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_method_exists(file_path, method_names):
    """Check if methods exist in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find all method/function definitions
        found_methods = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_methods.append(node.name)
        
        # Check for required methods
        missing_methods = []
        for method in method_names:
            if method not in found_methods:
                missing_methods.append(method)
        
        return len(missing_methods) == 0, missing_methods, found_methods
        
    except Exception as e:
        return False, [f"Error: {e}"], []


def check_class_methods(file_path, class_name, method_names):
    """Check if specific methods exist in a class."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find the specific class
        target_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                target_class = node
                break
        
        if not target_class:
            return False, [f"Class {class_name} not found"], []
        
        # Find methods in the class
        found_methods = []
        for node in target_class.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found_methods.append(node.name)
        
        # Check for required methods
        missing_methods = []
        for method in method_names:
            if method not in found_methods:
                missing_methods.append(method)
        
        return len(missing_methods) == 0, missing_methods, found_methods
        
    except Exception as e:
        return False, [f"Error: {e}"], []


def check_function_parameters(file_path, function_name, required_params):
    """Check if a function has required parameters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find the function
        target_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                target_function = node
                break
        
        if not target_function:
            return False, f"Function {function_name} not found"
        
        # Get parameter names
        param_names = [arg.arg for arg in target_function.args.args]
        
        # Check for required parameters
        missing_params = []
        for param in required_params:
            if param not in param_names:
                missing_params.append(param)
        
        return len(missing_params) == 0, missing_params
        
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run syntax validation checks."""
    print("VLLM Batch Processing Syntax Validation")
    print("=" * 45)
    
    base_path = Path(__file__).parent.parent
    
    # Files to check
    files_to_check = [
        {
            'path': base_path / 'vlmeval/utils/batch_processing.py',
            'name': 'Batch Processing Utils',
            'functions': ['estimate_batch_processing_benefit'],
            'classes': {
                'BatchCollector': ['add_item', 'flush_all', 'get_stats'],
                'BatchProcessor': ['process_batch'],
                'BatchItem': []
            }
        },
        {
            'path': base_path / 'vlmeval/inference.py', 
            'name': 'Inference Module',
            'functions': [
                ('infer_data', ['batch_size']),
                ('infer_data_job', ['batch_size']), 
                ('infer_data_batch', [])
            ],
            'classes': {}
        },
        {
            'path': base_path / 'vlmeval/vlm/molmo.py',
            'name': 'Molmo Model',
            'functions': [],
            'classes': {
                'molmo': [
                    'generate_batch_vllm',
                    '_prepare_batch_content_vllm',
                    '_validate_batch_size',
                    '_estimate_batch_memory_usage', 
                    '_split_oversized_batch',
                    '_process_vllm_batch',
                    'supports_batch_processing',
                    'get_optimal_batch_size'
                ]
            }
        }
    ]
    
    all_passed = True
    
    for file_info in files_to_check:
        file_path = file_info['path']
        file_name = file_info['name']
        
        print(f"\nChecking {file_name}:")
        print("-" * len(f"Checking {file_name}:"))
        
        # Check if file exists
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            all_passed = False
            continue
        
        # Check syntax
        syntax_ok, syntax_error = check_python_syntax(file_path)
        if syntax_ok:
            print("✓ Syntax valid")
        else:
            print(f"✗ {syntax_error}")
            all_passed = False
            continue
        
        # Check functions
        for func_info in file_info.get('functions', []):
            if isinstance(func_info, tuple):
                func_name, required_params = func_info
                param_ok, missing = check_function_parameters(file_path, func_name, required_params)
                if param_ok:
                    print(f"✓ Function {func_name} has required parameters")
                else:
                    print(f"✗ Function {func_name} missing parameters: {missing}")
                    all_passed = False
            else:
                # Just check if function exists
                exists, missing, found = check_method_exists(file_path, [func_info])
                if exists:
                    print(f"✓ Function {func_info} exists")
                else:
                    print(f"✗ Function {func_info} not found")
                    all_passed = False
        
        # Check classes and their methods
        for class_name, methods in file_info.get('classes', {}).items():
            if methods:  # Only check if methods are specified
                exists, missing, found = check_class_methods(file_path, class_name, methods)
                if exists:
                    print(f"✓ Class {class_name} has all required methods")
                else:
                    print(f"✗ Class {class_name} missing methods: {missing}")
                    print(f"  Found methods: {found[:5]}{'...' if len(found) > 5 else ''}")
                    all_passed = False
            else:
                # Just check if class exists
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    if f"class {class_name}" in content:
                        print(f"✓ Class {class_name} exists")
                    else:
                        print(f"✗ Class {class_name} not found")
                        all_passed = False
                except Exception as e:
                    print(f"✗ Error checking class {class_name}: {e}")
                    all_passed = False
    
    # Check run.py for batch-size parameter
    print(f"\nChecking run.py:")
    print("-" * len("Checking run.py:"))
    
    run_py_path = base_path / 'run.py'
    if run_py_path.exists():
        try:
            with open(run_py_path, 'r') as f:
                content = f.read()
            
            if '--batch-size' in content:
                print("✓ --batch-size parameter added")
            else:
                print("✗ --batch-size parameter not found")
                all_passed = False
                
            if 'batch_size=args.batch_size' in content:
                print("✓ batch_size parameter passed to inference")
            else:
                print("✗ batch_size parameter not passed to inference")
                all_passed = False
                
        except Exception as e:
            print(f"✗ Error checking run.py: {e}")
            all_passed = False
    else:
        print("✗ run.py not found")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 45)
    print("VALIDATION SUMMARY")
    print("=" * 45)
    
    if all_passed:
        print("🎉 All syntax checks passed!")
        print("\nImplementation appears to be structurally sound.")
        print("Next steps:")
        print("1. Install dependencies: pip install torch transformers vllm pandas")
        print("2. Test basic functionality")
        print("3. Run with actual models")
        print("\nUsage:")
        print("  python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 4")
        return True
    else:
        print("❌ Some syntax/structure checks failed.")
        print("Please review the issues above before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)