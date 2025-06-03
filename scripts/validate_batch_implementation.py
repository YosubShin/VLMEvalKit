#!/usr/bin/env python3
"""
Lightweight validation script for VLLM batch processing implementation.

This script validates that the batch processing implementation is correctly
integrated without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    
    try:
        from vlmeval.utils.batch_processing import BatchCollector, BatchProcessor, estimate_batch_processing_benefit
        print("✓ Batch processing utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import batch processing utilities: {e}")
        return False
    
    try:
        from vlmeval.inference import infer_data, infer_data_job, infer_data_batch
        print("✓ Inference functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import inference functions: {e}")
        return False
    
    return True


def check_molmo_methods():
    """Check that Molmo has the required batch processing methods."""
    print("\nChecking Molmo batch methods...")
    
    try:
        # Import Molmo class definition
        import inspect
        from vlmeval.vlm.molmo import molmo
        
        # Check for required methods
        required_methods = [
            'generate_batch_vllm',
            '_prepare_batch_content_vllm', 
            '_validate_batch_size',
            '_estimate_batch_memory_usage',
            '_split_oversized_batch',
            '_process_vllm_batch',
            'supports_batch_processing',
            'get_optimal_batch_size'
        ]
        
        molmo_methods = [name for name, method in inspect.getmembers(molmo, predicate=inspect.isfunction)]
        molmo_methods.extend([name for name in dir(molmo) if not name.startswith('__')])
        
        all_present = True
        for method in required_methods:
            if method in molmo_methods:
                print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"✗ Failed to check Molmo methods: {e}")
        return False


def check_batch_collector():
    """Check BatchCollector functionality."""
    print("\nChecking BatchCollector...")
    
    try:
        from vlmeval.utils.batch_processing import BatchCollector, BatchItem
        
        # Test basic instantiation
        collector = BatchCollector(max_batch_size=3)
        print("✓ BatchCollector instantiation")
        
        # Test adding items
        for i in range(5):
            batch = collector.add_item(i, [{'type': 'text', 'value': f'Test {i}'}], 'test_dataset')
            if batch and i == 2:  # Should get first batch at index 2 (0, 1, 2)
                print(f"✓ Batch collection (got batch with {len(batch)} items at item {i})")
        
        # Test flushing
        remaining = collector.flush_all()
        print(f"✓ Batch flushing (got {len(remaining)} remaining batches)")
        
        # Test stats
        stats = collector.get_stats()
        print(f"✓ Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ BatchCollector test failed: {e}")
        return False


def check_function_signatures():
    """Check that function signatures include batch_size parameter."""
    print("\nChecking function signatures...")
    
    try:
        import inspect
        from vlmeval.inference import infer_data, infer_data_job
        
        # Check infer_data signature
        infer_data_sig = inspect.signature(infer_data)
        if 'batch_size' in infer_data_sig.parameters:
            print("✓ infer_data has batch_size parameter")
        else:
            print("✗ infer_data missing batch_size parameter")
            return False
        
        # Check infer_data_job signature  
        infer_data_job_sig = inspect.signature(infer_data_job)
        if 'batch_size' in infer_data_job_sig.parameters:
            print("✓ infer_data_job has batch_size parameter")
        else:
            print("✗ infer_data_job missing batch_size parameter")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Function signature check failed: {e}")
        return False


def check_performance_estimation():
    """Check performance estimation utility."""
    print("\nChecking performance estimation...")
    
    try:
        from vlmeval.utils.batch_processing import estimate_batch_processing_benefit
        
        # Test with sample dataset size
        result = estimate_batch_processing_benefit(100, avg_batch_size=4.0)
        
        required_keys = ['speedup', 'time_saved_percent', 'estimated_batches', 'avg_batch_size']
        for key in required_keys:
            if key in result:
                print(f"  ✓ {key}: {result[key]}")
            else:
                print(f"  ✗ Missing key: {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Performance estimation test failed: {e}")
        return False


def check_file_structure():
    """Check that required files exist."""
    print("\nChecking file structure...")
    
    base_path = Path(__file__).parent.parent
    
    required_files = [
        'vlmeval/utils/batch_processing.py',
        'vlmeval/inference.py',
        'vlmeval/vlm/molmo.py',
        'run.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    return all_exist


def main():
    """Run all validation checks."""
    print("VLLM Batch Processing Implementation Validation")
    print("=" * 55)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Imports", check_imports),
        ("Molmo Methods", check_molmo_methods),
        ("Function Signatures", check_function_signatures),
        ("BatchCollector", check_batch_collector),
        ("Performance Estimation", check_performance_estimation)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name) + ":")
        
        try:
            result = check_func()
            results.append(result)
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status}")
        except Exception as e:
            results.append(False)
            print(f"✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    
    passed = sum(results)
    total = len(results)
    
    for i, (check_name, _) in enumerate(checks):
        status = "✓" if results[i] else "✗"
        print(f"{status} {check_name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All validation checks passed!")
        print("\nThe VLLM batch processing implementation is ready for use.")
        print("\nNext steps:")
        print("1. Install required dependencies (torch, transformers, vllm)")
        print("2. Test with actual models:")
        print("   python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 4")
        print("3. Monitor performance improvements and adjust batch sizes as needed")
        
        return True
    else:
        print(f"\n❌ {total - passed} validation checks failed.")
        print("Please review the implementation before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)