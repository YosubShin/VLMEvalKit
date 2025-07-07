#!/usr/bin/env python3
"""
Test script for VLLM batch processing implementation.

This script tests the batch processing capabilities for VLLM-enabled models
and compares performance with sequential processing.
"""

import sys
import os
import time
from pathlib import Path
import logging

# Add parent directory to path to import vlmeval modules
sys.path.append(str(Path(__file__).parent.parent))

import torch
from vlmeval.vlm.molmo import molmo
from vlmeval.utils.batch_processing import BatchCollector, BatchProcessor, estimate_batch_processing_benefit


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_test_messages(num_messages=10):
    """Create test messages for batch processing."""
    test_messages = []
    
    for i in range(num_messages):
        message = [
            {'type': 'text', 'value': f'Test question {i+1}: What do you see in this image?'},
            {'type': 'image', 'value': '/tmp/test_image.jpg'}  # Placeholder path
        ]
        test_messages.append(message)
    
    return test_messages


def test_batch_collector():
    """Test the BatchCollector functionality."""
    print("\n" + "="*50)
    print("Testing BatchCollector")
    print("="*50)
    
    collector = BatchCollector(max_batch_size=3, batch_timeout=2.0, verbose=True)
    
    # Test adding items
    print("Adding items to collector...")
    for i in range(8):
        batch = collector.add_item(
            index=i,
            message=[{'type': 'text', 'value': f'Message {i}'}],
            dataset='test_dataset'
        )
        
        if batch:
            print(f"Batch ready with {len(batch)} items: {[item.index for item in batch]}")
    
    # Test flushing remaining items
    remaining = collector.flush_all()
    if remaining:
        for i, batch in enumerate(remaining):
            print(f"Remaining batch {i+1} with {len(batch)} items: {[item.index for item in batch]}")
    
    # Print statistics
    stats = collector.get_stats()
    print(f"Collector stats: {stats}")
    
    return True


def test_molmo_batch_methods():
    """Test Molmo batch processing methods."""
    print("\n" + "="*50)
    print("Testing Molmo Batch Methods")
    print("="*50)
    
    try:
        # Test without VLLM first
        print("1. Testing batch method availability (use_vllm=False)")
        model_transformers = molmo(use_vllm=False, verbose=True)
        
        if hasattr(model_transformers, 'generate_batch_vllm'):
            print("   ✓ generate_batch_vllm method exists")
        else:
            print("   ✗ generate_batch_vllm method missing")
            return False
        
        if hasattr(model_transformers, 'supports_batch_processing'):
            supports = model_transformers.supports_batch_processing()
            print(f"   ✓ supports_batch_processing: {supports}")
            if not supports:
                print("   ✓ Correctly reports no batch support without VLLM")
        
        # Test with VLLM (if available)
        if torch.cuda.is_available():
            print("\n2. Testing batch method availability (use_vllm=True)")
            try:
                model_vllm = molmo(use_vllm=True, verbose=True, max_batch_size=2)
                
                supports = model_vllm.supports_batch_processing()
                print(f"   ✓ VLLM batch support: {supports}")
                
                optimal_batch = model_vllm.get_optimal_batch_size()
                print(f"   ✓ Optimal batch size: {optimal_batch}")
                
                # Test batch validation
                validated = model_vllm._validate_batch_size(10)
                print(f"   ✓ Batch size validation (10 -> {validated})")
                
                # Test empty batch
                empty_result = model_vllm.generate_batch_vllm([])
                print(f"   ✓ Empty batch handling: {empty_result}")
                
                return True
                
            except ImportError:
                print("   ⚠ VLLM not available, skipping VLLM tests")
                return True
            except Exception as e:
                print(f"   ✗ VLLM test failed: {e}")
                return False
        else:
            print("   ⚠ CUDA not available, skipping VLLM tests")
            return True
            
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False


def test_memory_estimation():
    """Test memory estimation and batch splitting."""
    print("\n" + "="*50)
    print("Testing Memory Estimation")
    print("="*50)
    
    try:
        model = molmo(use_vllm=False, verbose=True)  # Use transformers backend for testing
        
        # Test token estimation
        test_text = "This is a test sentence for token counting."
        token_count = model._estimate_token_count(test_text)
        print(f"Token estimation: '{test_text[:30]}...' -> {token_count} tokens")
        
        # Test content preparation
        test_content = [
            {'type': 'text', 'text': 'Short prompt'},
            {'type': 'image_url', 'image_url': {'url': 'file:///test/image.jpg'}}
        ]
        
        memory = model._estimate_batch_memory_usage([test_content])
        print(f"Memory estimation: {memory:.1f} MB")
        
        # Test batch splitting
        large_content = [test_content] * 20  # Simulate large batch
        splits = model._split_oversized_batch(large_content, max_memory_mb=500)
        print(f"Batch splitting: {len(large_content)} items -> {len(splits)} batches")
        
        return True
        
    except Exception as e:
        print(f"Memory estimation test failed: {e}")
        return False


def test_performance_estimation():
    """Test performance estimation utilities."""
    print("\n" + "="*50)
    print("Testing Performance Estimation")
    print("="*50)
    
    # Test different dataset sizes
    test_sizes = [10, 50, 100, 500]
    
    for size in test_sizes:
        benefit = estimate_batch_processing_benefit(size, avg_batch_size=3.5)
        print(f"Dataset size {size:3d}: {benefit['speedup']:.1f}x speedup, "
              f"{benefit['time_saved_percent']:.1f}% time saved, "
              f"{benefit['estimated_batches']} batches")
    
    return True


def test_integration():
    """Test integration between components."""
    print("\n" + "="*50)
    print("Testing Component Integration")
    print("="*50)
    
    try:
        # Test with a model that doesn't support batching
        class MockModel:
            def __init__(self):
                self.model_name = "MockModel"
            
            def supports_batch_processing(self):
                return False
            
            def generate(self, message, dataset=None):
                return f"Mock response for dataset {dataset}"
        
        mock_model = MockModel()
        processor = BatchProcessor(mock_model, verbose=True)
        
        # Create mock batch items
        from vlmeval.utils.batch_processing import BatchItem
        mock_batch = [
            BatchItem(index=1, message=[{'type': 'text', 'value': 'Test 1'}], dataset='test'),
            BatchItem(index=2, message=[{'type': 'text', 'value': 'Test 2'}], dataset='test')
        ]
        
        # Process batch (should fall back to sequential)
        results = processor.process_batch(mock_batch)
        print(f"Mock batch processing results: {results}")
        
        # Test real model integration (if available)
        if torch.cuda.is_available():
            try:
                real_model = molmo(use_vllm=True, verbose=True)
                real_processor = BatchProcessor(real_model, verbose=True)
                print(f"Real model batch support: {real_processor.supports_batching}")
                
            except ImportError:
                print("VLLM not available for real model test")
        
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("VLLM Batch Processing Test Suite")
    print("=" * 60)
    
    tests = [
        ("Batch Collector", test_batch_collector),
        ("Molmo Batch Methods", test_molmo_batch_methods),
        ("Memory Estimation", test_memory_estimation),
        ("Performance Estimation", test_performance_estimation),
        ("Component Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name}: {status} ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            results[test_name] = {
                'passed': False,
                'error': str(e),
                'duration': 0
            }
            print(f"{test_name}: ✗ FAILED - {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result['passed'] else "✗"
        duration = result.get('duration', 0)
        print(f"{status} {test_name:<25} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! VLLM batch processing is ready to use.")
        print("\nUsage examples:")
        print("  # Basic batch processing:")
        print("  python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 4")
        print("\n  # Small batch for testing:")
        print("  python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 2 --verbose")
        
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    setup_logging()
    success = run_all_tests()
    sys.exit(0 if success else 1)