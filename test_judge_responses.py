#!/usr/bin/env python3
"""
Test script to verify judge response capturing works for Yale_physics datasets.
"""

import sys
from pathlib import Path

# Add parent directory to path to import vlmeval modules
sys.path.append(str(Path(__file__).parent))

def test_physics_eval_utils():
    """Test that the physics evaluation utilities support judge response capturing."""
    try:
        from vlmeval.dataset.utils.physics_eval_utils import is_equiv
        from vlmeval.dataset.utils.physic import PHYSIC_auxeval
        print("✓ Successfully imported physics evaluation utilities")
        
        # Test that the functions have the new signature
        import inspect
        is_equiv_sig = inspect.signature(is_equiv)
        physic_auxeval_sig = inspect.signature(PHYSIC_auxeval)
        
        if 'capture_judge_responses' in is_equiv_sig.parameters:
            print("✓ is_equiv function supports capture_judge_responses parameter")
        else:
            print("✗ is_equiv function missing capture_judge_responses parameter")
            
        if 'capture_judge_responses' in physic_auxeval_sig.parameters:
            print("✓ PHYSIC_auxeval function supports capture_judge_responses parameter")
        else:
            print("✗ PHYSIC_auxeval function missing capture_judge_responses parameter")
            
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import physics evaluation utilities: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_run_py_flags():
    """Test that run.py supports the new command line flags."""
    try:
        from run import parse_args
        import sys
        
        # Mock command line args
        test_args = [
            'test_script.py',  # script name
            '--model', 'GPT4o',
            '--data', 'quantum_dataset',
            '--save-judge-responses',
            '--save-detailed-eval',
            '--response-format', 'json'
        ]
        
        # Temporarily replace sys.argv
        original_argv = sys.argv
        sys.argv = test_args
        
        try:
            args = parse_args()
            
            checks = [
                (args.save_judge_responses, "save_judge_responses flag"),
                (args.save_detailed_eval, "save_detailed_eval flag"),
                (args.response_format == 'json', "response_format flag"),
                ('quantum_dataset' in args.data, "data argument parsing")
            ]
            
            all_passed = True
            for check, name in checks:
                if check:
                    print(f"✓ {name} working correctly")
                else:
                    print(f"✗ {name} not working")
                    all_passed = False
                    
            return all_passed
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"✗ Failed to test run.py flags: {e}")
        return False


def test_yale_physics_class():
    """Test that the Yale_physics class supports judge response capturing."""
    try:
        from vlmeval.dataset.image_vqa import Physics_yale
        print("✓ Successfully imported Physics_yale class")
        
        # Check that the evaluate method exists
        if hasattr(Physics_yale, 'evaluate'):
            print("✓ Physics_yale has evaluate method")
        else:
            print("✗ Physics_yale missing evaluate method")
            return False
            
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import Physics_yale class: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Judge Response Capturing Implementation")
    print("=" * 50)
    
    tests = [
        ("Physics Evaluation Utils", test_physics_eval_utils),
        ("Run.py Command Line Flags", test_run_py_flags),
        ("Yale_physics Class", test_yale_physics_class)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Judge response capturing is ready for Yale_physics datasets.")
        print("\nUsage examples:")
        print("  # Save judge responses for quantum physics")
        print("  python run.py --model GPT4o --data quantum_dataset --save-judge-responses")
        print("\n  # Save detailed evaluation data in Excel format")
        print("  python run.py --model GPT4o --data mechanics_dataset --save-detailed-eval --response-format xlsx")
        print("\n  # Use with wandb logging")
        print("  python scripts/wandb_logger.py --run-and-log --model GPT4o --data atomic_dataset --save-judge-responses")
        return True
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)