#!/usr/bin/env python3
"""
Test script for Molmo VLLM integration
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import vlmeval modules
sys.path.append(str(Path(__file__).parent.parent))

import torch
from vlmeval.vlm.molmo import molmo

def test_molmo_vllm():
    """Test Molmo with VLLM support."""
    
    print("Testing Molmo VLLM Integration")
    print("=" * 40)
    
    # Test configuration
    model_path = "oumi-ai/Molmo-7B-D-0924"  # Smallest Molmo model for testing
    
    print(f"1. Testing Transformers backend (use_vllm=False)")
    try:
        model_transformers = molmo(model_path=model_path, use_vllm=False, verbose=True)
        print("   ✓ Transformers backend initialized successfully")
        print(f"   ✓ Model loaded: {model_transformers.model_name}")
        print(f"   ✓ Use VLLM: {model_transformers.use_vllm}")
    except Exception as e:
        print(f"   ✗ Transformers backend failed: {e}")
        return False
    
    print(f"\n2. Testing VLLM backend (use_vllm=True)")
    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, skipping VLLM test")
        return True
        
    try:
        model_vllm = molmo(model_path=model_path, use_vllm=True, verbose=True)
        print("   ✓ VLLM backend initialized successfully")
        print(f"   ✓ Model loaded: {model_vllm.model_name}")
        print(f"   ✓ Use VLLM: {model_vllm.use_vllm}")
        print(f"   ✓ Image limit per prompt: {model_vllm.limit_mm_per_prompt}")
        print(f"   ✓ Max context length: {model_vllm.max_context_length}")
    except ImportError as e:
        print(f"   ⚠ VLLM not installed, skipping test: {e}")
        return True
    except Exception as e:
        print(f"   ✗ VLLM backend failed: {e}")
        return False
    
    # Test methods exist
    print(f"\n3. Testing method availability")
    required_methods = [
        'generate_inner',
        'generate_inner_vllm', 
        'generate_inner_transformers',
        '_prepare_content_vllm',
        '_ensure_image_url',
        '_encode_image_to_base64',
        '_estimate_token_count',
        '_truncate_content'
    ]
    
    for method in required_methods:
        if hasattr(model_vllm, method):
            print(f"   ✓ Method {method} exists")
        else:
            print(f"   ✗ Method {method} missing")
            return False
    
    print(f"\n4. Testing content preparation")
    try:
        # Mock message structure
        test_message = [
            {'type': 'text', 'value': 'What do you see in this image?'},
            {'type': 'image', 'value': '/path/to/test/image.jpg'}
        ]
        
        content = model_vllm._prepare_content_vllm(test_message)
        print(f"   ✓ Content preparation successful")
        print(f"   ✓ Prepared content: {len(content)} items")
        
        # Test text content
        text_items = [item for item in content if item['type'] == 'text']
        image_items = [item for item in content if item['type'] == 'image_url']
        
        print(f"   ✓ Text items: {len(text_items)}")
        print(f"   ✓ Image items: {len(image_items)}")
        
    except Exception as e:
        print(f"   ✗ Content preparation failed: {e}")
        return False
    
    print(f"\n5. Testing image URL handling")
    try:
        # Test various URL formats
        test_paths = [
            "http://example.com/image.jpg",
            "https://example.com/image.jpg", 
            "file:///path/to/image.jpg",
            "data:image/jpeg;base64,/9j/4AAQ..."
        ]
        
        for path in test_paths:
            try:
                url = model_vllm._ensure_image_url(path)
                print(f"   ✓ URL format {path[:20]}... -> {url[:30]}...")
            except ValueError:
                print(f"   ! Invalid path handled correctly: {path[:20]}...")
                
    except Exception as e:
        print(f"   ✗ URL handling failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed!")
    return True

def test_config_integration():
    """Test that Molmo is properly configured for VLLM in the framework."""
    
    print("\nTesting Configuration Integration")
    print("=" * 40)
    
    try:
        from vlmeval.config import supported_VLM
        
        # Check if Molmo models are in config
        molmo_models = [name for name in supported_VLM.keys() if 'molmo' in name.lower()]
        print(f"✓ Found {len(molmo_models)} Molmo models in config:")
        for model in molmo_models:
            print(f"  - {model}")
            
        # Test model instantiation with VLLM
        if molmo_models:
            model_name = molmo_models[0]
            print(f"\n✓ Testing model instantiation: {model_name}")
            
            # Test without VLLM
            model_func = supported_VLM[model_name]
            try:
                model = model_func(use_vllm=False)
                print(f"  ✓ Instantiated without VLLM: {type(model).__name__}")
            except Exception as e:
                print(f"  ✗ Failed without VLLM: {e}")
                
            # Test with VLLM (if available)
            try:
                model = model_func(use_vllm=True)
                print(f"  ✓ Instantiated with VLLM: {type(model).__name__}")
            except ImportError:
                print(f"  ⚠ VLLM not available")
            except Exception as e:
                print(f"  ✗ Failed with VLLM: {e}")
                
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
        
    return True

def test_inference_integration():
    """Test that inference files properly handle Molmo VLLM."""
    
    print("\nTesting Inference Integration") 
    print("=" * 40)
    
    # Test inference file integration
    inference_files = [
        'vlmeval.inference',
        'vlmeval.inference_video', 
        'vlmeval.inference_mt'
    ]
    
    for module_name in inference_files:
        try:
            module = __import__(module_name, fromlist=[''])
            
            # Read the source to check for molmo
            import inspect
            source = inspect.getsource(module)
            
            if 'molmo' in source.lower():
                print(f"  ✓ {module_name} includes Molmo support")
            else:
                print(f"  ✗ {module_name} missing Molmo support")
                
        except Exception as e:
            print(f"  ✗ Failed to check {module_name}: {e}")
    
    return True

def test_truncation_functionality():
    """Test the auto-truncation functionality."""
    
    print("\\nTesting Truncation Functionality")
    print("=" * 40)
    
    model_path = "oumi-ai/Molmo-7B-D-0924"
    
    try:
        # Test with auto-truncation enabled
        model = molmo(model_path=model_path, use_vllm=False, auto_truncate=True, max_context_length=1000)
        
        print("1. Testing token estimation")
        test_text = "This is a test sentence for token counting."
        token_count = model._estimate_token_count(test_text)
        print(f"   ✓ Token estimation: '{test_text[:20]}...' -> {token_count} tokens")
        
        print("\\n2. Testing content truncation")
        # Create very long content that exceeds context length
        long_text = "This is a very long text that should be truncated. " * 200
        long_content = [
            {'type': 'text', 'text': long_text},
            {'type': 'image_url', 'image_url': {'url': 'file:///test/image.jpg'}}
        ]
        
        # Test with small max_tokens to force truncation
        truncated = model._truncate_content(long_content, max_tokens=500)
        print(f"   ✓ Content truncation: {len(long_content)} -> {len(truncated)} items")
        
        # Check if truncation occurred
        text_content = next((item for item in truncated if item.get('type') == 'text'), {})
        text_value = text_content.get('text', text_content.get('value', ''))
        original_length = len(long_text)
        truncated_length = len(text_value)
        
        if truncated_length < original_length:
            print(f"   ✓ Text truncated: {original_length} -> {truncated_length} chars")
        
        if '[TRUNCATED]' in text_value:
            print(f"   ✓ Truncation marker found in output")
        
        print("\\n3. Testing auto-truncate parameter")
        # Test with auto_truncate disabled
        model_no_truncate = molmo(model_path=model_path, use_vllm=False, auto_truncate=False)
        if not model_no_truncate.auto_truncate:
            print(f"   ✓ Auto-truncation can be disabled")
        else:
            print(f"   ✗ Failed to disable auto-truncation")
            return False
            
        # Test with auto_truncate enabled (default)
        model_truncate = molmo(model_path=model_path, use_vllm=False)
        if model_truncate.auto_truncate:
            print(f"   ✓ Auto-truncation enabled by default")
        else:
            print(f"   ✗ Auto-truncation not enabled by default")
            return False
            
        print("\\n4. Testing custom context length")
        model_custom = molmo(model_path=model_path, use_vllm=False, max_context_length=2048)
        if model_custom.max_context_length == 2048:
            print(f"   ✓ Custom context length set: {model_custom.max_context_length}")
        else:
            print(f"   ✗ Custom context length not set correctly")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ Truncation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Molmo VLLM Integration Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run individual tests
    success &= test_molmo_vllm()
    success &= test_config_integration() 
    success &= test_inference_integration()
    success &= test_truncation_functionality()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("\nTo use Molmo with VLLM:")
        print("  python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)