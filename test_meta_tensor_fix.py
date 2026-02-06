#!/usr/bin/env python3
"""Test script to verify meta tensor .item() fix"""

import torch
from src.kugelaudio_open.models import KugelAudioForConditionalGenerationInference

def test_model_loading():
    """Test that model can be loaded without meta tensor errors"""
    print("Testing model loading with meta tensor fix...")
    
    try:
        # Try loading with default settings
        print("\n1. Testing basic model loading...")
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            "kugelaudio/kugelaudio-0-open",
            torch_dtype=torch.float32,
        )
        print("[OK] Model loaded successfully!")
        
        # Check that the model is properly initialized
        print("\n2. Checking model components...")
        assert hasattr(model, 'acoustic_tokenizer'), "Missing acoustic_tokenizer"
        assert hasattr(model, 'semantic_tokenizer'), "Missing semantic_tokenizer"
        print("[OK] Model components present!")
        
        print("\n[SUCCESS] All tests passed! The meta tensor issue is fixed.")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)
