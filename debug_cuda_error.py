#!/usr/bin/env python3
"""Debug script to identify CUDA error with Gemma-3-270m-it model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model(model_name, device="cuda"):
    """Test a model to see if it causes CUDA errors."""
    logger.info(f"\nTesting {model_name} on {device}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Vocab size: {tokenizer.vocab_size}")
        
        # Load model
        if device == "cuda" and torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        logger.info(f"Model loaded, config vocab_size: {model.config.vocab_size}")
        
        # Test tokenization
        test_prompt = "Input: Is 2+2=4? Label:"
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Max token ID in input: {inputs['input_ids'].max().item()}")
        
        # Get logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Expected shape: torch.Size([{model.config.vocab_size}])")
        
        # Test token ID access
        test_tokens = [
            tokenizer.encode("True", add_special_tokens=False)[0],
            tokenizer.encode("False", add_special_tokens=False)[0],
            tokenizer.encode("true", add_special_tokens=False)[0],
            tokenizer.encode("false", add_special_tokens=False)[0],
        ]
        
        logger.info(f"Test token IDs: {test_tokens}")
        
        for tid in test_tokens:
            logger.info(f"Checking token {tid} < {logits.shape[0]}: {tid < logits.shape[0]}")
            if tid >= logits.shape[0]:
                logger.error(f"Token {tid} is out of bounds for logits of size {logits.shape[0]}")
                continue
                
            # Try to access the logit
            try:
                # Method 1: Direct indexing
                value1 = logits[tid].item()
                logger.info(f"  Direct access logits[{tid}] = {value1:.4f}")
            except Exception as e:
                logger.error(f"  Direct access failed: {e}")
            
            try:
                # Method 2: Tensor indexing
                indices = torch.tensor([tid], device=logits.device, dtype=torch.long)
                value2 = logits[indices].max().item()
                logger.info(f"  Tensor access = {value2:.4f}")
            except Exception as e:
                logger.error(f"  Tensor access failed: {e}")
        
        logger.info(f"✓ {model_name} test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ {model_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check available devices
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        device = "cuda"
    else:
        logger.info("CUDA not available, using CPU")
        device = "cpu"
    
    # Test both models
    models = [
        "google/gemma-3-270m-it",
        "google/gemma-3-1b-it"
    ]
    
    for model_name in models:
        success = test_model(model_name, device)
        if not success and device == "cuda":
            logger.info(f"Retrying {model_name} on CPU...")
            test_model(model_name, "cpu")