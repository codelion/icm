#!/usr/bin/env python3
"""
Test script to verify all benchmark datasets work with ICM.
Tests each dataset with 1 sample to ensure conversion functions work properly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from icm.datasets import load_icm_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset(dataset_name, task_type=None, config=None):
    """Test a single dataset with 1 sample."""
    try:
        logger.info(f"\n=== Testing {dataset_name} ===")
        
        # Load 1 sample
        dataset = load_icm_dataset(
            dataset_name=dataset_name,
            task_type=task_type or "auto",
            sample_size=1,
            config=config
        )
        
        logger.info(f"✓ Successfully loaded {len(dataset)} ICM examples")
        
        # Print first example for verification
        if len(dataset) > 0:
            example = dataset[0]
            logger.info(f"Example input preview: {example.input_text[:200]}...")
            logger.info(f"Example metadata: {example.metadata}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load {dataset_name}: {str(e)}")
        return False

def main():
    """Test all benchmark datasets."""
    datasets_to_test = [
        # Dataset name, task_type, config
        ("Rowan/hellaswag", "hellaswag", None),
        ("piqa", "piqa", None),
        ("allenai/ai2_arc", "arc_challenge", "ARC-Challenge"),
        ("allenai/winogrande", "winogrande", "winogrande_xl"),
        ("truthful_qa", "truthfulqa", "multiple_choice"),  # Existing
        ("gsm8k", "gsm8k", "main"),  # Existing
    ]
    
    results = {}
    
    for dataset_name, task_type, config in datasets_to_test:
        success = test_dataset(dataset_name, task_type, config)
        results[dataset_name] = success
    
    # BIG-Bench Hard and IFEval might need special handling
    logger.info("\n=== Testing BIG-Bench Hard ===")
    # Note: BIG-Bench Hard might not be available as a single HF dataset
    # Will handle this separately
    
    logger.info("\n=== Testing IFEval ===")
    # Note: IFEval might need special dataset loading
    # Will handle this separately
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    for dataset_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{dataset_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    logger.info(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.info("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())