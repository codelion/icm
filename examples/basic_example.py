#!/usr/bin/env python3
"""
Basic ICM Example: Truth/Falsehood Classification

This example demonstrates how to use ICM for a simple truth/falsehood
classification task using a small model.
"""

import logging
from icm import ICMSearcher, load_icm_dataset, ICMStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run basic ICM example."""
    
    # Create a simple synthetic dataset for demonstration
    logger.info("Creating synthetic dataset...")
    
    # For this example, we'll use a very simple synthetic dataset
    # In practice, you would load a real dataset
    from icm.datasets import create_synthetic_dataset
    
    dataset = create_synthetic_dataset(
        task_type="math",
        num_examples=50,  # Small for demonstration
        seed=42
    )
    
    logger.info(f"Created dataset with {len(dataset)} examples")
    
    # Create ICM searcher with conservative parameters for demo
    logger.info("Initializing ICM searcher...")
    
    searcher = ICMSearcher(
        model_name="distilgpt2",  # Small model for demo
        alpha=30.0,               # Lower alpha for faster convergence
        initial_temperature=5.0,   # Lower temperature
        final_temperature=0.01,
        max_iterations=100,        # Fewer iterations for demo
        initial_examples=4,        # Fewer initial examples
        generation_temperature=0.8,
        seed=42
    )
    
    # Run ICM search
    logger.info("Running ICM search...")
    
    result = searcher.search(
        dataset=dataset,
        task_type="classification",
        max_examples=20  # Process only first 20 examples
    )
    
    logger.info(f"ICM search completed!")
    logger.info(f"Final score: {result.score:.4f}")
    logger.info(f"Generated {len(result.labeled_examples)} labeled examples")
    
    # Show label distribution
    label_counts = {}
    for example in result.labeled_examples:
        label = example["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(result.labeled_examples)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Save results
    storage = ICMStorage("examples/results")
    result_path = storage.save_result(result, "basic_example")
    
    logger.info(f"Results saved to: {result_path}")
    
    # Show some example outputs
    logger.info("\nSample labeled examples:")
    for i, example in enumerate(result.labeled_examples[:5]):
        logger.info(f"Example {i+1}:")
        logger.info(f"  Input: {example['input'][:100]}...")
        logger.info(f"  Label: {example['label']}")
        logger.info("")
    
    return result

if __name__ == "__main__":
    result = main()
    print(f"\nBasic ICM example completed successfully!")
    print(f"Generated {len(result.labeled_examples)} labeled examples with score {result.score:.4f}")
