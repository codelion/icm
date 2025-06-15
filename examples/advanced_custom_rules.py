#!/usr/bin/env python3
"""
Advanced ICM Example: Custom Consistency Rules

This example shows how to create custom consistency rules
and use them with ICM for specialized tasks.
"""

import logging
from typing import List, Tuple
from icm import ICMSearcher, ICMDataset, ICMExample
from icm.consistency import ConsistencyRule, LogicalConsistencyChecker
from icm.storage import ICMStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemperatureConsistencyRule(ConsistencyRule):
    """Custom rule for temperature comparison tasks."""
    
    def check(self, example1: ICMExample, example2: ICMExample, label1: str, label2: str) -> bool:
        """Check if temperature comparisons are consistent."""
        temp1 = self._extract_temperature(example1.input_text)
        temp2 = self._extract_temperature(example2.input_text)
        
        if temp1 is not None and temp2 is not None:
            # If we're comparing the same temperatures, both can't claim different ordering
            if temp1 == temp2:
                # A == B, so "A > B" and "B > A" can't both be true
                if "greater than" in example1.input_text and "greater than" in example2.input_text:
                    return not (label1 == "True" and label2 == "True")
        
        return True
    
    def get_consistent_options(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        current_label1: str, 
        current_label2: str
    ) -> List[Tuple[str, str]]:
        """Get consistent options for temperature comparisons."""
        temp1 = self._extract_temperature(example1.input_text)
        temp2 = self._extract_temperature(example2.input_text)
        
        if temp1 is not None and temp2 is not None and temp1 == temp2:
            if "greater than" in example1.input_text and "greater than" in example2.input_text:
                # Same temperatures can't both be greater than each other
                return [("True", "False"), ("False", "True"), ("False", "False")]
        
        return [("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")]
    
    def has_relationship(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Check if examples involve temperature comparisons."""
        return ("temperature" in example1.input_text.lower() and 
                "temperature" in example2.input_text.lower() and
                "greater than" in example1.input_text.lower() and
                "greater than" in example2.input_text.lower())
    
    def _extract_temperature(self, text: str) -> float:
        """Extract temperature value from text."""
        import re
        # Look for patterns like "25°C" or "25 degrees"
        patterns = [
            r"(\d+\.?\d*)\s*°C",
            r"(\d+\.?\d*)\s*degrees?\s*C",
            r"(\d+\.?\d*)\s*celsius"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None


def create_temperature_dataset() -> ICMDataset:
    """Create a dataset with temperature comparison examples."""
    examples = []
    
    # Temperature comparison examples
    temperatures = [20, 25, 30, 35, 40]
    
    for i, temp1 in enumerate(temperatures):
        for j, temp2 in enumerate(temperatures):
            if i != j:  # Don't compare temperature with itself
                # Create comparison claims
                input_text = f"Is {temp1}°C greater than {temp2}°C in temperature?"
                
                # Determine correct label
                correct_label = "True" if temp1 > temp2 else "False"
                
                examples.append(ICMExample(
                    input_text, 
                    {
                        "temp1": temp1,
                        "temp2": temp2,
                        "gold_label": correct_label,
                        "task": "temperature_comparison"
                    }
                ))
    
    logger.info(f"Created temperature dataset with {len(examples)} examples")
    return ICMDataset(examples, {"task_type": "temperature_comparison"})


def main():
    """Run advanced ICM example with custom consistency rules."""
    
    # Create temperature comparison dataset
    logger.info("Creating temperature comparison dataset...")
    dataset = create_temperature_dataset()
    
    # Create custom consistency checker with our temperature rule
    logger.info("Setting up custom consistency rules...")
    custom_checker = LogicalConsistencyChecker()
    custom_checker.add_rule(TemperatureConsistencyRule())
    
    # Create ICM searcher with custom consistency checker
    logger.info("Initializing ICM searcher with custom rules...")
    
    searcher = ICMSearcher(
        model_name="google/gemma-3-1b-it",
        alpha=40.0,
        initial_temperature=8.0,
        final_temperature=0.01,
        max_iterations=200,
        initial_examples=6,
        consistency_checker=custom_checker,  # Use our custom checker
        generation_temperature=0.7,
        seed=42
    )
    
    # Run ICM search
    logger.info("Running ICM search with custom consistency rules...")
    
    result = searcher.search(
        dataset=dataset,
        task_type="comparison",
        max_examples=30
    )
    
    logger.info(f"ICM search completed!")
    logger.info(f"Final score: {result.score:.4f}")
    logger.info(f"Generated {len(result.labeled_examples)} labeled examples")
    
    # Analyze results
    label_counts = {}
    correct_predictions = 0
    
    for example in result.labeled_examples:
        label = example["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Check accuracy if we have gold labels
        gold_label = example.get("metadata", {}).get("gold_label")
        if gold_label and gold_label == label:
            correct_predictions += 1
    
    logger.info("Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(result.labeled_examples)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    if correct_predictions > 0:
        accuracy = (correct_predictions / len(result.labeled_examples)) * 100
        logger.info(f"Accuracy vs gold labels: {accuracy:.1f}%")
    
    # Check consistency violations
    consistency_violations = custom_checker.count_inconsistencies(result.labeled_examples)
    logger.info(f"Consistency violations: {consistency_violations}")
    
    # Save results
    storage = ICMStorage("examples/results")
    result_path = storage.save_result(result, "advanced_custom_rules")
    
    logger.info(f"Results saved to: {result_path}")
    
    # Show some example outputs
    logger.info("\nSample labeled examples:")
    for i, example in enumerate(result.labeled_examples[:3]):
        logger.info(f"Example {i+1}:")
        logger.info(f"  Input: {example['input']}")
        logger.info(f"  Predicted Label: {example['label']}")
        gold_label = example.get("metadata", {}).get("gold_label")
        if gold_label:
            logger.info(f"  Gold Label: {gold_label}")
            status = "✓" if gold_label == example['label'] else "✗"
            logger.info(f"  Status: {status}")
        logger.info("")
    
    return result


if __name__ == "__main__":
    result = main()
    print(f"\nAdvanced ICM example completed successfully!")
    print(f"Generated {len(result.labeled_examples)} labeled examples with score {result.score:.4f}")
