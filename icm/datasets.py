"""
Dataset handling for ICM.

This module provides dataset loading and formatting for ICM tasks,
supporting various formats like TruthfulQA and GSM8K.
"""

import json
import random
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datasets import load_dataset as hf_load_dataset
import logging


@dataclass
class ICMExample:
    """Single example for ICM processing."""
    input_text: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the example after initialization."""
        if not isinstance(self.input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


class ICMDataset:
    """Dataset container for ICM examples."""
    
    def __init__(self, examples: List[ICMExample], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize ICM dataset.
        
        Args:
            examples: List of ICM examples
            metadata: Dataset-level metadata
        """
        self.examples = examples
        self.metadata = metadata or {}
        self.logger = logging.getLogger(__name__)
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> ICMExample:
        """Get example by index."""
        return self.examples[idx]
    
    def shuffle(self, seed: Optional[int] = None) -> 'ICMDataset':
        """Shuffle the dataset."""
        if seed is not None:
            random.seed(seed)
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        return ICMDataset(shuffled_examples, self.metadata)
    
    def sample(self, n: int, seed: Optional[int] = None) -> 'ICMDataset':
        """Sample n examples from the dataset."""
        if seed is not None:
            random.seed(seed)
        sampled_examples = random.sample(self.examples, min(n, len(self.examples)))
        return ICMDataset(sampled_examples, self.metadata)
    
    def filter_by_metadata(self, key: str, value: Any) -> 'ICMDataset':
        """Filter examples by metadata value."""
        filtered_examples = [
            ex for ex in self.examples 
            if ex.metadata.get(key) == value
        ]
        return ICMDataset(filtered_examples, self.metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "num_examples": len(self.examples),
            "avg_input_length": sum(len(ex.input_text) for ex in self.examples) / len(self.examples),
            "metadata_keys": set()
        }
        
        for ex in self.examples:
            stats["metadata_keys"].update(ex.metadata.keys())
        
        stats["metadata_keys"] = list(stats["metadata_keys"])
        return stats


def load_icm_dataset(
    dataset_name: str,
    task_type: str = "auto",
    split: str = "train", 
    config: Optional[str] = None,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> ICMDataset:
    """
    Load dataset for ICM processing.
    
    Args:
        dataset_name: Name of dataset or path to local file
        task_type: Type of task (classification, comparison, auto)
        split: Dataset split to load (auto-detected for some datasets)
        config: Dataset configuration (auto-detected for some datasets)
        sample_size: Number of examples to sample
        seed: Random seed
        
    Returns:
        ICMDataset ready for processing
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Auto-detect split for known datasets if using default "train"
    if split == "train":
        default_split = _get_default_split(dataset_name, config)
        if default_split != "train":
            logger.info(f"Using default split '{default_split}' for {dataset_name}")
            split = default_split
    
    # Load raw dataset
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        raw_examples = _load_local_file(dataset_name)
    else:
        raw_examples = _load_huggingface_dataset(dataset_name, split, config)
    
    # Detect task type if auto
    if task_type == "auto":
        task_type = _detect_task_type(raw_examples, dataset_name)
    
    # Convert to ICM examples based on task type
    if task_type == "truthfulqa":
        examples = _convert_truthfulqa(raw_examples)
    elif task_type == "gsm8k":
        examples = _convert_gsm8k(raw_examples)
    elif task_type == "classification":
        examples = _convert_classification(raw_examples)
    elif task_type == "comparison":
        examples = _convert_comparison(raw_examples)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Create dataset
    dataset = ICMDataset(examples, {"task_type": task_type, "source": dataset_name})
    
    # Sample if requested
    if sample_size is not None:
        dataset = dataset.sample(sample_size, seed)
    
    logger.info(f"Loaded {len(dataset)} examples for {task_type} task")
    return dataset


def _load_local_file(filepath: str) -> List[Dict[str, Any]]:
    """Load dataset from local JSON/JSONL file."""
    examples = []
    
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                examples = data
            else:
                examples = [data]
    
    return examples


def _get_default_config(dataset_name: str) -> Optional[str]:
    """Get default config for known datasets that require configuration."""
    dataset_configs = {
        "truthful_qa": "multiple_choice",
        "gsm8k": "main",  # Default to main config for gsm8k
        "super_glue": "boolq",  # Default to boolq for super_glue
        "glue": "cola",  # Default to cola for glue
    }
    
    for dataset_key, default_config in dataset_configs.items():
        if dataset_key in dataset_name.lower():
            return default_config
    
    return None


def _get_default_split(dataset_name: str, config: Optional[str] = None) -> str:
    """Get default split for known datasets."""
    # Some datasets only have specific splits available
    dataset_splits = {
        "truthful_qa": "validation",  # TruthfulQA only has validation split
    }
    
    for dataset_key, default_split in dataset_splits.items():
        if dataset_key in dataset_name.lower():
            return default_split
    
    return "train"  # Default fallback


def _load_huggingface_dataset(dataset_name: str, split: str, config: Optional[str]) -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face."""
    logger = logging.getLogger(__name__)
    
    # Auto-detect config if not provided
    if config is None:
        config = _get_default_config(dataset_name)
        if config:
            logger.info(f"Auto-detected config '{config}' for {dataset_name}")
    
    # Auto-detect split if the requested split doesn't exist
    original_split = split
    
    try:
        if config:
            dataset = hf_load_dataset(dataset_name, config, split=split)
        else:
            dataset = hf_load_dataset(dataset_name, split=split)
        return list(dataset)
    except Exception as e:
        error_msg = str(e)
        
        # Check if this is a split error
        if "Unknown split" in error_msg or "split" in error_msg.lower():
            # Try with default split for this dataset
            default_split = _get_default_split(dataset_name, config)
            if default_split != original_split:
                logger.info(f"Split '{original_split}' not found, trying default split '{default_split}' for {dataset_name}")
                try:
                    if config:
                        dataset = hf_load_dataset(dataset_name, config, split=default_split)
                    else:
                        dataset = hf_load_dataset(dataset_name, split=default_split)
                    return list(dataset)
                except Exception as e2:
                    logger.warning(f"Failed with default split {default_split}: {e2}")
        
        # Check if this is a missing config error
        elif "Config name is missing" in error_msg or "available configs" in error_msg:
            # Try to auto-detect appropriate config for known datasets
            auto_config = _get_default_config(dataset_name)
            if auto_config and auto_config != config:  # Only try if different from what we already tried
                logger.info(f"Auto-detected config '{auto_config}' for {dataset_name}")
                try:
                    dataset = hf_load_dataset(dataset_name, auto_config, split=split)
                    return list(dataset)
                except Exception as e2:
                    # Also try with default split
                    default_split = _get_default_split(dataset_name, auto_config)
                    if default_split != split:
                        logger.info(f"Also trying default split '{default_split}'")
                        try:
                            dataset = hf_load_dataset(dataset_name, auto_config, split=default_split)
                            return list(dataset)
                        except Exception as e3:
                            logger.warning(f"Failed with auto-config {auto_config} and split {default_split}: {e3}")
            
            # Provide helpful error message
            raise ValueError(f"Dataset {dataset_name} requires a config parameter. {error_msg}")
        
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")


def _detect_task_type(examples: List[Dict[str, Any]], dataset_name: str) -> str:
    """Auto-detect task type from dataset."""
    dataset_name_lower = dataset_name.lower()
    
    if "truthfulqa" in dataset_name_lower:
        return "truthfulqa"
    elif "gsm8k" in dataset_name_lower:
        return "gsm8k"
    
    # Look at example structure
    if examples:
        example = examples[0]
        
        # Check for comparison format
        if any(key in example for key in ["response_a", "response_b", "chosen", "rejected"]):
            return "comparison"
        
        # Check for Q&A format
        if any(key in example for key in ["question", "answer"]):
            return "classification"
        
        # Check for instruction format
        if any(key in example for key in ["instruction", "input", "output"]):
            return "classification"
    
    return "classification"  # Default


def _convert_truthfulqa(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert TruthfulQA examples to ICM format."""
    icm_examples = []
    
    for example in examples:
        question = example.get("question", "")
        
        # Handle multiple choice answers
        if "mc1_targets" in example:
            choices = example["mc1_targets"]["choices"]
            labels = example["mc1_targets"]["labels"]
            
            for choice, label in zip(choices, labels):
                input_text = f"Question: {question}\nClaim: {choice}\nI think this Claim is [True/False]"
                metadata = {
                    "question": question,
                    "choice": choice,
                    "gold_label": "True" if label == 1 else "False",
                    "task": "truthfulness"
                }
                icm_examples.append(ICMExample(input_text, metadata))
        
        # Handle best answer format
        elif "best_answer" in example:
            best_answer = example["best_answer"]
            input_text = f"Question: {question}\nClaim: {best_answer}\nI think this Claim is [True/False]"
            metadata = {
                "question": question,
                "answer": best_answer,
                "gold_label": "True",
                "task": "truthfulness"
            }
            icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _convert_gsm8k(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert GSM8K examples to ICM format with diverse solution generation."""
    icm_examples = []
    
    for example in examples:
        question = example.get("question", "")
        original_answer = example.get("answer", "")
        
        # Generate diverse solutions for each question
        solutions = _generate_diverse_solutions(question, original_answer)
        
        for solution in solutions:
            # Create verification task - NO pre-set gold_label
            input_text = f"Question: {question}\nClaim: {solution}\nI think this Claim is [True/False]"
            metadata = {
                "question": question,
                "solution": solution,
                "original_solution": original_answer,  # Keep original for reference
                "task": "mathematical_correctness"
                # No gold_label - ICM will determine this
            }
            icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _generate_diverse_solutions(question: str, original_answer: str, num_solutions: int = 4) -> List[str]:
    """
    Generate diverse solutions for a math question.
    Returns a mix of correct and incorrect solutions as the paper describes.
    """
    solutions = []
    
    # Include the original correct solution
    solutions.append(original_answer)
    
    # Generate variations with different approaches/errors
    # Solution 2: Simplified/shortened version
    lines = original_answer.split('\n')
    if len(lines) > 2:
        # Take first and last line for a shortened version
        simplified = f"{lines[0]}\n{lines[-1]}"
        solutions.append(simplified)
    
    # Solution 3: With a calculation error
    import re
    error_solution = original_answer
    # Find calculations and introduce errors
    calc_pattern = r'<<([^>]+)>>'
    calculations = re.findall(calc_pattern, original_answer)
    if calculations:
        # Modify the first calculation to introduce an error
        original_calc = calculations[0]
        if '=' in original_calc:
            parts = original_calc.split('=')
            if len(parts) == 2 and parts[1].strip().isdigit():
                wrong_result = str(int(parts[1].strip()) + 1)  # Off by 1 error
                wrong_calc = f"{parts[0]}={wrong_result}"
                error_solution = error_solution.replace(f"<<{original_calc}>>", f"<<{wrong_calc}>>")
                # Also update the final answer
                final_answer_pattern = r'#### (\d+)'
                match = re.search(final_answer_pattern, error_solution)
                if match:
                    old_final = match.group(1)
                    new_final = str(int(old_final) + 1)
                    error_solution = error_solution.replace(f"#### {old_final}", f"#### {new_final}")
        solutions.append(error_solution)
    
    # Solution 4: Different approach with wrong logic
    wrong_approach = f"I'll solve this step by step.\n{question.split('?')[0]}?\nThe answer is clearly 42.\n#### 42"
    solutions.append(wrong_approach)
    
    return solutions[:num_solutions]


def _convert_classification(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert generic classification examples to ICM format."""
    icm_examples = []
    
    for example in examples:
        # Try to find text and label fields
        text_fields = ["text", "input", "question", "instruction", "content"]
        label_fields = ["label", "output", "answer", "target"]
        
        text = None
        label = None
        
        for field in text_fields:
            if field in example:
                text = example[field]
                break
        
        for field in label_fields:
            if field in example:
                label = example[field]
                break
        
        if text is None:
            continue
        
        input_text = f"Input: {text}\nClaim: [This needs to be classified]\nI think this Claim is [True/False]"
        metadata = {
            "original_text": text,
            "gold_label": str(label) if label is not None else "Unknown",
            "task": "classification"
        }
        icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _convert_comparison(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert comparison examples to ICM format."""
    icm_examples = []
    
    for example in examples:
        # Try to find comparison fields
        if "chosen" in example and "rejected" in example:
            response_a = example["chosen"]
            response_b = example["rejected"]
            preferred = "A"
        elif "response_a" in example and "response_b" in example:
            response_a = example["response_a"]
            response_b = example["response_b"]
            preferred = example.get("preferred", "A")
        else:
            continue
        
        query = example.get("query", example.get("prompt", "Compare these responses"))
        
        input_text = f"Query: {query}\nResponse A: {response_a}\nResponse B: {response_b}\nClaim: Response A is better than Response B\nI think this Claim is [True/False]"
        metadata = {
            "query": query,
            "response_a": response_a,
            "response_b": response_b,
            "gold_label": "True" if preferred == "A" else "False",
            "task": "comparison"
        }
        icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def create_synthetic_dataset(
    task_type: str,
    num_examples: int = 100,
    seed: int = 42
) -> ICMDataset:
    """
    Create a synthetic dataset for testing.
    
    Args:
        task_type: Type of task to create
        num_examples: Number of examples to generate
        seed: Random seed
        
    Returns:
        Synthetic ICM dataset
    """
    random.seed(seed)
    examples = []
    
    if task_type == "math":
        for i in range(num_examples):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            correct_answer = a + b
            wrong_answer = correct_answer + random.randint(1, 10)
            
            # Correct solution
            input_text = f"Question: What is {a} + {b}?\nClaim: {a} + {b} = {correct_answer}\nI think this Claim is [True/False]"
            examples.append(ICMExample(input_text, {"gold_label": "True", "task": "math"}))
            
            # Wrong solution
            input_text = f"Question: What is {a} + {b}?\nClaim: {a} + {b} = {wrong_answer}\nI think this Claim is [True/False]"
            examples.append(ICMExample(input_text, {"gold_label": "False", "task": "math"}))
    
    elif task_type == "comparison":
        for i in range(num_examples):
            query = f"Which number is larger?"
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            
            input_text = f"Query: {query}\nResponse A: {a}\nResponse B: {b}\nClaim: Response A is larger than Response B\nI think this Claim is [True/False]"
            examples.append(ICMExample(input_text, {"gold_label": "True" if a > b else "False", "task": "comparison"}))
    
    return ICMDataset(examples, {"task_type": task_type, "synthetic": True})
