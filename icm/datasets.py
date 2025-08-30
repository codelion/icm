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
    
    # Sample raw examples BEFORE conversion to control number of base questions
    if sample_size is not None:
        if sample_size < len(raw_examples):
            logger.info(f"Sampling {sample_size} base questions from {len(raw_examples)} available")
            random.seed(seed)
            raw_examples = random.sample(raw_examples, sample_size)
    
    # Convert to ICM examples based on task type (this will multiply examples)
    if task_type == "truthfulqa":
        examples = _convert_truthfulqa(raw_examples)
    elif task_type == "gsm8k":
        examples = _convert_gsm8k(raw_examples)
    elif task_type == "hellaswag":
        examples = _convert_hellaswag(raw_examples)
    elif task_type == "piqa":
        examples = _convert_piqa(raw_examples)
    elif task_type == "arc_challenge":
        examples = _convert_arc_challenge(raw_examples)
    elif task_type == "winogrande":
        examples = _convert_winogrande(raw_examples)
    elif task_type == "bigbench_hard":
        examples = _convert_bigbench_hard(raw_examples)
    elif task_type == "ifeval":
        examples = _convert_ifeval(raw_examples)
    elif task_type == "classification":
        examples = _convert_classification(raw_examples)
    elif task_type == "comparison":
        examples = _convert_comparison(raw_examples)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Create dataset
    dataset = ICMDataset(examples, {"task_type": task_type, "source": dataset_name})
    
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
        "ai2_arc": "ARC-Challenge",  # ARC Challenge configuration
        "allenai/ai2_arc": "ARC-Challenge",
        "arc": "ARC-Challenge",
        "winogrande": "winogrande_xl",  # Default WinoGrande config
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
        "hellaswag": "validation",  # HellaSwag uses validation for eval
        "piqa": "validation",  # PIQA uses validation for eval
        "ai2_arc": "test",  # ARC uses test split
        "allenai/ai2_arc": "test",
        "arc": "test",
        "winogrande": "validation",  # WinoGrande uses validation
        "bigbench": "train",  # BIG-Bench Hard uses train split
        "maveriq/bigbenchhard": "train",  # Full BigBench Hard dataset name
    }
    
    for dataset_key, default_split in dataset_splits.items():
        if dataset_key in dataset_name.lower():
            return default_split
    
    return "train"  # Default fallback


def _load_huggingface_dataset(dataset_name: str, split: str, config: Optional[str]) -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face."""
    logger = logging.getLogger(__name__)
    
    # Some datasets require trust_remote_code
    trust_remote_datasets = ["piqa"]
    kwargs = {}
    if any(d in dataset_name.lower() for d in trust_remote_datasets):
        kwargs["trust_remote_code"] = True
        logger.info(f"Using trust_remote_code=True for {dataset_name}")
    
    # Auto-detect config if not provided
    if config is None:
        config = _get_default_config(dataset_name)
        if config:
            logger.info(f"Auto-detected config '{config}' for {dataset_name}")
    
    # Auto-detect split if the requested split doesn't exist
    original_split = split
    
    try:
        if config:
            dataset = hf_load_dataset(dataset_name, config, split=split, **kwargs)
        else:
            dataset = hf_load_dataset(dataset_name, split=split, **kwargs)
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
                        dataset = hf_load_dataset(dataset_name, config, split=default_split, **kwargs)
                    else:
                        dataset = hf_load_dataset(dataset_name, split=default_split, **kwargs)
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
                    dataset = hf_load_dataset(dataset_name, auto_config, split=split, **kwargs)
                    return list(dataset)
                except Exception as e2:
                    # Also try with default split
                    default_split = _get_default_split(dataset_name, auto_config)
                    if default_split != split:
                        logger.info(f"Also trying default split '{default_split}'")
                        try:
                            dataset = hf_load_dataset(dataset_name, auto_config, split=default_split, **kwargs)
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
    elif "hellaswag" in dataset_name_lower:
        return "hellaswag"
    elif "piqa" in dataset_name_lower:
        return "piqa"
    elif "arc" in dataset_name_lower or "ai2_arc" in dataset_name_lower:
        return "arc_challenge"
    elif "winogrande" in dataset_name_lower:
        return "winogrande"
    elif "bigbench" in dataset_name_lower or "bbh" in dataset_name_lower:
        return "bigbench_hard"
    elif "ifeval" in dataset_name_lower:
        return "ifeval"
    
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
    """Convert TruthfulQA examples to ICM format using actual answers only."""
    icm_examples = []
    
    for example in examples:
        question = example.get("question", "")
        
        # Handle multiple choice answers
        if "mc1_targets" in example:
            choices = example["mc1_targets"]["choices"]
            
            # Use ALL actual choices from the dataset - let ICM discover which are true
            for choice in choices:
                input_text = f"Question: {question}\nAnswer: {choice}\nClaim: This answer is truthful\nI think this Claim is [True/False]"
                metadata = {
                    "question": question,
                    "choice": choice,
                    "task": "truthfulness",
                    "response_text": choice,  # Just the answer for clean DPO responses
                    "prompt_text": question  # Store question separately
                    # No gold_label - ICM will determine this through internal coherence
                }
                icm_examples.append(ICMExample(input_text, metadata))
        
        # Handle best answer format
        elif "best_answer" in example:
            best_answer = example["best_answer"]
            incorrect_answers = example.get("incorrect_answers", [])
            
            # Use actual answers from dataset - both correct and incorrect
            all_answers = [best_answer] + incorrect_answers[:5]  # Limit to reasonable number
            
            for answer in all_answers:
                input_text = f"Question: {question}\nAnswer: {answer}\nClaim: This answer is truthful\nI think this Claim is [True/False]"
                metadata = {
                    "question": question,
                    "answer": answer,
                    "task": "truthfulness",
                    "response_text": answer,  # Just the answer for clean DPO responses
                    "prompt_text": question  # Store question separately
                    # No gold_label - ICM will determine this through internal coherence
                }
                icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _convert_gsm8k(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert GSM8K examples to ICM format using original solutions only."""
    icm_examples = []
    
    for example in examples:
        question = example.get("question", "")
        original_answer = example.get("answer", "")
        
        # Use ONLY the original answer - let ICM discover if it's correct
        # Create verification task - NO pre-set gold_label
        input_text = f"Question: {question}\nSolution: {original_answer}\nClaim: This solution correctly solves the problem\nI think this Claim is [True/False]"
        metadata = {
            "question": question,
            "solution": original_answer,
            "task": "mathematical_correctness",
            "response_text": original_answer,  # Just the solution for clean DPO responses
            "prompt_text": question  # Store question separately
            # No gold_label - ICM will determine this through internal coherence
        }
        icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples










def _convert_classification(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert generic classification examples to ICM format with diverse claims."""
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
        
        # Use simple claim about the text - let ICM discover the label
        claim = f"This text matches its given classification"
        
        # Create single example per text
        input_text = f"Input: {text}\nLabel: {label}\nClaim: This classification is correct\nI think this Claim is [True/False]"
        metadata = {
            "original_text": text,
            "original_label": label,
            "task": "classification",
            "response_text": str(label),  # Just the label for clean DPO responses
            "prompt_text": text  # Store text separately
            # No gold_label - ICM will determine this through internal coherence
        }
        icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _convert_comparison(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert comparison examples to ICM format with diverse comparison claims."""
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
        
        # Use simple comparison claim - let ICM discover the preference
        claim = f"Response A is better than Response B"
        input_text = f"Query: {query}\nResponse A: {response_a}\nResponse B: {response_b}\nClaim: {claim}\nI think this Claim is [True/False]"
        metadata = {
            "query": query,
            "response_a": response_a,
            "response_b": response_b,
            "claim": claim,
            "preferred": preferred,
            "task": "comparison",
            "response_text": f"A is preferred" if preferred == "A" else "B is preferred",
            "prompt_text": query
            # No gold_label - ICM will determine this through internal coherence
        }
        icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _convert_hellaswag(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert HellaSwag examples to ICM format with diverse ending verification."""
    icm_examples = []
    
    for example in examples:
        ctx_a = example.get("ctx_a", "")
        ctx_b = example.get("ctx_b", "")
        ctx = f"{ctx_a} {ctx_b}".strip()
        
        # HellaSwag has 4 possible endings
        endings = example.get("endings", [])
        activity_label = example.get("activity_label", "")
        
        # Create diverse claims for each ending
        for i, ending in enumerate(endings):
            # Main claim - direct completion
            claim = f"Ending {i+1} correctly completes this context"
            input_text = f"Context: {ctx}\nEnding {i+1}: {ending}\nClaim: {claim}\nI think this Claim is [True/False]"
            
            metadata = {
                "context": ctx,
                "ending": ending,
                "ending_index": i,
                "activity_label": activity_label,
                "claim": claim,
                "task": "common_sense_completion",
                "response_text": ending,  # Just the ending for clean DPO responses
                "prompt_text": ctx  # Store context separately
            }
            icm_examples.append(ICMExample(input_text, metadata))
            
            # Alternative claim - coherence based
            alt_claim = f"This ending makes logical sense given the context"
            alt_input_text = f"Context: {ctx}\nEnding: {ending}\nClaim: {alt_claim}\nI think this Claim is [True/False]"
            
            alt_metadata = {
                "context": ctx,
                "ending": ending,
                "ending_index": i,
                "activity_label": activity_label,
                "claim": alt_claim,
                "task": "common_sense_completion",
                "response_text": ending,  # Just the ending for clean DPO responses
                "prompt_text": ctx  # Store context separately
            }
            icm_examples.append(ICMExample(alt_input_text, alt_metadata))
    
    return icm_examples


def _convert_piqa(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert PIQA examples to ICM format with solution verification."""
    icm_examples = []
    
    for example in examples:
        goal = example.get("goal", "")
        sol1 = example.get("sol1", "")
        sol2 = example.get("sol2", "")
        
        solutions = [sol1, sol2] if sol1 and sol2 else []
        
        for i, solution in enumerate(solutions):
            # Main claim - solution achieves goal
            claim = f"Solution {i+1} achieves the goal"
            input_text = f"Goal: {goal}\nSolution {i+1}: {solution}\nClaim: {claim}\nI think this Claim is [True/False]"
            
            metadata = {
                "goal": goal,
                "solution": solution,
                "solution_index": i,
                "claim": claim,
                "task": "physical_reasoning",
                "response_text": f"Goal: {goal}\n\nSolution: {solution}"
            }
            icm_examples.append(ICMExample(input_text, metadata))
            
            # Alternative claim - practical feasibility
            alt_claim = f"This solution is practically feasible"
            alt_input_text = f"Goal: {goal}\nSolution: {solution}\nClaim: {alt_claim}\nI think this Claim is [True/False]"
            
            alt_metadata = {
                "goal": goal,
                "solution": solution,
                "solution_index": i,
                "claim": alt_claim,
                "task": "physical_reasoning",
                "response_text": f"Goal: {goal}\n\nSolution: {solution}"
            }
            icm_examples.append(ICMExample(alt_input_text, alt_metadata))
    
    return icm_examples


def _convert_arc_challenge(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert ARC-Challenge examples to ICM format with answer verification."""
    icm_examples = []
    
    for example in examples:
        question = example.get("question", "")
        choices = example.get("choices", {})
        
        # ARC choices are in format: {"text": [...], "label": [...]}
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        
        for i, (choice_text, choice_label) in enumerate(zip(choice_texts, choice_labels)):
            # Main claim - correctness
            claim = f"Answer {choice_label} is correct"
            input_text = f"Question: {question}\nAnswer {choice_label}: {choice_text}\nClaim: {claim}\nI think this Claim is [True/False]"
            
            metadata = {
                "question": question,
                "answer": choice_text,
                "answer_label": choice_label,
                "answer_index": i,
                "claim": claim,
                "task": "science_qa",
                "response_text": f"Answer {choice_label}: {choice_text}",  # Just the answer for clean DPO responses
                "prompt_text": question  # Store question separately
            }
            icm_examples.append(ICMExample(input_text, metadata))
            
            # Alternative claim - scientific validity
            alt_claim = f"This answer is scientifically valid"
            alt_input_text = f"Question: {question}\nAnswer: {choice_text}\nClaim: {alt_claim}\nI think this Claim is [True/False]"
            
            alt_metadata = {
                "question": question,
                "answer": choice_text,
                "answer_label": choice_label,
                "answer_index": i,
                "claim": alt_claim,
                "task": "science_qa",
                "response_text": f"Answer {choice_label}: {choice_text}",  # Just the answer for clean DPO responses
                "prompt_text": question  # Store question separately
            }
            icm_examples.append(ICMExample(alt_input_text, alt_metadata))
    
    return icm_examples


def _convert_winogrande(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert WinoGrande examples to ICM format with pronoun resolution verification."""
    icm_examples = []
    
    for example in examples:
        sentence = example.get("sentence", "")
        option1 = example.get("option1", "")
        option2 = example.get("option2", "")
        
        options = [option1, option2] if option1 and option2 else []
        
        for i, option in enumerate(options):
            # Main claim - pronoun resolution
            claim = f"Option {i+1} correctly resolves the pronoun reference"
            # Replace the underscore with the option for context
            filled_sentence = sentence.replace("_", option)
            input_text = f"Original: {sentence}\nWith Option {i+1}: {filled_sentence}\nClaim: {claim}\nI think this Claim is [True/False]"
            
            metadata = {
                "sentence": sentence,
                "option": option,
                "option_index": i,
                "filled_sentence": filled_sentence,
                "claim": claim,
                "task": "pronoun_resolution",
                "response_text": f"Sentence: {filled_sentence}"
            }
            icm_examples.append(ICMExample(input_text, metadata))
            
            # Alternative claim - semantic coherence
            alt_claim = f"This sentence makes semantic sense"
            alt_input_text = f"Sentence: {filled_sentence}\nClaim: {alt_claim}\nI think this Claim is [True/False]"
            
            alt_metadata = {
                "sentence": sentence,
                "option": option,
                "option_index": i,
                "filled_sentence": filled_sentence,
                "claim": alt_claim,
                "task": "pronoun_resolution",
                "response_text": f"Sentence: {filled_sentence}"
            }
            icm_examples.append(ICMExample(alt_input_text, alt_metadata))
    
    return icm_examples


def _convert_bigbench_hard(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert BIG-Bench Hard examples to ICM format with task-specific verification."""
    icm_examples = []
    
    for example in examples:
        # BIG-Bench Hard has various task formats
        input_text_raw = example.get("input", "")
        target = example.get("target", "")
        
        # Handle multiple choice if available
        if "multiple_choice_targets" in example:
            choices = example["multiple_choice_targets"]
            for i, choice in enumerate(choices):
                claim = f"Choice {i+1} is the correct answer"
                input_text = f"Problem: {input_text_raw}\nChoice {i+1}: {choice}\nClaim: {claim}\nI think this Claim is [True/False]"
                
                metadata = {
                    "problem": input_text_raw,
                    "choice": choice,
                    "choice_index": i,
                    "target": target,
                    "claim": claim,
                    "task": "reasoning",
                    "response_text": f"Choice {i+1}: {choice}",  # Just the choice for clean DPO responses
                    "prompt_text": input_text_raw  # Store problem separately
                }
                icm_examples.append(ICMExample(input_text, metadata))
        else:
            # Handle as open-ended reasoning
            claim = f"This answer correctly solves the problem"
            input_text = f"Problem: {input_text_raw}\nAnswer: {target}\nClaim: {claim}\nI think this Claim is [True/False]"
            
            metadata = {
                "problem": input_text_raw,
                "answer": target,
                "claim": claim,
                "task": "reasoning",
                "response_text": target,  # Just the answer for clean DPO responses
                "prompt_text": input_text_raw  # Store problem separately
            }
            icm_examples.append(ICMExample(input_text, metadata))
    
    return icm_examples


def _convert_ifeval(examples: List[Dict[str, Any]]) -> List[ICMExample]:
    """Convert IFEval examples to ICM format with instruction-following verification."""
    icm_examples = []
    
    for example in examples:
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        instruction_id_list = example.get("instruction_id_list", [])
        kwargs = example.get("kwargs", [])
        
        # Create verification claims for instruction following
        claim = f"This response correctly follows the given instruction"
        input_text = f"Instruction: {prompt}\nResponse: {response}\nClaim: {claim}\nI think this Claim is [True/False]"
        
        metadata = {
            "instruction": prompt,
            "response": response,
            "instruction_ids": instruction_id_list,
            "kwargs": kwargs,
            "claim": claim,
            "task": "instruction_following",
            "response_text": f"Instruction: {prompt}\n\nResponse: {response}"
        }
        icm_examples.append(ICMExample(input_text, metadata))
        
        # Alternative claim - completeness
        alt_claim = f"This response completely addresses the instruction"
        alt_input_text = f"Instruction: {prompt}\nResponse: {response}\nClaim: {alt_claim}\nI think this Claim is [True/False]"
        
        alt_metadata = {
            "instruction": prompt,
            "response": response,
            "instruction_ids": instruction_id_list,
            "kwargs": kwargs,
            "claim": alt_claim,
            "task": "instruction_following",
            "response_text": f"Instruction: {prompt}\n\nResponse: {response}"
        }
        icm_examples.append(ICMExample(alt_input_text, alt_metadata))
        
        # Create a contrasting poor response
        poor_response = "I don't understand the question."
        poor_claim = f"This response correctly follows the given instruction"
        poor_input_text = f"Instruction: {prompt}\nResponse: {poor_response}\nClaim: {poor_claim}\nI think this Claim is [True/False]"
        
        poor_metadata = {
            "instruction": prompt,
            "response": poor_response,
            "instruction_ids": instruction_id_list,
            "kwargs": kwargs,
            "claim": poor_claim,
            "task": "instruction_following",
            "response_text": f"Instruction: {prompt}\n\nResponse: {poor_response}"
        }
        icm_examples.append(ICMExample(poor_input_text, poor_metadata))
    
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
        num_examples: Number of base questions to generate (will be multiplied by diverse solutions)
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
            question = f"What is {a} + {b}?"
            
            # Use only the correct solution - let ICM discover if it's correct
            solution = f"{a} + {b} = {correct_answer}"
            
            # Create single example per question
            input_text = f"Question: {question}\nSolution: {solution}\nClaim: This solution is correct\nI think this Claim is [True/False]"
            examples.append(ICMExample(input_text, {"question": question, "solution": solution, "task": "math"}))
    
    elif task_type == "comparison":
        for i in range(num_examples):
            query = f"Which number is larger?"
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            
            # Generate diverse comparison claims
            claims = [
                "Response A is larger than Response B",
                "Response B is larger than Response A", 
                "Response A is equal to Response B",
                "Both responses are the same"
            ]
            
            for claim in claims:
                input_text = f"Query: {query}\nResponse A: {a}\nResponse B: {b}\nClaim: {claim}\nI think this Claim is [True/False]"
                examples.append(ICMExample(input_text, {
                    "query": query, 
                    "response_a": str(a), 
                    "response_b": str(b), 
                    "claim": claim, 
                    "task": "comparison",
                    "response_text": f"Query: {query}\n\nResponse A: {str(a)}\n\nResponse B: {str(b)}\n\nComparison: {claim}"
                }))
    
    return ICMDataset(examples, {"task_type": task_type, "synthetic": True})
