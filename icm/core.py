"""
Core ICM Algorithm Implementation.

This module implements the Internal Coherence Maximization algorithm with
mutual predictability scoring and simulated annealing search.
"""

import logging
import random
import math
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

from .consistency import LogicalConsistencyChecker
from .datasets import ICMDataset, ICMExample


@dataclass
class ICMResult:
    """Result from ICM search containing labeled examples and metadata."""
    labeled_examples: List[Dict[str, Any]]
    score: float
    iterations: int
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ICMSearcher:
    """
    Internal Coherence Maximization searcher.
    
    Implements the ICM algorithm using mutual predictability and logical consistency
    to generate labels for unlabeled datasets without external supervision.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        alpha: float = 100.0,
        initial_temperature: float = 3.0,
        final_temperature: float = 0.001,
        cooling_rate: float = 0.98,
        initial_examples: int = 20,
        max_iterations: int = 1000,
        consistency_fix_iterations: int = 10,
        generation_temperature: float = 0.2,
        generation_top_p: float = 0.9,
        generation_max_tokens: int = 512,
        consistency_checker: Optional[LogicalConsistencyChecker] = None,
        confidence_threshold: float = 0.1,
        seed: int = 42,
        log_level: str = "INFO"
    ):
        """
        Initialize ICM searcher.
        
        Args:
            model_name: Name or path of the model to use
            device: Device to run on ("cuda", "mps", "cpu", "auto", or None for auto-detection)
            alpha: Weight for mutual predictability vs consistency
            initial_temperature: Starting temperature for simulated annealing
            final_temperature: Ending temperature for simulated annealing
            cooling_rate: Rate of temperature cooling
            initial_examples: Number of initial randomly labeled examples (K)
            max_iterations: Maximum number of search iterations
            consistency_fix_iterations: Max iterations for consistency fixing
            generation_temperature: Temperature for text generation
            generation_top_p: Top-p for text generation
            generation_max_tokens: Max tokens for generation
            consistency_checker: Custom consistency checker
            confidence_threshold: Minimum confidence to label an example (0-1)
            seed: Random seed
            log_level: Logging level
        """
        self.model_name = model_name
        
        # Smart device selection with priority: CUDA > MPS > CPU
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.alpha = alpha
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.initial_examples = initial_examples
        self.max_iterations = max_iterations
        self.consistency_fix_iterations = consistency_fix_iterations
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.generation_max_tokens = generation_max_tokens
        self.confidence_threshold = confidence_threshold
        self.seed = seed
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Log device selection
        available_devices = []
        if torch.cuda.is_available():
            available_devices.append("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append("mps")
        available_devices.append("cpu")
        
        self.logger.info(f"Device selection: {self.device}")
        self.logger.info(f"Available devices: {', '.join(available_devices)}")
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Setup device optimizations
        if self.device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            # Enable TensorFloat-32 for better performance on Ampere+ GPUs
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere (A100, RTX 30xx) or newer
                torch.set_float32_matmul_precision('high')
                self.logger.info("Enabled TensorFloat-32 (TF32) for improved performance")
            
            self.logger.info("Applied CUDA optimizations")
        
        # Load model and tokenizer
        self.logger.info(f"Loading model {model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set a reasonable max_length to avoid truncation warnings
        if not hasattr(self.tokenizer, 'model_max_length') or self.tokenizer.model_max_length > 100000:
            self.tokenizer.model_max_length = 2048  # Reasonable default
        
        # Configure model loading based on device
        model_kwargs = {}
        if self.device == "cuda":
            # Only use device_map for large models that need multi-GPU
            # Check model size from config to decide
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            # Use the dtype specified in model config if available
            if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
                # Model has a preferred dtype (e.g., bfloat16 for Gemma-3-270M)
                if config.torch_dtype == torch.bfloat16:
                    # Check if bfloat16 is supported
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        model_kwargs["torch_dtype"] = torch.bfloat16
                        self.logger.info(f"Using bfloat16 as specified by model config")
                    else:
                        # Fallback to float32 if bfloat16 not supported
                        model_kwargs["torch_dtype"] = torch.float32
                        self.logger.warning(f"Model prefers bfloat16 but it's not supported, using float32")
                else:
                    model_kwargs["torch_dtype"] = config.torch_dtype
                    self.logger.info(f"Using {config.torch_dtype} as specified by model config")
            else:
                # Default to float16 for CUDA if no preference specified
                model_kwargs["torch_dtype"] = torch.float16
            
            # Estimate model size: params = hidden_size * layers * 4 (rough estimate)
            # Models under 1B params can typically fit on a single GPU
            estimated_params = getattr(config, 'num_parameters', None)
            
            if estimated_params is None:
                # Estimate based on config attributes if num_parameters not available
                hidden_size = getattr(config, 'hidden_size', 768)
                num_layers = getattr(config, 'num_hidden_layers', 12)
                estimated_params = hidden_size * num_layers * 4 * 1000  # Rough estimate
            
            # Only use device_map for models > 1B parameters
            if estimated_params > 1_000_000_000:
                model_kwargs["device_map"] = "auto"
                self.logger.info(f"Using device_map='auto' for large model ({estimated_params:,} params)")
        elif self.device == "mps":
            # Special handling for Gemma models on MPS - they need fp32 to avoid NaN/inf errors
            if "gemma" in model_name.lower():
                model_kwargs["torch_dtype"] = torch.float32
                self.logger.info("Using float32 for Gemma model on MPS to avoid numerical instability")
            else:
                model_kwargs["torch_dtype"] = torch.float16  # MPS supports float16 for other models
        else:  # CPU
            model_kwargs["torch_dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Get vocabulary size from model config
        self.vocab_size = self.model.config.vocab_size
        self.logger.info(f"Model vocabulary size: {self.vocab_size}")
        
        # Move model to device if not using device_map
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        
        # Initialize consistency checker
        self.consistency_checker = consistency_checker or LogicalConsistencyChecker()
        
        self.logger.info("ICM Searcher initialized successfully")
    
    def search(
        self,
        dataset: ICMDataset,
        task_type: str = "classification",
        max_examples: Optional[int] = None
    ) -> ICMResult:
        """
        Run ICM search on a dataset.
        
        Args:
            dataset: ICM dataset to search on
            task_type: Type of task (classification, comparison, etc.)
            max_examples: Maximum number of examples to process
        
        Returns:
            ICMResult with labeled examples and metadata
        """
        self.logger.info(f"Starting ICM search on {len(dataset)} examples")
        
        # Use all examples since sampling is now handled at dataset loading level
        examples = dataset.examples
        
        # Apply max_examples limit if specified
        if max_examples and max_examples < len(examples):
            self.logger.info(f"Limiting to {max_examples} examples for labeling")
            examples = examples[:max_examples]
        
        # Initialize with K randomly labeled examples
        labeled_data = self._initialize_labeled_data(examples, task_type)
        
        # Run consistency fix on initial data
        labeled_data = self._fix_inconsistencies(labeled_data)
        
        # Main search loop
        best_score = self._calculate_score(labeled_data)
        temperature = self.initial_temperature
        
        self.logger.info(f"Initial score: {best_score:.4f}")
        
        try:
            for iteration in tqdm(range(self.max_iterations), desc="ICM Search"):
                try:
                    # Update temperature
                    temperature = max(
                        self.final_temperature,
                        self.initial_temperature / (1 + self.cooling_rate * math.log(iteration + 1))
                    )
                    
                    # Sample new example to label
                    example_idx = self._sample_example_to_label(examples, labeled_data)
                    example = examples[example_idx]
                    
                    # Generate label for the example
                    new_label = self._generate_label(example, labeled_data, task_type)
                    
                    # Calculate confidence for this labeling decision (for logging only)
                    confidence = self._calculate_example_confidence(
                        example_idx, example, new_label, labeled_data, best_score
                    )
                    
                    # Create new labeled data with the proposed label
                    new_labeled_data = labeled_data.copy()
                    new_labeled_data[example_idx] = {
                        "example": example,
                        "label": new_label,
                        "index": example_idx
                    }
                    
                    # Fix inconsistencies
                    new_labeled_data = self._fix_inconsistencies(new_labeled_data)
                    
                    # Calculate new score
                    new_score = self._calculate_score(new_labeled_data)
                    
                    # Accept or reject based on simulated annealing
                    delta = new_score - best_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temperature):
                        labeled_data = new_labeled_data
                        best_score = new_score
                        self.logger.debug(f"Iteration {iteration}: Accepted example {example_idx}, confidence {confidence:.3f}, score = {best_score:.4f}")
                    else:
                        self.logger.debug(f"Iteration {iteration}: Rejected, score = {new_score:.4f}")
                    
                    # Progress logging
                    if iteration > 0 and iteration % 100 == 0:
                        self.logger.info(f"Iteration {iteration}: score = {best_score:.4f}, temp = {temperature:.6f}, labeled = {len(labeled_data)}/{len(examples)}")
                        
                    # Check if all examples are labeled
                    if len(labeled_data) >= len(examples):
                        self.logger.info(f"All {len(examples)} examples labeled. Stopping early.")
                        break
                        
                    # Early stopping if temperature is very low and no improvement
                    if iteration > 100 and temperature < self.final_temperature * 2:
                        if iteration % 50 == 0:
                            self.logger.debug(f"Low temperature check at iteration {iteration}")
                            
                except Exception as iteration_error:
                    self.logger.error(f"Error in iteration {iteration}: {iteration_error}")
                    self.logger.error(f"Detailed traceback: {traceback.format_exc()}")
                    # Continue with next iteration instead of crashing
                    continue
                    
        except Exception as search_error:
            self.logger.error(f"Critical error in search loop: {search_error}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise search_error
        
        # Convert to final format
        labeled_examples = []
        for idx, data in labeled_data.items():
            labeled_examples.append({
                "input": data["example"].input_text,
                "label": data["label"],
                "metadata": data["example"].metadata
            })
        
        result = ICMResult(
            labeled_examples=labeled_examples,
            score=best_score,
            iterations=self.max_iterations,
            convergence_info={
                "final_temperature": temperature,
                "labeled_count": len(labeled_data)
            },
            metadata={
                "model_name": self.model_name,
                "alpha": self.alpha,
                "task_type": task_type,
                "dataset_size": len(examples)
            }
        )
        
        self.logger.info(f"ICM search completed. Final score: {best_score:.4f}")
        return result
    
    def _initialize_labeled_data(
        self, 
        examples: List[ICMExample], 
        task_type: str
    ) -> Dict[int, Dict[str, Any]]:
        """Initialize with K randomly labeled examples."""
        labeled_data = {}
        
        # Randomly select K examples to label
        selected_indices = random.sample(range(len(examples)), min(self.initial_examples, len(examples)))
        
        for idx in selected_indices:
            example = examples[idx]
            # Generate random label based on task type
            if task_type == "classification":
                label = random.choice(["True", "False"])
            elif task_type == "comparison":
                label = random.choice(["True", "False"])
            else:
                label = random.choice(["True", "False"])  # Default binary
            
            labeled_data[idx] = {
                "example": example,
                "label": label,
                "index": idx
            }
        
        self.logger.info(f"Initialized with {len(labeled_data)} randomly labeled examples")
        return labeled_data
    
    def _sample_example_to_label(
        self, 
        examples: List[ICMExample], 
        labeled_data: Dict[int, Dict[str, Any]]
    ) -> int:
        """Sample an example to label, prioritizing unlabeled examples with consistency relationships."""
        unlabeled_indices = [i for i in range(len(examples)) if i not in labeled_data]
        
        if not unlabeled_indices:
            # If all examples are labeled, sample from existing ones to potentially relabel
            return random.choice(list(labeled_data.keys()))
        
        # Calculate weights for each unlabeled example (memory-efficient approach)
        weights = []
        
        for idx in unlabeled_indices:
            base_weight = 1.0
            relationship_count = 0
            
            # Count consistency relationships with labeled examples
            for labeled_idx in labeled_data:
                if self.consistency_checker.has_relationship(examples[idx], examples[labeled_idx]):
                    relationship_count += 1
            
            # Use additive weighting instead of multiplicative to prevent memory explosion
            # Each relationship adds significant weight but doesn't multiply exponentially
            if relationship_count > 0:
                weight = base_weight + (relationship_count * 50.0)  # 50x boost per relationship
            else:
                weight = base_weight
            
            weights.append(weight)
        
        # Memory-efficient weighted random sampling using cumulative distribution
        if not weights:
            return random.choice(unlabeled_indices)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(unlabeled_indices)
        
        # Generate random number and find corresponding index
        r = random.random() * total_weight
        cumulative_weight = 0.0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return unlabeled_indices[i]
        
        # Fallback (shouldn't reach here, but just in case)
        return unlabeled_indices[-1]
    
    def _generate_label(
        self, 
        example: ICMExample, 
        labeled_data: Dict[int, Dict[str, Any]], 
        task_type: str
    ) -> str:
        """Generate label for an example based on mutual predictability."""
        # Create context from other labeled examples
        context_examples = []
        for data in labeled_data.values():
            context_examples.append({
                "input": data["example"].input_text,
                "label": data["label"]
            })
        
        # Create prompt for label prediction
        prompt = self._build_prediction_prompt(example, context_examples, task_type)
        
        # Generate label using the model
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=min(1024, self.tokenizer.model_max_length)  # Explicit max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short for labels
                    temperature=self.generation_temperature,
                    top_p=self.generation_top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Extract label from generated text
            label = self._extract_label(generated_text, task_type)
            return label
            
        except Exception as e:
            self.logger.warning(f"Error generating label: {e}")
            # Fallback to random label
            return random.choice(["True", "False"])
    
    def _build_prediction_prompt(
        self, 
        example: ICMExample, 
        context_examples: List[Dict[str, Any]], 
        task_type: str
    ) -> str:
        """Build prompt for label prediction with mathematical reasoning."""
        prompt_parts = []
        
        # Add instruction for mathematical reasoning
        if "math" in str(task_type).lower() or any("+" in ctx['input'] or "=" in ctx['input'] for ctx in context_examples):
            prompt_parts.append("Evaluate whether each mathematical claim is correct or incorrect.")
            prompt_parts.append("")
        
        # Add context examples
        for ctx in context_examples:
            # Extract and format for better mathematical understanding
            input_text = ctx['input']
            if "Claim:" in input_text and "=" in input_text:
                # For math problems, emphasize the mathematical evaluation
                prompt_parts.append(f"Problem: {input_text}")
                prompt_parts.append(f"Evaluation: The claim is {ctx['label']}")
            else:
                prompt_parts.append(f"Input: {input_text}")
                prompt_parts.append(f"Label: {ctx['label']}")
            prompt_parts.append("")
        
        # Add target example with mathematical framing
        if "Claim:" in example.input_text and "=" in example.input_text:
            prompt_parts.append(f"Problem: {example.input_text}")
            prompt_parts.append("Evaluation: The claim is")
        else:
            prompt_parts.append(f"Input: {example.input_text}")
            prompt_parts.append("Label:")
        
        return "\n".join(prompt_parts)
    
    def _extract_label(self, generated_text: str, task_type: str) -> str:
        """Extract label from generated text."""
        text = generated_text.strip().lower()
        
        # Look for true/false patterns
        if "true" in text:
            return "True"
        elif "false" in text:
            return "False"
        else:
            # Default fallback
            return random.choice(["True", "False"])
    
    def _fix_inconsistencies(self, labeled_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Fix logical inconsistencies in labeled data (Algorithm 2)."""
        for iteration in range(self.consistency_fix_iterations):
            inconsistencies = self._find_inconsistencies(labeled_data)
            
            if not inconsistencies:
                break
            
            # Sample a random inconsistent pair
            pair_idx1, pair_idx2 = random.choice(inconsistencies)
            
            # Get current labels
            example1 = labeled_data[pair_idx1]["example"]
            example2 = labeled_data[pair_idx2]["example"]
            current_label1 = labeled_data[pair_idx1]["label"]
            current_label2 = labeled_data[pair_idx2]["label"]
            
            # Generate all consistent label combinations
            consistent_options = self.consistency_checker.get_consistent_options(
                example1, example2, current_label1, current_label2
            )
            
            if not consistent_options:
                continue
            
            # Evaluate each option and select the best
            best_score = -float('inf')
            best_option = None
            
            for label1, label2 in consistent_options:
                # Create temporary labeled data with this option
                temp_data = labeled_data.copy()
                temp_data[pair_idx1]["label"] = label1
                temp_data[pair_idx2]["label"] = label2
                
                score = self._calculate_score(temp_data)
                if score > best_score:
                    best_score = score
                    best_option = (label1, label2)
            
            # Apply the best option if it improves the score
            if best_option and best_score > self._calculate_score(labeled_data):
                labeled_data[pair_idx1]["label"] = best_option[0]
                labeled_data[pair_idx2]["label"] = best_option[1]
        
        return labeled_data
    
    def _find_inconsistencies(self, labeled_data: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Find pairs of examples with inconsistent labels."""
        inconsistencies = []
        
        indices = list(labeled_data.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                example1 = labeled_data[idx1]["example"]
                example2 = labeled_data[idx2]["example"]
                label1 = labeled_data[idx1]["label"]
                label2 = labeled_data[idx2]["label"]
                
                if not self.consistency_checker.check_consistency(example1, example2, label1, label2):
                    inconsistencies.append((idx1, idx2))
        
        return inconsistencies
    
    def _calculate_score(self, labeled_data: Dict[int, Dict[str, Any]]) -> float:
        """Calculate the ICM scoring function U(D) = α * P_θ(D) - I(D)."""
        if not labeled_data:
            return 0.0
        
        # Calculate mutual predictability
        mutual_predictability = self._calculate_mutual_predictability(labeled_data)
        
        # Calculate inconsistency penalty
        inconsistency_penalty = len(self._find_inconsistencies(labeled_data))
        
        # Combine scores
        score = self.alpha * mutual_predictability - inconsistency_penalty
        
        return score
    
    def _calculate_mutual_predictability(self, labeled_data: Dict[int, Dict[str, Any]]) -> float:
        """Calculate mutual predictability P_θ(D)."""
        if len(labeled_data) < 2:
            return 0.0
        
        total_log_prob = 0.0
        count = 0
        
        # For each labeled example, calculate probability given other examples
        for target_idx, target_data in labeled_data.items():
            # Create context from all other examples
            context_examples = []
            for idx, data in labeled_data.items():
                if idx != target_idx:
                    context_examples.append({
                        "input": data["example"].input_text,
                        "label": data["label"]
                    })
            
            if not context_examples:
                continue
            
            # Calculate probability of target label given context
            log_prob = self._calculate_conditional_probability(
                target_data["example"],
                target_data["label"],
                context_examples
            )
            
            total_log_prob += log_prob
            count += 1
        
        return total_log_prob / count if count > 0 else 0.0
    
    def _calculate_conditional_probability(
        self, 
        target_example: ICMExample, 
        target_label: str, 
        context_examples: List[Dict[str, Any]]
    ) -> float:
        """Calculate log P(target_label | target_example, context_examples)."""
        try:
            # Build prompt with context
            prompt = self._build_prediction_prompt(target_example, context_examples, "classification")
            
            # Tokenize with explicit max_length
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=min(1024, self.tokenizer.model_max_length)
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get probabilities for True/False tokens with multiple variants
            true_tokens = []
            false_tokens = []
            
            # Try multiple token variants
            for variant in ["True", "true", "TRUE", "correct", "yes"]:
                try:
                    token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    if token_ids:
                        true_tokens.append(token_ids[0])
                except:
                    pass
            
            for variant in ["False", "false", "FALSE", "incorrect", "no"]:
                try:
                    token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    if token_ids:
                        false_tokens.append(token_ids[0])
                except:
                    pass
            
            # Get maximum logit for each category with bounds checking
            # Verify logits tensor matches expected vocabulary size
            logits_vocab_size = logits.size(-1)
            if logits_vocab_size != self.vocab_size:
                self.logger.warning(f"Logits size {logits_vocab_size} doesn't match model vocab size {self.vocab_size}")

            # Use the minimum of actual logits size and expected vocab size for safety
            effective_vocab_size = min(logits_vocab_size, self.vocab_size)

            # Filter tokens to ensure they're within valid range
            valid_true_tokens = [tid for tid in true_tokens if 0 <= tid < effective_vocab_size]
            valid_false_tokens = [tid for tid in false_tokens if 0 <= tid < effective_vocab_size]

            # Debug logging if tokens are filtered
            if len(true_tokens) > len(valid_true_tokens):
                filtered = [t for t in true_tokens if t >= effective_vocab_size]
                self.logger.debug(f"Filtered out-of-bounds true tokens: {filtered}")
            if len(false_tokens) > len(valid_false_tokens):
                filtered = [t for t in false_tokens if t >= effective_vocab_size]
                self.logger.debug(f"Filtered out-of-bounds false tokens: {filtered}")
            
            # Convert Python integers to tensor indices on correct device for CUDA compatibility
            if valid_true_tokens:
                # Convert to tensor on same device as logits for proper indexing
                token_indices = torch.tensor(valid_true_tokens, device=logits.device, dtype=torch.long)
                true_logit = logits[token_indices].max().item()
            else:
                true_logit = -float('inf')

            if valid_false_tokens:
                token_indices = torch.tensor(valid_false_tokens, device=logits.device, dtype=torch.long)
                false_logit = logits[token_indices].max().item()
            else:
                false_logit = -float('inf')
            
            # Apply softmax
            exp_true = math.exp(true_logit)
            exp_false = math.exp(false_logit)
            total = exp_true + exp_false
            
            if target_label == "True":
                prob = exp_true / total
            else:
                prob = exp_false / total
            
            # Return log probability
            return math.log(max(prob, 1e-10))  # Avoid log(0)
        
        except Exception as e:
            self.logger.warning(f"Error calculating conditional probability: {e}")
            return -10.0  # Large negative log probability

    def _calculate_example_confidence(
        self, 
        example_idx: int,
        example: ICMExample,
        new_label: str,
        labeled_data: Dict[int, Dict[str, Any]],
        current_score: float
    ) -> float:
        """
        Calculate confidence for labeling a specific example.
        
        Confidence is based on how much the mutual predictability improves
        when we add this example with the proposed label.
        
        Returns:
            float: Confidence score (0 to 1), higher is more confident
        """
        try:
            # Create temporary labeled data with the new example
            temp_labeled_data = labeled_data.copy()
            temp_labeled_data[example_idx] = {
                "example": example,
                "label": new_label,
                "index": example_idx
            }
            
            # Calculate the new mutual predictability
            new_mutual_predictability = self._calculate_mutual_predictability(temp_labeled_data)
            
            # Calculate current mutual predictability (without this example)
            current_mutual_predictability = self._calculate_mutual_predictability(labeled_data) if labeled_data else 0.0
            
            # Confidence is the improvement in mutual predictability, normalized
            improvement = new_mutual_predictability - current_mutual_predictability
            
            # Normalize based on the scale of current mutual predictability
            # Use a reasonable normalization factor that doesn't depend on alpha
            normalization = max(1.0, abs(current_mutual_predictability) * 0.1) if current_mutual_predictability != 0 else 1.0
            confidence = max(0.0, min(1.0, improvement / normalization))
            
            # Log the confidence calculation for debugging
            self.logger.debug(f"Confidence calculation: improvement={improvement:.4f}, normalization={normalization:.4f}, confidence={confidence:.4f}")
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Error calculating example confidence: {e}")
            return 0.0  # Low confidence on error
