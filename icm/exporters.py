"""
Export utilities for ICM results.

This module provides functionality to export ICM results to various formats
and push them to Hugging Face Hub.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .storage import ICMStorage


class ICMExporter:
    """Exporter for ICM results to various formats."""
    
    def __init__(self, storage: Optional[ICMStorage] = None):
        """
        Initialize exporter.
        
        Args:
            storage: ICM storage instance
        """
        self.storage = storage or ICMStorage()
        self.logger = logging.getLogger(__name__)
    
    def export_to_huggingface(
        self,
        labeled_examples: List[Dict[str, Any]],
        repo_id: str,
        task_type: str,
        model_name: str,
        private: bool = False,
        create_readme: bool = True
    ) -> str:
        """
        Export labeled examples to Hugging Face Hub.
        
        Args:
            labeled_examples: List of labeled examples
            repo_id: Hugging Face repository ID
            task_type: Type of task
            model_name: Model used for generation
            private: Whether to make repo private
            create_readme: Whether to create README
            
        Returns:
            URL to the uploaded dataset
        """
        try:
            from huggingface_hub import create_repo, upload_file
            from datasets import Dataset
        except ImportError:
            raise ImportError("huggingface_hub and datasets are required for HF export")
        
        self.logger.info(f"Exporting to Hugging Face: {repo_id}")
        
        # Create repository
        create_repo(
            repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        
        # Prepare data for HF Dataset
        inputs = [ex["input"] for ex in labeled_examples]
        labels = [ex["label"] for ex in labeled_examples]
        metadata = [ex.get("metadata", {}) for ex in labeled_examples]
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input": inputs,
            "label": labels,
            "metadata": metadata
        })
        
        # Save dataset locally first
        temp_dir = "temp_hf_dataset"
        os.makedirs(temp_dir, exist_ok=True)
        dataset.save_to_disk(temp_dir)
        
        # Upload dataset files
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
        
        # Create and upload README
        if create_readme:
            readme_content = self._generate_readme(
                labeled_examples, task_type, model_name
            )
            
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset"
            )
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        url = f"https://huggingface.co/datasets/{repo_id}"
        self.logger.info(f"Successfully exported to {url}")
        return url
    
    def export_to_json(
        self,
        labeled_examples: List[Dict[str, Any]],
        output_path: str,
        include_stats: bool = True
    ) -> str:
        """
        Export to JSON format.
        
        Args:
            labeled_examples: List of labeled examples
            output_path: Output file path
            include_stats: Whether to include statistics
            
        Returns:
            Path to exported file
        """
        export_data = {
            "examples": labeled_examples,
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "num_examples": len(labeled_examples),
                "exporter": "ICM"
            }
        }
        
        if include_stats:
            export_data["statistics"] = self._calculate_export_stats(labeled_examples)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported to JSON: {output_path}")
        return output_path
    
    def export_to_dpo_format(
        self,
        labeled_examples: List[Dict[str, Any]],
        output_path: str,
        create_pairs: bool = True
    ) -> str:
        """
        Export to DPO training format.
        Creates preferred/rejected pairs from ICM labels:
        - True solutions (ICM labeled) = Preferred responses
        - False solutions (ICM labeled) = Rejected responses
        
        Args:
            labeled_examples: List of labeled examples
            output_path: Output file path
            create_pairs: Whether to create chosen/rejected pairs
            
        Returns:
            Path to exported file
        """
        dpo_examples = []
        
        if create_pairs:
            # Group by question to create pairs from same question
            question_groups = {}
            for ex in labeled_examples:
                # Extract question from metadata or input
                question = ex.get("metadata", {}).get("question", "")
                if not question:
                    # Fallback: extract question from input text
                    input_text = ex["input"]
                    if "Question:" in input_text:
                        question = input_text.split("Question:")[1].split("\n")[0].strip()
                    else:
                        question = input_text.split("\n")[0].strip()
                
                if question not in question_groups:
                    question_groups[question] = []
                question_groups[question].append(ex)
            
            # Create preferred/rejected pairs from each question group
            for question, examples in question_groups.items():
                true_examples = [ex for ex in examples if ex["label"] == "True"]
                false_examples = [ex for ex in examples if ex["label"] == "False"]
                
                # Create all possible (preferred, rejected) pairs
                for true_ex in true_examples:  # Preferred (correct solutions)
                    for false_ex in false_examples:  # Rejected (incorrect solutions)
                        # Extract the response from metadata
                        preferred_solution = true_ex.get("metadata", {}).get("response_text", "")
                        rejected_solution = false_ex.get("metadata", {}).get("response_text", "")
                        
                        # Skip if chosen and rejected are identical (not a valid preference pair)
                        if preferred_solution != rejected_solution:
                            dpo_example = {
                                "prompt": question,  # The mathematical question
                                "chosen": preferred_solution,  # ICM-labeled True solution
                                "rejected": rejected_solution,  # ICM-labeled False solution
                                "chosen_metadata": true_ex.get("metadata", {}),
                                "rejected_metadata": false_ex.get("metadata", {})
                            }
                            dpo_examples.append(dpo_example)
        else:
            # Simple format - create pairs within same question groups
            # Group by question first
            question_groups = {}
            for ex in labeled_examples:
                question = ex.get("metadata", {}).get("question", "")
                if not question:
                    # Fallback: extract question from input text
                    input_text = ex["input"]
                    if "Question:" in input_text:
                        question = input_text.split("Question:")[1].split("\n")[0].strip()
                    else:
                        question = input_text.split("\n")[0].strip()
                
                if question not in question_groups:
                    question_groups[question] = []
                question_groups[question].append(ex)
            
            # For each question, find a preferred and rejected solution
            for question, examples in question_groups.items():
                true_examples = [ex for ex in examples if ex["label"] == "True"]
                false_examples = [ex for ex in examples if ex["label"] == "False"]
                
                # If we have both true and false examples, create a pair
                if true_examples and false_examples:
                    preferred_solution = true_examples[0].get("metadata", {}).get("response_text", "")
                    rejected_solution = false_examples[0].get("metadata", {}).get("response_text", "")
                    
                    # Skip if chosen and rejected are identical (not a valid preference pair)
                    if preferred_solution != rejected_solution:
                        dpo_example = {
                            "prompt": question,
                            "chosen": preferred_solution,
                            "rejected": rejected_solution
                        }
                        dpo_examples.append(dpo_example)
        
        with open(output_path, 'w') as f:
            for example in dpo_examples:
                f.write(json.dumps(example) + '\n')
        
        self.logger.info(f"Exported {len(dpo_examples)} DPO pairs to: {output_path}")
        return output_path
    
    
    def export_to_csv(
        self,
        labeled_examples: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Export to CSV format.
        
        Args:
            labeled_examples: List of labeled examples
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        import csv
        
        if not labeled_examples:
            raise ValueError("No examples to export")
        
        # Determine all possible metadata fields
        metadata_fields = set()
        for ex in labeled_examples:
            metadata_fields.update(ex.get("metadata", {}).keys())
        
        fieldnames = ["input", "label"] + sorted(metadata_fields)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for ex in labeled_examples:
                row = {
                    "input": ex["input"],
                    "label": ex["label"]
                }
                # Add metadata fields
                for field in metadata_fields:
                    row[field] = ex.get("metadata", {}).get(field, "")
                
                writer.writerow(row)
        
        self.logger.info(f"Exported to CSV: {output_path}")
        return output_path
    
    def export_analysis_report(
        self,
        labeled_examples: List[Dict[str, Any]],
        output_path: str,
        include_examples: bool = True
    ) -> str:
        """
        Export analysis report.
        
        Args:
            labeled_examples: List of labeled examples
            output_path: Output file path
            include_examples: Whether to include example details
            
        Returns:
            Path to exported file
        """
        stats = self._calculate_export_stats(labeled_examples)
        
        report = {
            "summary": {
                "total_examples": len(labeled_examples),
                "generation_timestamp": datetime.now().isoformat(),
                "icm_version": "0.1.0"
            },
            "label_distribution": stats["label_distribution"],
            "statistics": stats,
        }
        
        if include_examples:
            report["examples"] = labeled_examples[:10]  # Include first 10 examples
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Exported analysis report: {output_path}")
        return output_path
    
    def _calculate_export_stats(self, labeled_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for labeled examples."""
        if not labeled_examples:
            return {}
        
        # Label distribution
        label_counts = {}
        for ex in labeled_examples:
            label = ex["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Input length statistics
        input_lengths = [len(ex["input"]) for ex in labeled_examples]
        
        # Task distribution
        task_counts = {}
        for ex in labeled_examples:
            task = ex.get("metadata", {}).get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
        
        stats = {
            "label_distribution": label_counts,
            "input_length_stats": {
                "min": min(input_lengths),
                "max": max(input_lengths),
                "avg": sum(input_lengths) / len(input_lengths),
                "median": sorted(input_lengths)[len(input_lengths) // 2]
            },
            "task_distribution": task_counts
        }
        
        return stats
    
    def _generate_readme(
        self,
        labeled_examples: List[Dict[str, Any]],
        task_type: str,
        model_name: str
    ) -> str:
        """Generate README content for HF dataset."""
        stats = self._calculate_export_stats(labeled_examples)
        
        readme = f"""# ICM Generated Dataset

This dataset was generated using Internal Coherence Maximization (ICM), an unsupervised method for eliciting knowledge from language models.

## Dataset Info

- **Task Type**: {task_type}
- **Model Used**: {model_name}
- **Total Examples**: {len(labeled_examples)}
- **Generation Date**: {datetime.now().strftime('%Y-%m-%d')}

## Label Distribution

"""
        
        for label, count in stats.get("label_distribution", {}).items():
            percentage = (count / len(labeled_examples)) * 100
            readme += f"- **{label}**: {count} ({percentage:.1f}%)\n"
        
        readme += f"""
## Usage

```python
from datasets import load_dataset

dataset = load_dataset("path/to/this/dataset")
```

## Methodology

This dataset was created using the Internal Coherence Maximization (ICM) algorithm, which:

1. **Mutual Predictability**: Finds labels that are mutually predictable given the model's understanding
2. **Logical Consistency**: Enforces logical constraints to prevent degenerate solutions
3. **Simulated Annealing**: Uses temperature-based search to find optimal label assignments

## Citation

If you use this dataset, please cite the original ICM paper:

```bibtex
@article{{icm2024,
  title={{Unsupervised Elicitation of Language Models}},
  author={{Wen, Jiaxin and others}},
  journal={{arXiv preprint}},
  year={{2024}}
}}
```

## License

This dataset is released under the same license as the source data and model used for generation.
"""
        
        return readme


def extract_answer_from_response(response_text: str, metadata: Dict[str, Any]) -> str:
    """
    Extract only the answer portion from response_text, removing question repetition.
    
    Args:
        response_text: Full response text from ICM results
        metadata: Example metadata for fallback extraction
        
    Returns:
        Clean answer text without question repetition
    """
    if not response_text:
        return ""
    
    # Handle different response formats
    if "\n\nSolution:" in response_text:
        # GSM8K format: "Question: X\n\nSolution: Y"
        return response_text.split("\n\nSolution:", 1)[1].strip()
    elif "\n\nAnswer:" in response_text:
        # General format: "Problem: X\n\nAnswer: Y"
        return response_text.split("\n\nAnswer:", 1)[1].strip()
    elif "\n\nResponse:" in response_text:
        # IFEval format: "Instruction: X\n\nResponse: Y"
        return response_text.split("\n\nResponse:", 1)[1].strip()
    elif "\n\nChoice" in response_text:
        # BigBench multiple choice: "Problem: X\n\nChoice Y: Z"
        parts = response_text.split("\n\nChoice", 1)[1]
        return "Choice" + parts.strip()
    elif "\n\n" in response_text and ("Question:" in response_text or "Problem:" in response_text or "Instruction:" in response_text):
        # Generic format: "[Prefix]: X\n\nY"
        return response_text.split("\n\n", 1)[1].strip()
    else:
        # Fallback: try to use solution/answer from metadata if cleaner
        solution = metadata.get("solution", "")
        answer = metadata.get("answer", "")
        
        if solution and len(solution) < len(response_text) * 0.8:
            return solution
        elif answer and len(answer) < len(response_text) * 0.8:
            return answer
        else:
            # Last resort: return original response_text
            return response_text


def select_best_icm_files(result_files: List[str]) -> List[str]:
    """
    Select the highest quality ICM files based on balance ratio and other metrics.
    
    Args:
        result_files: List of all available ICM result files
        
    Returns:
        List of selected high-quality files
    """
    logger = logging.getLogger(__name__)
    
    # Group files by dataset
    files_by_dataset = {}
    for file in result_files:
        dataset_name = os.path.basename(file).split('_')[0]
        if dataset_name not in files_by_dataset:
            files_by_dataset[dataset_name] = []
        files_by_dataset[dataset_name].append(file)
    
    selected_files = []
    
    # Per-benchmark selection limits (to ensure balanced coverage)
    selection_limits = {
        'bigbenchhard': 8,  # Select best 8 out of 27 BigBench files
        'winogrande': 5,    # All 5 winogrande files (different sizes)
        'gsm8k': 2,         # Both gsm8k files (main + socratic)
        'ai2': 2,           # Both ARC files (challenge + easy) 
        'truthful': 2,      # Both truthful files
        'hellaswag': 1,     # Single hellaswag file
        'piqa': 1,          # Single piqa file
        'IFEval': 0         # Skip IFEval (empty responses)
    }
    
    for dataset_name, files in files_by_dataset.items():
        max_files = selection_limits.get(dataset_name, len(files))
        
        if max_files == 0:
            logger.info(f"Skipping {dataset_name} - no usable data (empty responses)")
            continue
        
        # Calculate quality metrics for each file
        file_qualities = []
        for file in files:
            try:
                with open(file, 'r') as f:
                    examples = [json.loads(line) for line in f]
                
                true_count = sum(1 for ex in examples if ex["label"] == "True")
                false_count = sum(1 for ex in examples if ex["label"] == "False")
                
                # Quality score based on balance ratio (prefer 0.7-1.0)
                if true_count > 0 and false_count > 0:
                    balance_ratio = min(true_count, false_count) / max(true_count, false_count)
                    usable_pairs = min(true_count, false_count)
                    
                    # Boost score for better balance and more pairs
                    quality_score = balance_ratio * (1 + usable_pairs / 1000)  
                else:
                    quality_score = 0  # No pairs possible
                
                file_qualities.append((file, quality_score, balance_ratio, usable_pairs))
                
            except Exception as e:
                logger.warning(f"Failed to analyze {file}: {e}")
                continue
        
        # Sort by quality score and select the best files
        file_qualities.sort(key=lambda x: x[1], reverse=True)
        selected_for_dataset = file_qualities[:max_files]
        
        for file, score, balance, pairs in selected_for_dataset:
            selected_files.append(file)
            logger.info(f"Selected {dataset_name}: {os.path.basename(file)} "
                       f"(balance: {balance:.2f}, pairs: {pairs}, quality: {score:.3f})")
    
    logger.info(f"Selected {len(selected_files)} high-quality files out of {len(result_files)} total")
    return selected_files


def combine_icm_results_to_dpo(
    result_files: List[str],
    output_path: str,
    storage: Optional[ICMStorage] = None,
    fix_responses: bool = True,
    balance_strategy: str = "none",
    max_per_benchmark: int = 1000
) -> str:
    """
    Combine multiple ICM result files into a single DPO dataset.
    
    Args:
        result_files: List of ICM result file paths
        output_path: Output file path for combined DPO dataset
        storage: ICM storage instance (optional)
        fix_responses: Whether to extract clean answers from response_text
        balance_strategy: "equal", "proportional", or "none"
        max_per_benchmark: Maximum examples per benchmark group
        
    Returns:
        Path to combined DPO dataset
    """
    logger = logging.getLogger(__name__)
    storage = storage or ICMStorage()
    
    all_dpo_examples = []
    dataset_sources = {}
    
    # Dataset balancing configuration
    BENCHMARK_GROUPS = {
        "hellaswag": "hellaswag",
        "piqa": "piqa", 
        "ai2": "arc",  # ai2_arc maps to arc benchmark group
        "winogrande": "winogrande",
        "bigbenchhard": "bigbench",
        "IFEval": "ifeval",
        "truthful": "truthfulqa",
        "gsm8k": "gsm8k"
    }
    
    # Track counts per benchmark group for balancing
    benchmark_counts = {}
    max_per_group = max_per_benchmark if balance_strategy != "none" else float('inf')
    
    # Statistics tracking for filtered pairs
    skipped_stats = {
        "too_short": 0,
        "identical": 0,
        "hardcoded_42": 0,  # Should be 0 with our fix, kept as safety check
        "missing_metadata": 0,
        "total_processed": 0
    }
    
    for result_file in result_files:
        logger.info(f"Loading ICM results from {result_file}")
        
        # Load the ICM result
        result = storage.load_result(result_file)
        
        # Check if we got valid labeled examples
        has_valid_result = (result and 
                           hasattr(result, 'labeled_examples') and 
                           len(result.labeled_examples) > 0)
        
        if has_valid_result:
            # Full ICM result with metadata
            labeled_examples = result.labeled_examples
            dataset_name = result.metadata.get("dataset", "unknown")
        else:
            # Try to load as raw labeled examples (current format)
            try:
                labeled_examples = []
                with open(result_file, 'r') as f:
                    for line in f:
                        example = json.loads(line)
                        # Raw examples don't have type field, just add them directly
                        labeled_examples.append(example)
                
                # Extract dataset name from filename
                import os
                filename = os.path.basename(result_file)
                # Extract dataset name (e.g., "hellaswag" from "hellaswag_gemma-3-270m-it_icm_20250818_065956.jsonl")
                dataset_name = filename.split('_')[0]
                
                logger.info(f"Loaded {len(labeled_examples)} raw examples from {dataset_name}")
            except Exception as e:
                logger.warning(f"Could not load {result_file}: {e}, skipping")
                continue
        
        if not labeled_examples:
            logger.warning(f"No examples found in {result_file}, skipping")
            continue
        
        # Determine benchmark group for balancing
        benchmark_group = BENCHMARK_GROUPS.get(dataset_name, dataset_name)
        
        # Apply balancing if enabled
        if balance_strategy != "none":
            current_count = benchmark_counts.get(benchmark_group, 0)
            if current_count >= max_per_group:
                logger.info(f"Skipping {dataset_name} - benchmark group '{benchmark_group}' already has {current_count} examples (max: {max_per_group})")
                continue
            
            # Limit examples to stay within budget
            remaining_budget = max_per_group - current_count
            if len(labeled_examples) > remaining_budget:
                logger.info(f"Limiting {dataset_name} from {len(labeled_examples)} to {remaining_budget} examples for balancing")
                labeled_examples = labeled_examples[:remaining_budget]
        
        # Track source dataset counts
        dataset_sources[dataset_name] = dataset_sources.get(dataset_name, 0) + len(labeled_examples)
        benchmark_counts[benchmark_group] = benchmark_counts.get(benchmark_group, 0) + len(labeled_examples)
        
        # Group by question to create pairs
        question_groups = {}
        for ex in labeled_examples:
            # Extract question from metadata or input
            question = ex.get("metadata", {}).get("question", "")
            if not question:
                # Try to extract from other fields
                question = ex.get("metadata", {}).get("goal", "")  # PIQA
                if not question:
                    question = ex.get("metadata", {}).get("context", "")  # HellaSwag
                if not question:
                    question = ex.get("metadata", {}).get("sentence", "")  # WinoGrande
                if not question:
                    question = ex.get("metadata", {}).get("instruction", "")  # IFEval
                if not question:
                    # Fallback: extract from input text
                    input_text = ex["input"]
                    if "Question:" in input_text:
                        question = input_text.split("Question:")[1].split("\n")[0].strip()
                    elif "Goal:" in input_text:
                        question = input_text.split("Goal:")[1].split("\n")[0].strip()
                    elif "Context:" in input_text:
                        question = input_text.split("Context:")[1].split("\n")[0].strip()
                    else:
                        question = input_text.split("\n")[0].strip()
            
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(ex)
        
        # Create DPO pairs from each question group
        logger.info(f"Processing {len(question_groups)} question groups for dataset {dataset_name}")
        
        for question, examples in question_groups.items():
            true_examples = [ex for ex in examples if ex["label"] == "True"]
            false_examples = [ex for ex in examples if ex["label"] == "False"]
            
            # Debug logging for pair creation
            if len(true_examples) == 0 and len(false_examples) == 0:
                logger.warning(f"Question '{question[:50]}...' has no valid examples")
            elif len(true_examples) == 0:
                logger.debug(f"Question '{question[:50]}...' has no True examples ({len(false_examples)} False)")
            elif len(false_examples) == 0:
                logger.debug(f"Question '{question[:50]}...' has no False examples ({len(true_examples)} True)")
            else:
                logger.debug(f"Question '{question[:50]}...' has {len(true_examples)} True and {len(false_examples)} False examples")
            
            # Create pairs (with limit to prevent single question dominating dataset)
            max_pairs_per_question = 5  # Limit pairs per question to prevent dominance
            pairs_created_for_question = 0
            
            for true_ex in true_examples:
                if pairs_created_for_question >= max_pairs_per_question:
                    break
                for false_ex in false_examples:
                    if pairs_created_for_question >= max_pairs_per_question:
                        break
                    skipped_stats["total_processed"] += 1
                    
                    # Extract solutions/answers - USE RESPONSE_TEXT FIELD ONLY
                    task_type = true_ex.get("metadata", {}).get("task", "")
                    
                    # Extract responses - use clean answer extraction if enabled
                    preferred_raw = true_ex.get("metadata", {}).get("response_text")
                    rejected_raw = false_ex.get("metadata", {}).get("response_text")
                    
                    if fix_responses:
                        # Extract only the answer portion, removing question repetition
                        preferred = extract_answer_from_response(preferred_raw or "", true_ex.get("metadata", {}))
                        rejected = extract_answer_from_response(rejected_raw or "", false_ex.get("metadata", {}))
                    else:
                        # Use original response_text (legacy behavior)
                        preferred = preferred_raw
                        rejected = rejected_raw
                    
                    # VALIDATION - SKIP PAIRS WITH ISSUES (DON'T FAIL)
                    if not preferred or not rejected:
                        skipped_stats["missing_metadata"] += 1
                        logger.debug(f"Skipping pair with missing response_text for task '{task_type}': "
                                   f"preferred={'✓' if preferred else '✗'}, rejected={'✓' if rejected else '✗'}")
                        continue
                    
                    # Check response length - SKIP if too short (adjust threshold by dataset)
                    # Skip IFEval entirely due to empty responses, allow short BigBench answers
                    if dataset_name == 'IFEval':
                        # Skip IFEval - has empty responses, can't generate meaningful DPO pairs
                        skipped_stats["too_short"] += 1
                        logger.debug(f"Skipping IFEval pair - dataset has empty responses")
                        continue
                    elif dataset_name == 'bigbenchhard':
                        # Allow very short BigBench answers (e.g., "valid", "invalid", single words)
                        min_length = 1
                    elif dataset_name in ['piqa']:
                        min_length = 15
                    else:
                        min_length = 30
                    
                    if len(preferred) < min_length or len(rejected) < min_length:
                        skipped_stats["too_short"] += 1
                        logger.debug(f"Skipping short response pair for task '{task_type}' (min {min_length}): "
                                   f"chosen={len(preferred)} chars, rejected={len(rejected)} chars")
                        continue
                    
                    # Check for identical responses - SKIP if same
                    if preferred == rejected:
                        skipped_stats["identical"] += 1
                        logger.debug(f"Skipping identical pair for task '{task_type}': "
                                   f"'{preferred[:50]}...'")
                        continue
                    
                    # Safety check - should never trigger with our fix
                    if "The answer is 42" in preferred or "The answer is 42" in rejected:
                        skipped_stats["hardcoded_42"] += 1
                        logger.warning(f"UNEXPECTED: Found hardcoded 'answer is 42' for task '{task_type}' - this shouldn't happen!")
                        logger.warning(f"  Chosen: '{preferred[:100]}...'")
                        logger.warning(f"  Rejected: '{rejected[:100]}...'")
                        continue
                    
                    # All validations passed - create DPO pair
                    dpo_example = {
                        "prompt": question,
                        "chosen": preferred,
                        "rejected": rejected,
                        "source_dataset": dataset_name,
                        "task_type": task_type
                    }
                    all_dpo_examples.append(dpo_example)
                    pairs_created_for_question += 1
    
    # Write combined DPO dataset
    with open(output_path, 'w') as f:
        for example in all_dpo_examples:
            f.write(json.dumps(example) + '\n')
    
    # Log comprehensive statistics
    logger.info(f"Combined DPO dataset created: {output_path}")
    logger.info(f"DPO Export Statistics:")
    logger.info(f"  Total pairs processed: {skipped_stats['total_processed']}")
    logger.info(f"  Valid pairs created: {len(all_dpo_examples)}")
    logger.info(f"  Success rate: {100*len(all_dpo_examples)/max(skipped_stats['total_processed'], 1):.1f}%")
    
    # Log skipped pairs
    total_skipped = sum(v for k, v in skipped_stats.items() if k != 'total_processed')
    if total_skipped > 0:
        logger.info(f"Skipped pairs breakdown:")
        logger.info(f"  - Too short (< min chars): {skipped_stats['too_short']} ({100*skipped_stats['too_short']/skipped_stats['total_processed']:.1f}%)")
        logger.info(f"  - Identical responses: {skipped_stats['identical']} ({100*skipped_stats['identical']/skipped_stats['total_processed']:.1f}%)")
        logger.info(f"  - Missing metadata: {skipped_stats['missing_metadata']} ({100*skipped_stats['missing_metadata']/skipped_stats['total_processed']:.1f}%)")
        
        if skipped_stats['hardcoded_42'] > 0:
            logger.error(f"  - UNEXPECTED hardcoded '42': {skipped_stats['hardcoded_42']} (this indicates a bug in generation!)")
        else:
            logger.info(f"  - Hardcoded '42' responses: 0 ✓ (as expected after our fix)")
    else:
        logger.info("✅ No pairs were skipped - all generated pairs met quality standards!")
    
    logger.info("Source datasets:")
    for dataset, count in dataset_sources.items():
        logger.info(f"  - {dataset}: {count} examples")
    
    if balance_strategy != "none":
        logger.info(f"\\nBalancing applied (strategy: {balance_strategy}, max per benchmark: {max_per_benchmark}):")
        logger.info("Benchmark group distribution:")
        total_balanced = 0
        for group, count in benchmark_counts.items():
            logger.info(f"  - {group}: {count} examples")
            total_balanced += count
        logger.info(f"Total balanced examples: {total_balanced}")
    
    if fix_responses:
        logger.info("✅ Response extraction applied - answers extracted from full response_text")
    else:
        logger.info("⚠️  Using original response_text format (includes question repetition)")
    
    return output_path


def push_to_huggingface(
    file_path: str,
    repo_id: str,
    file_name: Optional[str] = None,
    private: bool = False
) -> str:
    """
    Push a file to Hugging Face Hub.
    
    Args:
        file_path: Local file path
        repo_id: HF repository ID
        file_name: Name for file in repo (defaults to basename)
        private: Whether repo should be private
        
    Returns:
        URL to uploaded file
    """
    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError:
        raise ImportError("huggingface_hub is required for HF upload")
    
    logger = logging.getLogger(__name__)
    
    # Create repo if needed
    create_repo(
        repo_id,
        repo_type="dataset", 
        private=private,
        exist_ok=True
    )
    
    # Upload file
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name or os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Uploaded {file_path} to {url}")
    return url
