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
                        # Extract the solution from the input
                        preferred_solution = true_ex.get("metadata", {}).get("solution", "")
                        rejected_solution = false_ex.get("metadata", {}).get("solution", "")
                        
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
                    preferred_solution = true_examples[0].get("metadata", {}).get("solution", "")
                    rejected_solution = false_examples[0].get("metadata", {}).get("solution", "")
                    
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


def combine_icm_results_to_dpo(
    result_files: List[str],
    output_path: str,
    storage: Optional[ICMStorage] = None
) -> str:
    """
    Combine multiple ICM result files into a single DPO dataset.
    
    Args:
        result_files: List of ICM result file paths
        output_path: Output file path for combined DPO dataset
        storage: ICM storage instance (optional)
        
    Returns:
        Path to combined DPO dataset
    """
    logger = logging.getLogger(__name__)
    storage = storage or ICMStorage()
    
    all_dpo_examples = []
    dataset_sources = {}
    
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
        
        # Track source dataset counts
        dataset_sources[dataset_name] = dataset_sources.get(dataset_name, 0) + len(labeled_examples)
        
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
            
            # Create pairs
            for true_ex in true_examples:
                for false_ex in false_examples:
                    skipped_stats["total_processed"] += 1
                    
                    # Extract solutions/answers - USE RESPONSE_TEXT FIELD ONLY
                    task_type = true_ex.get("metadata", {}).get("task", "")
                    
                    # STRICT EXTRACTION - NO FALLBACKS TO EMPTY STRINGS
                    preferred = true_ex.get("metadata", {}).get("response_text")
                    rejected = false_ex.get("metadata", {}).get("response_text")
                    
                    # VALIDATION - SKIP PAIRS WITH ISSUES (DON'T FAIL)
                    if not preferred or not rejected:
                        skipped_stats["missing_metadata"] += 1
                        logger.debug(f"Skipping pair with missing response_text for task '{task_type}': "
                                   f"preferred={'✓' if preferred else '✗'}, rejected={'✓' if rejected else '✗'}")
                        continue
                    
                    # Check response length - SKIP if too short
                    if len(preferred) < 50 or len(rejected) < 50:
                        skipped_stats["too_short"] += 1
                        logger.debug(f"Skipping short response pair for task '{task_type}': "
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
        logger.info(f"  - Too short (<50 chars): {skipped_stats['too_short']} ({100*skipped_stats['too_short']/skipped_stats['total_processed']:.1f}%)")
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
