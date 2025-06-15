"""
Storage utilities for ICM results.

This module provides functionality to save and load ICM results,
including labeled datasets and search metadata.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .core import ICMResult


class ICMStorage:
    """Storage manager for ICM results."""
    
    def __init__(self, base_path: str = "icm_results"):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base directory for storing results
        """
        self.base_path = base_path
        self.logger = logging.getLogger(__name__)
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
    
    def save_result(
        self, 
        result: ICMResult, 
        name: str, 
        include_metadata: bool = True
    ) -> str:
        """
        Save ICM result to file.
        
        Args:
            result: ICM result to save
            name: Name for the saved file
            include_metadata: Whether to include search metadata
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jsonl"
        filepath = os.path.join(self.base_path, filename)
        
        with open(filepath, 'w') as f:
            # Write metadata if requested
            if include_metadata:
                metadata = {
                    "type": "icm_metadata",
                    "timestamp": timestamp,
                    "score": result.score,
                    "iterations": result.iterations,
                    "convergence_info": result.convergence_info,
                    "metadata": result.metadata
                }
                f.write(json.dumps(metadata) + '\n')
            
            # Write labeled examples
            for example in result.labeled_examples:
                example_data = {
                    "type": "icm_example",
                    "input": example["input"],
                    "label": example["label"],
                    "metadata": example.get("metadata", {})
                }
                f.write(json.dumps(example_data) + '\n')
        
        self.logger.info(f"Saved ICM result to {filepath}")
        return filepath
    
    def load_result(self, filepath: str) -> ICMResult:
        """
        Load ICM result from file.
        
        Args:
            filepath: Path to result file
            
        Returns:
            Loaded ICM result
        """
        labeled_examples = []
        metadata = {}
        convergence_info = {}
        score = 0.0
        iterations = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                if data.get("type") == "icm_metadata":
                    score = data.get("score", 0.0)
                    iterations = data.get("iterations", 0)
                    convergence_info = data.get("convergence_info", {})
                    metadata = data.get("metadata", {})
                
                elif data.get("type") == "icm_example":
                    labeled_examples.append({
                        "input": data["input"],
                        "label": data["label"],
                        "metadata": data.get("metadata", {})
                    })
        
        result = ICMResult(
            labeled_examples=labeled_examples,
            score=score,
            iterations=iterations,
            convergence_info=convergence_info,
            metadata=metadata
        )
        
        self.logger.info(f"Loaded ICM result from {filepath}")
        return result
    
    def save_labeled_dataset(
        self, 
        labeled_examples: List[Dict[str, Any]], 
        name: str,
        format: str = "jsonl"
    ) -> str:
        """
        Save labeled dataset in specified format.
        
        Args:
            labeled_examples: List of labeled examples
            name: Name for the dataset
            format: Output format (jsonl, json, csv)
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "jsonl":
            filename = f"{name}_{timestamp}.jsonl"
            filepath = os.path.join(self.base_path, filename)
            
            with open(filepath, 'w') as f:
                for example in labeled_examples:
                    f.write(json.dumps(example) + '\n')
        
        elif format == "json":
            filename = f"{name}_{timestamp}.json"
            filepath = os.path.join(self.base_path, filename)
            
            with open(filepath, 'w') as f:
                json.dump(labeled_examples, f, indent=2)
        
        elif format == "csv":
            import csv
            filename = f"{name}_{timestamp}.csv"
            filepath = os.path.join(self.base_path, filename)
            
            if labeled_examples:
                fieldnames = ["input", "label"] + list(labeled_examples[0].get("metadata", {}).keys())
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for example in labeled_examples:
                        row = {
                            "input": example["input"],
                            "label": example["label"]
                        }
                        row.update(example.get("metadata", {}))
                        writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved labeled dataset to {filepath}")
        return filepath
    
    def list_results(self) -> List[Dict[str, Any]]:
        """
        List all saved results.
        
        Returns:
            List of result information
        """
        results = []
        
        for filename in os.listdir(self.base_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.base_path, filename)
                try:
                    # Read first line to get metadata
                    with open(filepath, 'r') as f:
                        first_line = f.readline()
                        if first_line:
                            data = json.loads(first_line)
                            if data.get("type") == "icm_metadata":
                                results.append({
                                    "filename": filename,
                                    "filepath": filepath,
                                    "timestamp": data.get("timestamp"),
                                    "score": data.get("score"),
                                    "iterations": data.get("iterations"),
                                    "metadata": data.get("metadata", {})
                                })
                except Exception as e:
                    self.logger.warning(f"Could not read metadata from {filename}: {e}")
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results
    
    def clean_old_results(self, keep_latest: int = 10):
        """
        Clean old results, keeping only the latest N.
        
        Args:
            keep_latest: Number of latest results to keep
        """
        results = self.list_results()
        
        if len(results) > keep_latest:
            to_delete = results[keep_latest:]
            
            for result in to_delete:
                try:
                    os.remove(result["filepath"])
                    self.logger.info(f"Deleted old result: {result['filename']}")
                except Exception as e:
                    self.logger.warning(f"Could not delete {result['filename']}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored results.
        
        Returns:
            Dictionary with statistics
        """
        results = self.list_results()
        
        if not results:
            return {"total_results": 0}
        
        scores = [r["score"] for r in results if r.get("score") is not None]
        iterations = [r["iterations"] for r in results if r.get("iterations") is not None]
        
        stats = {
            "total_results": len(results),
            "score_stats": {
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "avg": sum(scores) / len(scores) if scores else None
            },
            "iteration_stats": {
                "min": min(iterations) if iterations else None,
                "max": max(iterations) if iterations else None,
                "avg": sum(iterations) / len(iterations) if iterations else None
            },
            "latest_result": results[0] if results else None
        }
        
        return stats


def create_huggingface_dataset(
    labeled_examples: List[Dict[str, Any]],
    dataset_name: str,
    split: str = "train"
) -> Dict[str, Any]:
    """
    Create a Hugging Face dataset format from labeled examples.
    
    Args:
        labeled_examples: List of labeled examples
        dataset_name: Name for the dataset
        split: Dataset split name
        
    Returns:
        Dataset in HF format
    """
    # Convert to HF format
    hf_data = {
        "inputs": [],
        "labels": [],
        "metadata": []
    }
    
    for example in labeled_examples:
        hf_data["inputs"].append(example["input"])
        hf_data["labels"].append(example["label"])
        hf_data["metadata"].append(example.get("metadata", {}))
    
    # Add dataset metadata
    dataset_info = {
        "name": dataset_name,
        "description": f"Dataset generated using Internal Coherence Maximization (ICM)",
        "num_examples": len(labeled_examples),
        "splits": [split],
        "features": {
            "inputs": "string",
            "labels": "string", 
            "metadata": "dict"
        }
    }
    
    return {
        "data": {split: hf_data},
        "info": dataset_info
    }


def export_to_training_format(
    labeled_examples: List[Dict[str, Any]],
    format: str = "dpo",
    output_path: str = "icm_training_data.jsonl"
) -> str:
    """
    Export labeled examples to training format.
    
    Args:
        labeled_examples: List of labeled examples
        format: Training format (dpo, sft, classification)
        output_path: Output file path
        
    Returns:
        Path to exported file
    """
    with open(output_path, 'w') as f:
        for example in labeled_examples:
            if format == "dpo":
                # Create DPO format (chosen/rejected pairs)
                if example["label"] == "True":
                    training_example = {
                        "prompt": example["input"],
                        "chosen": "True",
                        "rejected": "False"
                    }
                else:
                    training_example = {
                        "prompt": example["input"],
                        "chosen": "False", 
                        "rejected": "True"
                    }
            
            elif format == "sft":
                # Create SFT format
                training_example = {
                    "instruction": example["input"],
                    "output": example["label"]
                }
            
            elif format == "classification":
                # Create classification format
                training_example = {
                    "text": example["input"],
                    "label": example["label"]
                }
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            f.write(json.dumps(training_example) + '\n')
    
    return output_path
