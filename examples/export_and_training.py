#!/usr/bin/env python3
"""
Export and Training Example: End-to-End ICM Pipeline

This example demonstrates the complete ICM pipeline:
1. Generate labeled dataset with ICM
2. Export to various training formats
3. Push to Hugging Face Hub
"""

import logging
import os
from icm import ICMSearcher, load_icm_dataset
from icm.storage import ICMStorage
from icm.exporters import ICMExporter
from icm.utils import create_experiment_id

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_icm_pipeline(
    model_name: str = "google/gemma-3-1b-it",
    dataset_name: str = None,
    task_type: str = "classification",
    max_examples: int = 50,
    export_formats: list = None,
    push_to_hf: bool = False,
    hf_repo_id: str = None
):
    """
    Run complete ICM pipeline with export and optional HF upload.
    
    Args:
        model_name: Model to use for ICM
        dataset_name: Dataset name or None for synthetic
        task_type: Type of task
        max_examples: Maximum examples to process
        export_formats: List of export formats
        push_to_hf: Whether to push to Hugging Face
        hf_repo_id: Hugging Face repository ID
    """
    
    if export_formats is None:
        export_formats = ["json", "dpo"]
    
    # Create experiment ID
    experiment_id = create_experiment_id(
        model_name, 
        dataset_name or "synthetic", 
        task_type
    )
    
    logger.info(f"Starting ICM pipeline: {experiment_id}")
    
    # Step 1: Load or create dataset
    if dataset_name:
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_icm_dataset(
                dataset_name=dataset_name,
                task_type=task_type,
                sample_size=max_examples * 2,  # Load more than we need
                seed=42
            )
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            logger.info("Falling back to synthetic dataset")
            from icm.datasets import create_synthetic_dataset
            dataset = create_synthetic_dataset(
                task_type="math" if task_type == "classification" else "comparison",
                num_examples=max_examples * 2,
                seed=42
            )
    else:
        logger.info("Creating synthetic dataset")
        from icm.datasets import create_synthetic_dataset
        dataset = create_synthetic_dataset(
            task_type="math" if task_type == "classification" else "comparison",
            num_examples=max_examples * 2,
            seed=42
        )
    
    logger.info(f"Dataset loaded: {len(dataset)} examples")
    
    # Step 2: Run ICM
    logger.info("Initializing ICM searcher...")
    
    searcher = ICMSearcher(
        model_name=model_name,
        alpha=40.0,
        initial_temperature=6.0,
        final_temperature=0.01,
        max_iterations=150,
        initial_examples=5,
        generation_temperature=0.7,
        seed=42
    )
    
    logger.info("Running ICM search...")
    
    result = searcher.search(
        dataset=dataset,
        task_type=task_type,
        max_examples=max_examples
    )
    
    logger.info(f"ICM completed! Score: {result.score:.4f}, Examples: {len(result.labeled_examples)}")
    
    # Step 3: Save results
    storage = ICMStorage("examples/results")
    result_path = storage.save_result(result, experiment_id)
    
    logger.info(f"Results saved to: {result_path}")
    
    # Step 4: Export to different formats
    exporter = ICMExporter(storage)
    exported_files = {}
    
    for format_name in export_formats:
        logger.info(f"Exporting to {format_name} format...")
        
        output_path = f"examples/exports/{experiment_id}_{format_name}"
        
        # Ensure export directory exists
        os.makedirs("examples/exports", exist_ok=True)
        
        if format_name == "json":
            output_path += ".json"
            file_path = exporter.export_to_json(
                result.labeled_examples,
                output_path,
                include_stats=True
            )
        
        elif format_name == "dpo":
            output_path += ".jsonl"
            file_path = exporter.export_to_dpo_format(
                result.labeled_examples,
                output_path,
                create_pairs=True
            )
        
        
        elif format_name == "csv":
            output_path += ".csv"
            file_path = exporter.export_to_csv(
                result.labeled_examples,
                output_path
            )
        
        elif format_name == "analysis":
            output_path += "_analysis.json"
            file_path = exporter.export_analysis_report(
                result.labeled_examples,
                output_path,
                include_examples=True
            )
        
        else:
            logger.warning(f"Unknown export format: {format_name}")
            continue
        
        exported_files[format_name] = file_path
        logger.info(f"Exported {format_name}: {file_path}")
    
    # Step 5: Push to Hugging Face (optional)
    if push_to_hf and hf_repo_id:
        logger.info(f"Pushing to Hugging Face: {hf_repo_id}")
        
        try:
            # Push the main DPO dataset
            if "dpo" in exported_files:
                url = exporter.export_to_huggingface(
                    result.labeled_examples,
                    repo_id=hf_repo_id,
                    task_type=task_type,
                    model_name=model_name,
                    private=False,
                    create_readme=True
                )
                logger.info(f"Dataset uploaded to: {url}")
            else:
                logger.warning("No DPO format available for HF upload")
        
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")
    
    # Step 6: Generate summary report
    logger.info("Generating summary report...")
    
    # Calculate statistics
    label_counts = {}
    for example in result.labeled_examples:
        label = example["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Create summary
    summary = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "dataset_info": {
            "name": dataset_name or "synthetic",
            "task_type": task_type,
            "total_examples": len(dataset),
            "processed_examples": len(result.labeled_examples)
        },
        "icm_results": {
            "final_score": result.score,
            "iterations": result.iterations,
            "label_distribution": label_counts
        },
        "exports": {
            format_name: path for format_name, path in exported_files.items()
        }
    }
    
    # Save summary
    summary_path = f"examples/exports/{experiment_id}_summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print final report
    print("\n" + "="*60)
    print(f"ICM PIPELINE COMPLETED: {experiment_id}")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name or 'synthetic'} ({task_type})")
    print(f"Examples processed: {len(result.labeled_examples)}")
    print(f"Final ICM score: {result.score:.4f}")
    print("\nLabel Distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(result.labeled_examples)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    print("\nExported Files:")
    for format_name, path in exported_files.items():
        print(f"  {format_name}: {path}")
    if push_to_hf and hf_repo_id:
        print(f"\nHugging Face: https://huggingface.co/datasets/{hf_repo_id}")
    print("="*60)
    
    return result, exported_files, summary


def main():
    """Run the export and training example."""
    
    # Example 1: Simple synthetic dataset
    logger.info("Running Example 1: Synthetic Math Dataset")
    
    result1, files1, summary1 = run_icm_pipeline(
        model_name="google/gemma-3-1b-it",
        dataset_name=None,  # Use synthetic
        task_type="classification",
        max_examples=30,
        export_formats=["json", "dpo", "analysis"],
        push_to_hf=False
    )
    
    # Example 2: Try with a comparison task
    logger.info("\nRunning Example 2: Synthetic Comparison Dataset")
    
    result2, files2, summary2 = run_icm_pipeline(
        model_name="google/gemma-3-1b-it",
        dataset_name=None,
        task_type="comparison",
        max_examples=25,
        export_formats=["dpo", "csv"],
        push_to_hf=False
    )
    
    print(f"\nBoth examples completed successfully!")
    print(f"Example 1 generated {len(result1.labeled_examples)} examples")
    print(f"Example 2 generated {len(result2.labeled_examples)} examples")
    
    return result1, result2


if __name__ == "__main__":
    main()
