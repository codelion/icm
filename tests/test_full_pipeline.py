#!/usr/bin/env python3
"""
Test the full ICM pipeline with all 6 Google-reported benchmarks using Gemma 3 270M-IT.
Benchmarks: HellaSwag, PIQA, ARC-Challenge, WinoGrande, BIG-Bench Hard, IFEval
Plus bonus: TruthfulQA, GSM8K
Tests 1 sample from each benchmark, runs ICM, and combines into DPO dataset.
"""

import subprocess
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log the result."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        logger.info(f"✓ {description} completed in {end_time - start_time:.1f}s")
        if result.stdout.strip():
            logger.info(f"Output: {result.stdout.strip()}")
    else:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {result.stderr.strip()}")
        return False
    
    return True

def main():
    """Run the full pipeline test."""
    logger.info("🚀 Starting ICM + DPO pipeline test with Gemma 3 270M-IT")
    
    # Model to use
    model = "google/gemma-3-270m-it"
    
    # Benchmark datasets to test - all 6 from Google's report plus bonuses
    benchmarks = [
        # Google's 6 benchmarks
        ("Rowan/hellaswag", "hellaswag", None),          # HellaSwag: 37.7%
        ("piqa", "piqa", None),                          # PIQA: 66.2%
        ("allenai/ai2_arc", "arc_challenge", "ARC-Challenge"),  # ARC-c: 28.2%
        ("allenai/winogrande", "winogrande", "winogrande_xl"),  # WinoGrande: 52.3%
        ("maveriq/bigbenchhard", "bigbench_hard", "causal_judgement"), # BIG-Bench Hard: 26.7%
        ("google/IFEval", "ifeval", None),               # IFEval: 51.2%
        # Bonus datasets
        ("truthful_qa", "truthfulqa", "multiple_choice"),
        ("gsm8k", "gsm8k", "main"),
    ]
    
    # Clean up any existing results
    logger.info("Cleaning up existing results...")
    if os.path.exists("icm_results"):
        run_command(["python", "-m", "icm.cli", "clean", "--keep-latest", "0"], "Clean old results")
    
    # Run ICM on each benchmark with 1 sample
    for dataset_info in benchmarks:
        dataset_name = dataset_info[0]
        task_type = dataset_info[1]
        config = dataset_info[2] if len(dataset_info) > 2 else None
        
        cmd = [
            "python", "-m", "icm.cli", "run",
            "--model", model,
            "--dataset", dataset_name,
            "--task-type", task_type,
            "--max-examples", "1",  # Only 1 sample for testing
            "--max-iterations", "50",  # Fewer iterations for testing
            "--log-level", "INFO"
        ]
        
        # Add config if specified
        if config:
            cmd.extend(["--config", config])
        
        success = run_command(cmd, f"ICM on {dataset_name}")
        if not success:
            logger.error(f"Failed to run ICM on {dataset_name}")
            return False
    
    # List the results
    logger.info("\n📋 Listing ICM results...")
    run_command(["python", "-m", "icm.cli", "list"], "List results")
    
    # Combine all results into DPO dataset
    logger.info("\n🔗 Combining results into DPO dataset...")
    cmd = [
        "python", "-m", "icm.cli", "export-combined",
        "--input-dir", "icm_results",
        "--output-path", "combined_benchmarks_dpo.jsonl",
        "--log-level", "INFO"
    ]
    
    success = run_command(cmd, "Combine results to DPO")
    if not success:
        logger.error("Failed to combine results")
        return False
    
    # Check the final DPO dataset
    if os.path.exists("combined_benchmarks_dpo.jsonl"):
        with open("combined_benchmarks_dpo.jsonl", "r") as f:
            lines = f.readlines()
        
        logger.info(f"\n📊 Final DPO dataset created: {len(lines)} preference pairs")
        
        # Show a sample
        if lines:
            import json
            sample = json.loads(lines[0])
            logger.info("Sample DPO pair:")
            logger.info(f"  Prompt: {sample.get('prompt', '')[:100]}...")
            logger.info(f"  Chosen: {sample.get('chosen', '')[:100]}...")
            logger.info(f"  Rejected: {sample.get('rejected', '')[:100]}...")
            logger.info(f"  Source: {sample.get('source_dataset', 'unknown')}")
    
    logger.info("\n🎉 Full pipeline test completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Run with more samples: --max-examples 100 or higher")
    logger.info("2. Fine-tune Gemma 3 270M-IT with DPO using the combined dataset")
    logger.info("3. Evaluate improved model on benchmarks")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)