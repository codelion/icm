"""
Command-line interface for Internal Coherence Maximization (ICM).

This module provides a command-line interface for using ICM to generate
labeled datasets without external supervision.
"""

import argparse
import logging
import sys
import os
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .core import ICMSearcher, ICMResult
from .datasets import load_icm_dataset, create_synthetic_dataset
from .storage import ICMStorage
from .exporters import ICMExporter, push_to_huggingface, combine_icm_results_to_dpo
from .consistency import LogicalConsistencyChecker


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def run_icm(args):
    """Run the ICM algorithm on a dataset."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running ICM with model {args.model}")
    logger.info(f"Parameters: alpha={args.alpha}, temp={args.initial_temperature}->{args.final_temperature}")
    
    try:
        # Load dataset
        if args.synthetic:
            logger.info(f"Creating synthetic {args.synthetic} dataset")
            dataset = create_synthetic_dataset(
                task_type=args.synthetic,
                num_examples=args.synthetic_size,
                seed=args.seed
            )
        else:
            logger.info(f"Loading dataset: {args.dataset}")
            dataset = load_icm_dataset(
            dataset_name=args.dataset,
            task_type=args.task_type,
            split=args.split,
            config=args.config,
            sample_size=args.sample_size,
            seed=args.seed
            )
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Create consistency checker if custom rules specified
        consistency_checker = LogicalConsistencyChecker()
        
        # Create ICM searcher
        searcher = ICMSearcher(
            model_name=args.model,
            device=args.device,
            alpha=args.alpha,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature,
            cooling_rate=args.cooling_rate,
            initial_examples=args.initial_examples,
            max_iterations=args.max_iterations,
            consistency_fix_iterations=args.consistency_fix_iterations,
            generation_temperature=args.generation_temperature,
            generation_top_p=args.generation_top_p,
            generation_max_tokens=args.generation_max_tokens,
            consistency_checker=consistency_checker,
            confidence_threshold=args.confidence_threshold,
            seed=args.seed,
            log_level=args.log_level
        )
        
        # Run ICM search
        logger.info("Starting ICM search...")
        result = searcher.search(
            dataset=dataset,
            task_type=dataset.metadata.get("task_type", args.task_type),
            max_examples=args.max_examples
        )
        
        logger.info(f"ICM search completed. Final score: {result.score:.4f}")
        logger.info(f"Generated {len(result.labeled_examples)} labeled examples")
        
        # Save results
        storage = ICMStorage(args.output_dir)
        
        if args.output_name:
            output_name = args.output_name
        else:
            # Generate name from dataset/synthetic and model
            if args.synthetic:
                dataset_name = f"synthetic_{args.synthetic}"
            else:
                dataset_name = args.dataset.split('/')[-1] if '/' in args.dataset else args.dataset
            
            model_name = args.model.split('/')[-1] if '/' in args.model else args.model
            output_name = f"{dataset_name}_{model_name}_icm"
        
        # Save clean labeled dataset (no metadata needed)
        dataset_path = storage.save_labeled_dataset(
            result.labeled_examples,
            output_name,
            format=args.output_format
        )
        
        logger.info(f"Dataset saved to: {dataset_path}")
        
        # Print summary statistics
        label_counts = {}
        for ex in result.labeled_examples:
            label = ex["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(result.labeled_examples)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    except Exception as e:
        error_msg = str(e) if str(e).strip() else "Unknown error occurred"
        logger.error(f"Error running ICM: {error_msg}")
        if args.log_level == "DEBUG":
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        else:
            logger.error("Run with --log-level DEBUG for full traceback")
        sys.exit(1)


def export_results(args):
    """Export ICM results to various formats."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Exporting from {args.input_path} to {args.output_path}")
    
    try:
        # Load results
        storage = ICMStorage()
        
        if args.input_path.endswith('.jsonl') and 'icm_metadata' in open(args.input_path).readline():
            # Load full ICM result
            result = storage.load_result(args.input_path)
            labeled_examples = result.labeled_examples
        else:
            # Load just labeled examples
            labeled_examples = []
            with open(args.input_path, 'r') as f:
                if args.input_path.endswith('.jsonl'):
                    for line in f:
                        data = json.loads(line)
                        if data.get("type") == "icm_example":
                            labeled_examples.append({
                                "input": data["input"],
                                "label": data["label"],
                                "metadata": data.get("metadata", {})
                            })
                        elif "input" in data and "label" in data:
                            labeled_examples.append(data)
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        labeled_examples = data
                    else:
                        labeled_examples = [data]
        
        logger.info(f"Loaded {len(labeled_examples)} labeled examples")
        
        # Create exporter
        exporter = ICMExporter(storage)
        
        # Export based on format
        if args.format == "json":
            output_path = exporter.export_to_json(labeled_examples, args.output_path, args.include_stats)
        elif args.format == "dpo":
            output_path = exporter.export_to_dpo_format(labeled_examples, args.output_path, args.create_pairs)
        elif args.format == "csv":
            output_path = exporter.export_to_csv(labeled_examples, args.output_path)
        elif args.format == "analysis":
            output_path = exporter.export_analysis_report(labeled_examples, args.output_path, args.include_examples)
        else:
            raise ValueError(f"Unsupported export format: {args.format}")
        
        logger.info(f"Exported to {output_path}")
        
        # Push to HF if requested
        if args.hf_push:
            if not args.hf_repo_id:
                raise ValueError("--hf-repo-id is required when using --hf-push")
            
            url = push_to_huggingface(
                file_path=output_path,
                repo_id=args.hf_repo_id,
                private=args.private
            )
            logger.info(f"Pushed to Hugging Face: {url}")
    
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


def push_to_hf(args):
    """Push a file to Hugging Face Hub."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Pushing {args.input_path} to Hugging Face: {args.hf_repo_id}")
    
    try:
        url = push_to_huggingface(
            file_path=args.input_path,
            repo_id=args.hf_repo_id,
            file_name=args.file_name,
            private=args.private
        )
        logger.info(f"Successfully pushed to: {url}")
    
    except Exception as e:
        logger.error(f"Error pushing to Hugging Face: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


def list_results(args):
    """List saved ICM results."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    storage = ICMStorage(args.results_dir)
    results = storage.list_results()
    
    if not results:
        logger.info("No saved results found")
        return
    
    logger.info(f"Found {len(results)} saved results:")
    logger.info("-" * 80)
    
    for result in results:
        logger.info(f"File: {result['filename']}")
        logger.info(f"Timestamp: {result.get('timestamp', 'Unknown')}")
        logger.info(f"Score: {result.get('score', 'Unknown')}")
        logger.info(f"Iterations: {result.get('iterations', 'Unknown')}")
        
        metadata = result.get('metadata', {})
        if metadata:
            logger.info(f"Model: {metadata.get('model_name', 'Unknown')}")
            logger.info(f"Task: {metadata.get('task_type', 'Unknown')}")
            logger.info(f"Dataset size: {metadata.get('dataset_size', 'Unknown')}")
        
        logger.info("-" * 80)


def analyze_results(args):
    """Analyze ICM results."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    storage = ICMStorage(args.results_dir)
    
    if args.result_file:
        # Analyze specific result
        result = storage.load_result(args.result_file)
        
        logger.info(f"Analysis of {args.result_file}:")
        logger.info(f"Final score: {result.score:.4f}")
        logger.info(f"Iterations: {result.iterations}")
        logger.info(f"Examples: {len(result.labeled_examples)}")
        
        # Label distribution
        label_counts = {}
        for ex in result.labeled_examples:
            label = ex["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(result.labeled_examples)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Convergence info
        if result.convergence_info:
            logger.info("Convergence info:")
            for key, value in result.convergence_info.items():
                logger.info(f"  {key}: {value}")
    
    else:
        # Analyze all results
        stats = storage.get_statistics()
        
        logger.info("Overall statistics:")
        logger.info(f"Total results: {stats['total_results']}")
        
        if stats['score_stats']['avg'] is not None:
            logger.info(f"Score range: {stats['score_stats']['min']:.4f} - {stats['score_stats']['max']:.4f}")
            logger.info(f"Average score: {stats['score_stats']['avg']:.4f}")
        
        if stats['iteration_stats']['avg'] is not None:
            logger.info(f"Iteration range: {stats['iteration_stats']['min']} - {stats['iteration_stats']['max']}")
            logger.info(f"Average iterations: {stats['iteration_stats']['avg']:.0f}")


def clean_results(args):
    """Clean old ICM results."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    storage = ICMStorage(args.results_dir)
    
    logger.info(f"Cleaning old results, keeping latest {args.keep_latest}")
    storage.clean_old_results(args.keep_latest)
    logger.info("Cleanup completed")


def export_combined(args):
    """Combine multiple ICM results into a single DPO dataset."""
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Find ICM result files in the input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {args.input_dir}")
        
        # Get all .jsonl files in the directory
        result_files = list(input_dir.glob("*.jsonl"))
        
        if not result_files:
            raise ValueError(f"No .jsonl files found in {args.input_dir}")
        
        logger.info(f"Found {len(result_files)} total result files")
        
        # Select best quality files if enabled
        if getattr(args, 'quality_selection', True):  # Default to True for better results
            from .exporters import select_best_icm_files
            selected_files = select_best_icm_files([str(f) for f in result_files])
            logger.info(f"Quality selection: using {len(selected_files)} out of {len(result_files)} files")
        else:
            selected_files = [str(f) for f in result_files]
            logger.info("Quality selection disabled - using all files")
        
        # Create combined DPO dataset
        storage = ICMStorage()
        output_path = combine_icm_results_to_dpo(
            result_files=selected_files,
            output_path=args.output_path,
            storage=storage,
            fix_responses=args.fix_responses,
            balance_strategy=args.balance_strategy,
            max_per_benchmark=args.max_per_benchmark
        )
        
        logger.info(f"Combined DPO dataset created: {output_path}")
        
        # Push to HF if requested
        if args.hf_push:
            if not args.hf_repo_id:
                raise ValueError("--hf-repo-id is required when using --hf-push")
            
            url = push_to_huggingface(
                file_path=output_path,
                repo_id=args.hf_repo_id,
                private=args.private
            )
            logger.info(f"Pushed to Hugging Face: {url}")
    
    except Exception as e:
        logger.error(f"Error combining results: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Internal Coherence Maximization (ICM) - Unsupervised Elicitation of Language Models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")
    
    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run ICM on a dataset")
    run_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    run_parser.add_argument("--dataset", type=str, help="Dataset name or path")
    run_parser.add_argument("--task-type", type=str, default="auto", 
                           choices=["auto", "classification", "comparison", "truthfulqa", "gsm8k",
                                   "hellaswag", "piqa", "arc_challenge", "winogrande", "bigbench_hard", "ifeval"],
                           help="Task type")
    run_parser.add_argument("--split", type=str, default="train", help="Dataset split")
    run_parser.add_argument("--config", type=str, default=None, help="Dataset configuration (e.g., 'multiple_choice' for truthful_qa)")
    run_parser.add_argument("--sample-size", type=int, default=None, help="Sample size from dataset")
    run_parser.add_argument("--max-examples", type=int, default=None, help="Maximum examples available for labeling (limits the pool of examples ICM can label)")
    run_parser.add_argument("--output-dir", type=str, default="icm_results", help="Output directory")
    run_parser.add_argument("--output-name", type=str, default=None, help="Output name prefix")
    run_parser.add_argument("--output-format", type=str, default="jsonl", 
                           choices=["jsonl", "json", "csv"], help="Output format")
    
    # ICM algorithm parameters
    run_parser.add_argument("--alpha", type=float, default=100.0, help="Weight for mutual predictability vs consistency")
    run_parser.add_argument("--initial-temperature", type=float, default=3.0, help="Initial temperature for simulated annealing")
    run_parser.add_argument("--final-temperature", type=float, default=0.001, help="Final temperature for simulated annealing")
    run_parser.add_argument("--cooling-rate", type=float, default=0.98, help="Temperature cooling rate")
    run_parser.add_argument("--initial-examples", type=int, default=20, help="Number of initial randomly labeled examples (K)")
    run_parser.add_argument("--max-iterations", type=int, default=1000, help="Maximum iterations")
    run_parser.add_argument("--consistency-fix-iterations", type=int, default=10, 
                           help="Max iterations for consistency fixing")
    run_parser.add_argument("--confidence-threshold", type=float, default=0.1, 
                           help="Minimum confidence to label an example (0-1)")
    
    # Generation parameters
    run_parser.add_argument("--generation-temperature", type=float, default=0.2, help="Temperature for text generation")
    run_parser.add_argument("--generation-top-p", type=float, default=0.9, help="Generation top-p")
    run_parser.add_argument("--generation-max-tokens", type=int, default=512, help="Max generation tokens")
    
    # System parameters
    run_parser.add_argument("--device", type=str, default=None, 
                           help="Device (auto/cuda/mps/cpu). Auto-detects: CUDA > MPS > CPU")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--log-level", type=str, default="INFO", 
                           choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    run_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # Synthetic dataset options
    run_parser.add_argument("--synthetic", type=str, default=None, 
                           choices=["math", "comparison"], help="Create synthetic dataset")
    run_parser.add_argument("--synthetic-size", type=int, default=100, help="Synthetic dataset size")
    
    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export ICM results")
    export_parser.add_argument("--input-path", type=str, required=True, help="Input file path")
    export_parser.add_argument("--output-path", type=str, required=True, help="Output file path")
    export_parser.add_argument("--format", type=str, required=True, 
                              choices=["json", "dpo", "csv", "analysis"], help="Export format")
    export_parser.add_argument("--include-stats", action="store_true", help="Include statistics")
    export_parser.add_argument("--include-examples", action="store_true", help="Include example details")
    export_parser.add_argument("--create-pairs", action="store_true", help="Create DPO pairs")
    export_parser.add_argument("--hf-push", action="store_true", help="Push to Hugging Face")
    export_parser.add_argument("--hf-repo-id", type=str, help="HF repository ID")
    export_parser.add_argument("--private", action="store_true", help="Make HF repo private")
    export_parser.add_argument("--log-level", type=str, default="INFO", 
                              choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    export_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # Push subcommand
    push_parser = subparsers.add_parser("push", help="Push file to Hugging Face")
    push_parser.add_argument("--input-path", type=str, required=True, help="Input file path")
    push_parser.add_argument("--hf-repo-id", type=str, required=True, help="HF repository ID")
    push_parser.add_argument("--file-name", type=str, default=None, help="File name in repo")
    push_parser.add_argument("--private", action="store_true", help="Make repo private")
    push_parser.add_argument("--log-level", type=str, default="INFO", 
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    push_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # List subcommand
    list_parser = subparsers.add_parser("list", help="List saved results")
    list_parser.add_argument("--results-dir", type=str, default="icm_results", help="Results directory")
    list_parser.add_argument("--log-level", type=str, default="INFO", 
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    list_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze ICM results")
    analyze_parser.add_argument("--results-dir", type=str, default="icm_results", help="Results directory")
    analyze_parser.add_argument("--result-file", type=str, default=None, help="Specific result file to analyze")
    analyze_parser.add_argument("--log-level", type=str, default="INFO", 
                               choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    analyze_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # Clean subcommand
    clean_parser = subparsers.add_parser("clean", help="Clean old results")
    clean_parser.add_argument("--results-dir", type=str, default="icm_results", help="Results directory")
    clean_parser.add_argument("--keep-latest", type=int, default=10, help="Number of latest results to keep")
    clean_parser.add_argument("--log-level", type=str, default="INFO", 
                             choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    clean_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    # Export-combined subcommand
    export_combined_parser = subparsers.add_parser("export-combined", help="Combine multiple ICM results into single DPO dataset")
    export_combined_parser.add_argument("--input-dir", type=str, default="icm_results", help="Directory containing ICM result files")
    export_combined_parser.add_argument("--output-path", type=str, required=True, help="Output path for combined DPO dataset")
    export_combined_parser.add_argument("--fix-responses", action="store_true", default=True, help="Extract clean answers from response_text (removes question repetition)")
    export_combined_parser.add_argument("--no-fix-responses", action="store_false", dest="fix_responses", help="Use original response_text format")
    export_combined_parser.add_argument("--balance-strategy", type=str, default="none", choices=["equal", "proportional", "none"], 
                                       help="Dataset balancing strategy")
    export_combined_parser.add_argument("--max-per-benchmark", type=int, default=1000, help="Maximum examples per benchmark group when balancing")
    export_combined_parser.add_argument("--quality-selection", action="store_true", default=True, help="Use quality-based file selection (default: enabled)")
    export_combined_parser.add_argument("--no-quality-selection", action="store_false", dest="quality_selection", help="Disable quality selection, use all files")
    export_combined_parser.add_argument("--hf-push", action="store_true", help="Push to Hugging Face")
    export_combined_parser.add_argument("--hf-repo-id", type=str, help="HF repository ID")
    export_combined_parser.add_argument("--private", action="store_true", help="Make HF repo private")
    export_combined_parser.add_argument("--log-level", type=str, default="INFO", 
                                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    export_combined_parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "run":
        run_icm(args)
    elif args.command == "export":
        export_results(args)
    elif args.command == "export-combined":
        export_combined(args)
    elif args.command == "push":
        push_to_hf(args)
    elif args.command == "list":
        list_results(args)
    elif args.command == "analyze":
        analyze_results(args)
    elif args.command == "clean":
        clean_results(args)
    else:
        print("Please specify a command: run, export, export-combined, push, list, analyze, or clean")
        print("Use 'icm --help' for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
