# Internal Coherence Maximization (ICM)

**ICM** (Internal Coherence Maximization) is a Python tool for unsupervised elicitation of language models. Based on the paper ["Unsupervised Elicitation of Language Models"](https://arxiv.org/abs/2506.10139), ICM fine-tunes pretrained language models on their own generated labels without external supervision.

## Key Features

- **Unsupervised Learning**: Generate high-quality labeled datasets without human supervision
- **Mutual Predictability**: Find labels that are logically consistent and mutually predictable
- **Multiple Task Types**: Support for classification, comparison, mathematical reasoning, and more
- **Flexible Export**: Export to various formats (DPO, CSV, JSON) and push to Hugging Face

## Installation

### From Source
```bash
git clone https://github.com/codelion/icm.git
cd icm
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Generate a labeled dataset using ICM:

```bash
icm run --model google/gemma-3-1b-it --dataset truthful_qa --task-type truthfulqa --max-examples 100
```

### Export to Training Format

```bash
icm export --input-path icm_results/truthfulqa_dialoGPT_20240115_143022.jsonl --output-path truthfulqa_dpo.jsonl --format dpo
```

### Push to Hugging Face

```bash
icm push --input-path truthfulqa_dpo.jsonl --hf-repo-id your-username/icm-truthfulqa-dataset
```

## Try Now

| Use Case | Dataset | Link |
|----------|----------|-------|
| Fine-tuning the model | dpo dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LfNDIvZs98hFBZzEc4ukYO1ZXLpM7WXS?usp=sharing) |

## Algorithm Overview

ICM uses two key components:

1. **Mutual Predictability**: Measures how well the model can predict each label given all other labels
2. **Logical Consistency**: Enforces simple logical constraints to prevent degenerate solutions

The algorithm uses simulated annealing to search for optimal label assignments that maximize:

```
U(D) = α × P_θ(D) - I(D)
```

Where:
- `P_θ(D)` is the mutual predictability score
- `I(D)` is the inconsistency penalty  
- `α` balances the two terms

## Supported Tasks

### TruthfulQA (Truthfulness)
```bash
# Fully automatic - detects config='multiple_choice' and split='validation'
icm run --model google/gemma-3-1b-it --dataset truthful_qa --task-type truthfulqa

# Or explicitly specify parameters
icm run --model google/gemma-3-1b-it --dataset truthful_qa --config multiple_choice --split validation --task-type truthfulqa
```

### GSM8K (Mathematical Reasoning)
```bash
# Fully automatic - detects config='main'
icm run --model google/gemma-3-1b-it --dataset gsm8k --task-type gsm8k

# Or explicitly specify parameters
icm run --model google/gemma-3-1b-it --dataset gsm8k --config main --task-type gsm8k
```

### Custom Datasets
```bash
icm run --model google/gemma-3-1b-it --dataset path/to/dataset.jsonl --task-type classification
```

## Synthetic Datasets

ICM can generate synthetic datasets for testing and experimentation. These are perfect for:
- **Testing ICM**: Validate the algorithm on simple, verifiable tasks
- **Quick experiments**: Generate datasets instantly without external dependencies
- **Educational purposes**: Understand how ICM works with clear logical relationships

### Available Synthetic Types

#### **Math Dataset** (`--synthetic math`)
Generates **simple addition problems** with both correct and incorrect solutions:

**Example Output:**
```
Question: What is 42 + 17?
Claim: 42 + 17 = 59
I think this Claim is [True/False]
```

**How it works:**
- Random numbers between 1-100
- Creates correct solutions (True labels)
- Creates incorrect solutions with random errors (False labels)  
- **Double the requested size**: `--synthetic-size 500` creates 1000 examples (500 correct + 500 incorrect)
- **Perfectly balanced**: 50% True, 50% False labels

#### **Comparison Dataset** (`--synthetic comparison`)
Generates **number comparison tasks**:

**Example Output:**
```
Query: Which number is larger?
Response A: 73
Response B: 45
Claim: Response A is larger than Response B
I think this Claim is [True/False]
```

**How it works:**
- Random pairs of numbers
- True/False based on actual comparison
- Single example per iteration (not doubled)

### Usage Examples

```bash
# Math problems - creates 1000 examples (500 pairs)
icm run --model google/gemma-3-1b-it --synthetic math --synthetic-size 500

# Number comparisons - creates 300 examples  
icm run --model google/gemma-3-1b-it --synthetic comparison --synthetic-size 300

# Quick test with defaults (100 examples)
icm run --model google/gemma-3-1b-it --synthetic math
```

### Why Use Synthetic Datasets?

1. **Instant generation**: No need to download or configure external datasets
2. **Verifiable ground truth**: Clear logical relationships for validation
3. **Reproducible**: Consistent results with same seed
4. **Perfect for testing**: Simple tasks ideal for algorithm validation
5. **No dependencies**: Works offline without internet connection

### Dataset Format

All synthetic examples follow the standard ICM format:
```json
{
  "input": "Question: What is 42 + 17?\nClaim: 42 + 17 = 59\nI think this Claim is [True/False]",
  "metadata": {
    "gold_label": "True",
    "task": "math"
  }
}
```

## Command Reference

### `icm run`

Run ICM on a dataset to generate labeled examples.

**Required Arguments:**
- `--model`: Model name or path (e.g., `google/gemma-3-1b-it`)

**Dataset Arguments:**
- `--dataset`: Dataset name or path
- `--task-type`: Task type (`auto`, `classification`, `comparison`, `truthfulqa`, `gsm8k`)
- `--split`: Dataset split (default: `train`)
- `--max-examples`: Maximum examples to process

**Synthetic Dataset Options:**
- `--synthetic`: Create synthetic dataset (`math`, `comparison`)
- `--synthetic-size`: Number of synthetic examples to generate (default: 100)

**ICM Algorithm Parameters:**
- `--alpha`: Weight for mutual predictability vs consistency (default: 100.0)
- `--initial-temperature`: Starting temperature for simulated annealing (default: 3.0)
- `--final-temperature`: Ending temperature (default: 0.001)
- `--cooling-rate`: Temperature cooling rate (default: 0.98)
- `--initial-examples`: Number of initial random examples (default: 20)
- `--max-iterations`: Maximum search iterations (default: 1000)

**Generation Parameters:**
- `--generation-temperature`: Temperature for text generation (default: 0.2)
- `--generation-top-p`: Top-p for nucleus sampling (default: 0.9)
- `--generation-max-tokens`: Maximum tokens to generate (default: 512)

**System Parameters:**
- `--device`: Computation device (`cuda`, `cpu`, `auto`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### `icm export`

Export ICM results to various formats.

**Required Arguments:**
- `--input-path`: Path to ICM result file
- `--output-path`: Output file path
- `--format`: Export format (`json`, `dpo`, `csv`, `analysis`)

**Optional Arguments:**
- `--include-stats`: Include statistics in JSON export
- `--create-pairs`: Create chosen/rejected pairs for DPO format
- `--hf-push`: Push to Hugging Face after export
- `--hf-repo-id`: Hugging Face repository ID
- `--private`: Make Hugging Face repository private

### `icm push`

Push files to Hugging Face Hub.

**Required Arguments:**
- `--input-path`: Local file path to upload
- `--hf-repo-id`: Hugging Face repository ID (e.g., `username/dataset-name`)

**Optional Arguments:**
- `--file-name`: Custom filename in repository
- `--private`: Make repository private

### `icm list`

List all saved ICM results.

```bash
icm list --results-dir icm_results
```

### `icm analyze`

Analyze ICM results and show statistics.

```bash
# Analyze all results
icm analyze

# Analyze specific result file
icm analyze --result-file icm_results/truthfulqa_gpt2_20240115_143022.jsonl
```

### `icm clean`

Clean old result files, keeping only the latest N results.

```bash
icm clean --keep-latest 10
```

## Configuration

### Using Configuration Files

Create a `config.json` file:

```json
{
  "search_params": {
    "alpha": 30.0,
    "initial_temperature": 15.0,
    "final_temperature": 0.005,
    "max_iterations": 2000
  },
  "model_params": {
    "generation_temperature": 0.8,
    "generation_top_p": 0.95
  },
  "system_params": {
    "device": "cuda",
    "seed": 123
  }
}
```

### Environment Variables

Set common parameters via environment variables:

```bash
export ICM_MODEL="google/gemma-3-1b-it"
export ICM_DEVICE="cuda"
export ICM_LOG_LEVEL="INFO"
```

## Python API

### Basic Usage

```python
from icm import ICMSearcher, load_icm_dataset

# Load dataset
dataset = load_icm_dataset("truthful_qa", task_type="truthfulqa")

# Create searcher
searcher = ICMSearcher(
    model_name="google/gemma-3-1b-it",
    alpha=50.0,
    max_iterations=1000
)

# Run ICM search
result = searcher.search(dataset, max_examples=100)

# Access results
print(f"Generated {len(result.labeled_examples)} labeled examples")
print(f"Final score: {result.score:.4f}")
```

### Advanced Usage

```python
from icm import ICMSearcher, ICMDataset, ICMExample
from icm.consistency import LogicalConsistencyChecker, MathConsistencyRule

# Create custom dataset
examples = [
    ICMExample("What is 2+2?", {"category": "math"}),
    ICMExample("What is 3+3?", {"category": "math"})
]
dataset = ICMDataset(examples)

# Custom consistency checker
checker = LogicalConsistencyChecker([MathConsistencyRule()])

# Advanced searcher
searcher = ICMSearcher(
    model_name="google/gemma-3-1b-it",
    alpha=30.0,
    initial_temperature=20.0,
    consistency_checker=checker,
    seed=42
)

result = searcher.search(dataset)
```

### Storage and Export

```python
from icm.storage import ICMStorage
from icm.exporters import ICMExporter

# Save results
storage = ICMStorage("my_results")
storage.save_result(result, "experiment_1")

# Export to DPO format
exporter = ICMExporter(storage)
exporter.export_to_dpo_format(
    result.labeled_examples,
    "training_data.jsonl"
)

# Push to Hugging Face
exporter.export_to_huggingface(
    result.labeled_examples,
    repo_id="username/my-icm-dataset",
    task_type="classification",
    model_name="google/gemma-3-1b-it"
)
```

## Examples

### Generate Math Dataset

```bash
# Create synthetic math dataset
icm run --model google/gemma-3-1b-it --synthetic math --synthetic-size 500 --max-iterations 500

# Use real GSM8K dataset  
icm run --model google/gemma-3-1b-it --dataset gsm8k --task-type gsm8k --max-examples 200
```

### Comparison Tasks

```bash
# Generate preference dataset
icm run --model google/gemma-3-1b-it --dataset anthropic/hh-rlhf --task-type comparison --alpha 30.0
```

### Export and Use

```bash
# Export to DPO format for training
icm export --input-path results.jsonl --output-path dpo_data.jsonl --format dpo --create-pairs

# Export analysis report
icm export --input-path results.jsonl --output-path analysis.json --format analysis --include-examples
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Use smaller model, MPS (Apple Silicon), or CPU
icm run --model google/gemma-3-1b-it --device cpu
# or on Apple Silicon:
icm run --model google/gemma-3-1b-it --device mps
```

**Model Loading Errors:**
```bash
# Verify model name and check internet connection
icm run --model google/gemma-3-1b-it --log-level DEBUG
```

**Poor Quality Results:**
```bash
# Increase alpha or iterations
icm run --model your-model --alpha 100.0 --max-iterations 2000
```

**Dataset Configuration Errors:**
```bash
# ICM now auto-detects both config and split for known datasets
# TruthfulQA: automatically uses config='multiple_choice' and split='validation'
# GSM8K: automatically uses config='main' and split='train'

# Your commands should work automatically:
icm run --model google/gemma-3-1b-it --dataset truthful_qa --task-type truthfulqa
icm run --model google/gemma-3-1b-it --dataset gsm8k --task-type gsm8k

# Or specify manually if needed:
icm run --model google/gemma-3-1b-it --dataset truthful_qa --config multiple_choice --split validation --task-type truthfulqa
icm run --model google/gemma-3-1b-it --dataset gsm8k --config main --task-type gsm8k
```

**Memory Usage Issues:**
```bash
# ICM uses memory-efficient sampling to handle large datasets
# If you still encounter memory issues, reduce the dataset size:
icm run --model google/gemma-3-1b-it --dataset large-dataset --max-examples 50

# Or use a smaller model:
icm run --model distilgpt2 --dataset your-dataset --max-examples 100
```

### Debug Mode

Enable detailed logging:

```bash
icm run --model google/gemma-3-1b-it --dataset your-data --log-level DEBUG --log-file debug.log
```

### Development Setup

```bash
git clone https://github.com/codelion/icm.git
cd icm
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## Citation

If you use ICM in your research, please cite:

```bibtex
@software{icm,
  title = {ICM: Internal Coherence Maximization},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/icm}
}
```

## Related Work

- **Eliciting Fine-Tuned Transformer Capabilities**: [Paper](https://arxiv.org/abs/2506.08060)
- **Weak-to-Strong Generalization**: [Paper](https://arxiv.org/abs/2312.09390)
- **Constitutional AI**: [Paper](https://arxiv.org/abs/2212.08073) 
- **Discovering Latent Knowledge**: [Paper](https://arxiv.org/abs/2212.03827)
