"""
Utility functions for ICM.

This module provides various utility functions used across the ICM package.
"""

import os
import json
import random
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> str:
    """Get the appropriate device for computation with smart prioritization.
    
    Priority order:
    1. CUDA (if available)
    2. MPS (Apple Silicon GPU, if available) 
    3. CPU (fallback)
    
    Args:
        device: Explicit device override ("cuda", "mps", "cpu", "auto", or None)
        
    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    if device and device != "auto":
        return device
    
    # Priority 1: CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    
    # Priority 2: MPS (Apple Silicon GPUs)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    # Priority 3: CPU (fallback)
    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about available devices."""
    info = {
        "available_devices": [],
        "recommended_device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    # Add CUDA info
    if torch.cuda.is_available():
        info["available_devices"].append("cuda")
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    # Add MPS info
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["available_devices"].append("mps")
    
    # CPU is always available
    info["available_devices"].append("cpu")
    
    return info


def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(filepath: str) -> Union[Dict, List]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json_file(data: Union[Dict, List], filepath: str, indent: int = 2):
    """Save data to JSON file."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl_file(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on common words."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_tokens_approximate(text: str) -> int:
    """Approximate token count (rough estimate)."""
    # Very rough approximation: 1 token ≈ 0.75 words ≈ 4 characters
    return len(text) // 4


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Batch items into chunks of specified size."""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent()
    }


def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    
    try:
        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        
        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved,
            "total_mb": memory_total,
            "utilization_percent": (memory_allocated / memory_total) * 100
        }
    except Exception:
        return None


def validate_model_name(model_name: str) -> bool:
    """Validate if model name appears to be valid."""
    # Basic validation - could be enhanced
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Check for common patterns
    valid_patterns = [
        "/",  # HF format like "microsoft/DialoGPT-medium"
        "-",  # Common separator
        "_",  # Common separator
        "."   # File paths
    ]
    
    return len(model_name) > 0 and any(char.isalnum() for char in model_name)


def extract_model_info(model_name: str) -> Dict[str, str]:
    """Extract information from model name."""
    info = {
        "full_name": model_name,
        "organization": "",
        "model": "",
        "short_name": ""
    }
    
    if "/" in model_name:
        parts = model_name.split("/")
        info["organization"] = parts[0]
        info["model"] = "/".join(parts[1:])
        info["short_name"] = parts[-1]
    else:
        info["model"] = model_name
        info["short_name"] = model_name
    
    return info


def create_experiment_id(
    model_name: str, 
    dataset_name: str, 
    task_type: str,
    timestamp: Optional[str] = None
) -> str:
    """Create a unique experiment identifier."""
    from datetime import datetime
    
    # Extract short names
    model_info = extract_model_info(model_name)
    dataset_short = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    
    # Clean names (remove special characters)
    model_clean = "".join(c for c in model_info["short_name"] if c.isalnum() or c in "-_")
    dataset_clean = "".join(c for c in dataset_short if c.isalnum() or c in "-_")
    task_clean = "".join(c for c in task_type if c.isalnum() or c in "-_")
    
    # Create ID
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{model_clean}_{dataset_clean}_{task_clean}_{timestamp}"


def setup_device_optimizations(device: str) -> bool:
    """Setup device-specific optimizations.
    
    Args:
        device: Device string ("cuda", "mps", or "cpu")
        
    Returns:
        True if optimizations were applied, False otherwise
    """
    if device == "cuda" and torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return True
    
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS-specific optimizations can be added here in the future
        # Currently, PyTorch handles MPS optimizations automatically
        return True
    
    return False


def log_system_info(logger: logging.Logger):
    """Log system information for debugging."""
    import platform
    import sys
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"  Recommended device: {device_info['recommended_device']}")
    logger.info(f"  Available devices: {', '.join(device_info['available_devices'])}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        gpu_mem = get_gpu_memory_usage()
        if gpu_mem:
            logger.info(f"  GPU Memory: {gpu_mem['total_mb']:.0f} MB total")
    else:
        logger.info("  CUDA: Not available")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("  MPS (Apple Silicon): Available")
    else:
        logger.info("  MPS (Apple Silicon): Not available")
    
    cpu_mem = get_memory_usage()
    logger.info(f"  RAM Usage: {cpu_mem['rss_mb']:.0f} MB ({cpu_mem['percent']:.1f}%)")


class ProgressTracker:
    """Simple progress tracker for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
        
    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress."""
        self.current += increment
        
        if message:
            self.logger.info(f"{self.description}: {message} ({self.current}/{self.total})")
        elif self.current % max(1, self.total // 10) == 0:  # Log every 10%
            percentage = (self.current / self.total) * 100
            self.logger.info(f"{self.description}: {percentage:.0f}% ({self.current}/{self.total})")
    
    def finish(self, message: Optional[str] = None):
        """Mark as finished."""
        final_message = message or "Completed"
        self.logger.info(f"{self.description}: {final_message} ({self.current}/{self.total})")


class ConfigValidator:
    """Validator for ICM configuration."""
    
    @staticmethod
    def validate_search_params(params: Dict[str, Any]) -> List[str]:
        """Validate ICM search parameters."""
        errors = []
        
        # Check alpha
        alpha = params.get("alpha", 50.0)
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            errors.append("alpha must be a positive number")
        
        # Check temperatures
        initial_temp = params.get("initial_temperature", 10.0)
        final_temp = params.get("final_temperature", 0.01)
        
        if not isinstance(initial_temp, (int, float)) or initial_temp <= 0:
            errors.append("initial_temperature must be positive")
        
        if not isinstance(final_temp, (int, float)) or final_temp <= 0:
            errors.append("final_temperature must be positive")
        
        if final_temp >= initial_temp:
            errors.append("final_temperature must be less than initial_temperature")
        
        # Check cooling rate
        cooling_rate = params.get("cooling_rate", 0.99)
        if not isinstance(cooling_rate, (int, float)) or not 0 < cooling_rate < 1:
            errors.append("cooling_rate must be between 0 and 1")
        
        # Check iterations
        max_iterations = params.get("max_iterations", 1000)
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            errors.append("max_iterations must be a positive integer")
        
        return errors
    
    @staticmethod
    def validate_model_params(params: Dict[str, Any]) -> List[str]:
        """Validate model parameters."""
        errors = []
        
        # Check model name
        model_name = params.get("model_name")
        if not model_name or not validate_model_name(model_name):
            errors.append("model_name must be a valid model identifier")
        
        # Check generation parameters
        gen_temp = params.get("generation_temperature", 0.7)
        if not isinstance(gen_temp, (int, float)) or not 0 < gen_temp <= 2.0:
            errors.append("generation_temperature must be between 0 and 2.0")
        
        gen_top_p = params.get("generation_top_p", 0.9)
        if not isinstance(gen_top_p, (int, float)) or not 0 < gen_top_p <= 1.0:
            errors.append("generation_top_p must be between 0 and 1.0")
        
        max_tokens = params.get("generation_max_tokens", 512)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            errors.append("generation_max_tokens must be a positive integer")
        
        return errors


def create_default_config() -> Dict[str, Any]:
    """Create default ICM configuration."""
    return {
        "search_params": {
            "alpha": 50.0,
            "initial_temperature": 10.0,
            "final_temperature": 0.01,
            "cooling_rate": 0.99,
            "initial_examples": 8,
            "max_iterations": 1000,
            "consistency_fix_iterations": 10
        },
        "model_params": {
            "generation_temperature": 0.7,
            "generation_top_p": 0.9,
            "generation_max_tokens": 512
        },
        "system_params": {
            "device": "auto",
            "seed": 42,
            "log_level": "INFO"
        }
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
