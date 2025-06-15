"""
Internal Coherence Maximization (ICM) - Unsupervised Elicitation of Language Models.

This module implements the Internal Coherence Maximization algorithm from the paper
"Unsupervised Elicitation of Language Models", which fine-tunes pretrained language models
on their own generated labels without external supervision.

The algorithm uses mutual predictability and logical consistency to find coherent
label assignments that reflect the model's internal understanding of concepts.
"""

from .core import ICMSearcher, ICMResult
from .datasets import ICMDataset, load_icm_dataset
from .consistency import LogicalConsistencyChecker
from .exporters import ICMExporter
from .storage import ICMStorage

__version__ = "0.1.0"
__author__ = "codelion"
__email__ = "codelion@okyasoft.com"

__all__ = [
    "ICMSearcher",
    "ICMResult", 
    "ICMDataset",
    "load_icm_dataset",
    "LogicalConsistencyChecker",
    "ICMExporter",
    "ICMStorage",
]
