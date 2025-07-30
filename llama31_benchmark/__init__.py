"""
Llama 3.1 Benchmark Package

A modular benchmarking framework for evaluating Llama 3.1 models of various sizes
on the ITALIC dataset with vLLM for efficient batching and automatic GPU optimization.

Requirements:
- vllm >= 0.2.0
- torch >= 2.0.0
- transformers >= 4.30.0
- pandas
- tqdm
- python-dotenv
"""

from .config import Llama31BenchmarkConfig
from .benchmark import Llama31Benchmark
from .utils import (
    load_italic_dataset,
    create_uniform_subset, 
    extract_answer_robust,
    analyze_results_by_category,
    cleanup_gpu_memory,
    get_gpu_info
)

__version__ = "1.0.0"
__author__ = "Llama 3.1 Benchmark Team"

__all__ = [
    # Main classes
    "Llama31BenchmarkConfig",
    "Llama31Benchmark",
    
    # Utility functions
    "load_italic_dataset",
    "create_uniform_subset",
    "extract_answer_robust", 
    "analyze_results_by_category",
    "cleanup_gpu_memory",
    "get_gpu_info"
]