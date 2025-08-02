"""
Llama 3.1 Benchmark Package

A modular benchmarking framework for evaluating Llama 3.1 models of various sizes
on the ITALIC dataset with vLLM for efficient batching and automatic GPU optimization.

NEW: Now supports LoRA fine-tuned models with automatic adapter merging!

Features:
- Efficient batching with vLLM for fast inference
- Automatic GPU memory optimization
- Support for base models and LoRA fine-tuned models
- Automatic LoRA adapter merging with base models
- Comprehensive result analysis and export
- Robust answer extraction from model outputs

LoRA Support:
- Automatically detects and loads LoRA adapters
- Merges adapters with base model for optimal performance
- Supports both temporary and persistent merged model saving
- Compatible with all standard LoRA configurations

Requirements:
- vllm >= 0.2.0
- torch >= 2.0.0
- transformers >= 4.30.0
- peft >= 0.5.0 (for LoRA support)
- pandas
- tqdm
- python-dotenv

Usage Examples:

# Basic usage with base model:
config = Llama31BenchmarkConfig(
    model_name="swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
    min_vram_gb=16.0,
    test_file="test.jsonl"
)

# Usage with LoRA fine-tuned model:
config = Llama31BenchmarkConfig(
    model_name="swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
    min_vram_gb=16.0,
    test_file="test.jsonl",
    lora_adapter_path="./path/to/lora/adapter",  # Directory with adapter files
    merged_model_save_path="./merged_model"      # Optional: save merged model
)

benchmark = Llama31Benchmark(config)
results, accuracy, category_stats = benchmark.run_benchmark()
"""

from .config import Llama31BenchmarkConfig
from .benchmark import Llama31Benchmark
from .utils import (
    load_italic_dataset,
    create_uniform_subset,
    extract_answer_robust,
    analyze_results_by_category,
    cleanup_gpu_memory,
    get_gpu_info,
)

__version__ = "1.1.0"
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
    "get_gpu_info",
]


# Convenience function to check LoRA requirements
def check_lora_requirements():
    """Check if LoRA requirements are satisfied"""
    try:
        import peft

        print(f"✅ PEFT library available (version: {peft.__version__})")
        return True
    except ImportError:
        print("❌ PEFT library not found. Install with: pip install peft")
        return False


# Add to __all__
__all__.append("check_lora_requirements")
