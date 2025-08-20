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
