import os
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Llama31BenchmarkConfig:
    """Configuration class for Llama 3.1 benchmarking with vLLM"""
    
    # Model configuration
    model_name: str
    min_vram_gb: float  # Minimum VRAM required for this model size
    
    # Dataset configuration
    test_file: str = "test.jsonl"
    max_eval_samples: Optional[int] = None  # None = use full dataset
    use_uniform_subset: bool = True
    random_seed: int = 42
    
    # vLLM configuration
    max_length: int = 30000
    max_new_tokens: int = 350
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    max_num_seqs: int = 256  # Maximum concurrent sequences
    
    # Generation parameters
    temperature: float = 0.0  # 0.0 for deterministic generation
    top_p: float = 1.0  # Set to 1.0 to consider all tokens
    top_k: int = -1  # Set to -1 to consider all tokens  
    repetition_penalty: float = 1.0
    
    # Batching configuration
    batch_size: Optional[int] = None  # Auto-detect if None
    force_batch_size: bool = False  # Override auto-detection
    
    # Prompt configuration
    system_message: str = "Sei un assistente utile."
    query_template: str = """
Rispondi alla seguente domanda a scelta multipla. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

{question}

{options}

Risposta:
""".strip()
    
    # Output configuration
    output_prefix: str = "llama31_benchmark"
    save_results: bool = True
    save_summary: bool = True
    
    # Environment configuration
    hf_token_env_var: str = "HF_TOKEN"
    suppress_warnings: bool = True
    
    def __post_init__(self):
        """Post-initialization setup"""
        
        # Set random seed
        import random
        random.seed(self.random_seed)
        
        # Suppress warnings if requested
        if self.suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        # Get HF token from environment
        self.hf_token = os.getenv(self.hf_token_env_var)
        if self.hf_token:
            print("Using Hugging Face token from environment")
            os.environ["HF_TOKEN"] = self.hf_token
        else:
            print("No HF token found in environment - proceeding without authentication")
        
        # Auto-detect batch size if not specified
        if self.batch_size is None and not self.force_batch_size:
            self.batch_size = self._auto_detect_batch_size()
        
        # Validate configuration
        self._validate_config()
    
    def _auto_detect_batch_size(self) -> int:
        """Auto-detect optimal batch size based on available VRAM and model size"""
        
        if not torch.cuda.is_available():
            print("CUDA not available, using batch size 1")
            return 1
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_memory:.1f} GB")
        print(f"Model requires: {self.min_vram_gb:.1f} GB")
        
        # Check if we have enough VRAM
        if total_memory < self.min_vram_gb:
            raise ValueError(f"Insufficient VRAM: {total_memory:.1f}GB available, {self.min_vram_gb:.1f}GB required")
        
        # Estimate available memory for batching after model loading
        # vLLM uses gpu_memory_utilization fraction
        used_memory = total_memory * self.gpu_memory_utilization
        available_for_batching = total_memory - used_memory
        
        # Determine batch size based on model size and available memory
        # Larger models need more conservative batching
        if self.min_vram_gb >= 140:  # 70B+ models
            batch_size = max(1, min(4, int(available_for_batching // 2)))
        elif self.min_vram_gb >= 32:  # Large models (8B-70B)
            batch_size = max(2, min(8, int(available_for_batching // 1)))
        else:  # Smaller models
            batch_size = max(4, min(16, int(available_for_batching // 0.5)))
        
        print(f"GPU memory utilization: {self.gpu_memory_utilization}")
        print(f"Auto-detected batch size: {batch_size}")
        
        return batch_size
    
    def _validate_config(self):
        """Validate configuration parameters"""
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test file not found: {self.test_file}")
        
        if not (0.1 <= self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be between 0.1 and 1.0")
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("LLAMA 3.1 BENCHMARK CONFIGURATION (vLLM)")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Test file: {self.test_file}")
        print(f"Max samples: {self.max_eval_samples or 'All'}")
        print(f"Batch size: {self.batch_size}")
        print(f"GPU memory utilization: {self.gpu_memory_utilization}")
        print(f"Max concurrent sequences: {self.max_num_seqs}")
        print(f"Temperature: {self.temperature}")
        print(f"Output prefix: {self.output_prefix}")
        print(f"System message: {self.system_message}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({total_memory:.1f}GB)")
        
        print("="*60)