import os
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Llama31BenchmarkConfig:
    """Configuration class for Llama 3.1 benchmarking with vLLM and LoRA support"""
    
    # Model configuration
    model_name: Optional[str] = None  # Optional when using LoRA adapters
    min_vram_gb: float = 16.0  # Minimum VRAM required for this model size
    
    # LoRA fine-tuning support
    lora_adapter_path: Optional[str] = None  # Path to LoRA adapter directory
    merge_adapter_in_memory: bool = True  # Merge adapter in memory vs save to disk first
    merged_model_save_path: Optional[str] = None  # Optional path to save merged model
    
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
        
        # Handle LoRA configuration and model name resolution
        if self.lora_adapter_path:
            self._validate_lora_config()
            self._resolve_model_name_from_adapter()
        else:
            if not self.model_name:
                raise ValueError("model_name is required when not using LoRA adapters")
        
        # Auto-detect batch size if not specified
        if self.batch_size is None and not self.force_batch_size:
            self.batch_size = self._auto_detect_batch_size()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_lora_config(self):
        """Validate LoRA adapter configuration"""
        if not os.path.exists(self.lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter path not found: {self.lora_adapter_path}")
        
        # Check for required LoRA files
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.lora_adapter_path, file)):
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing LoRA adapter files: {missing_files}")
        
        print(f"✅ LoRA adapter found at: {self.lora_adapter_path}")
    
    def _resolve_model_name_from_adapter(self):
        """Read base model name from adapter config and set as model_name"""
        import json
        
        adapter_config_path = os.path.join(self.lora_adapter_path, "adapter_config.json")
        
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError("base_model_name_or_path not found in adapter_config.json")
            
            # If model_name was provided, verify it matches the adapter
            if self.model_name and self.model_name != base_model_name:
                print(f"⚠️  Warning: Provided model_name '{self.model_name}' differs from adapter's base model '{base_model_name}'")
                print(f"📝 Using base model from adapter config: {base_model_name}")
            
            # Set model_name from adapter config
            self.model_name = base_model_name
            print(f"📋 Base model resolved from adapter: {self.model_name}")
            
            # Set output prefix to include lora info
            adapter_name = os.path.basename(self.lora_adapter_path.rstrip('/'))
            self.output_prefix = f"{self.output_prefix}_lora_{adapter_name}"
            
        except Exception as e:
            raise ValueError(f"Failed to read adapter configuration: {e}")
    
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
        # Reduce batch size if using LoRA (merger process uses extra memory)
        memory_multiplier = 0.8 if self.lora_adapter_path else 1.0
        
        if self.min_vram_gb >= 140:  # 70B+ models
            batch_size = max(1, min(4, int(available_for_batching * memory_multiplier // 2)))
        elif self.min_vram_gb >= 32:  # Large models (8B-70B)
            batch_size = max(2, min(8, int(available_for_batching * memory_multiplier // 1)))
        else:  # Smaller models
            batch_size = max(4, min(16, int(available_for_batching * memory_multiplier // 0.5)))
        
        print(f"GPU memory utilization: {self.gpu_memory_utilization}")
        if self.lora_adapter_path:
            print(f"LoRA adapter detected - applying conservative batch sizing")
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
        if self.lora_adapter_path:
            print("🔄 LoRA ADAPTER SUPPORT ENABLED")
        print("="*60)
        print(f"Model: {self.model_name}")
        
        if self.lora_adapter_path:
            print(f"LoRA Adapter: {self.lora_adapter_path}")
            print(f"Merge in memory: {self.merge_adapter_in_memory}")
            print(f"Base model from adapter config: {self.model_name}")
            if self.merged_model_save_path:
                print(f"Save merged model to: {self.merged_model_save_path}")
        
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