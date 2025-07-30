import json
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from vllm import LLM, SamplingParams
import gc
import os
import warnings

from .config import Llama31BenchmarkConfig
from .utils import (
    load_italic_dataset, 
    create_uniform_subset,
    configure_payload_zero_shot,
    extract_answer_robust,
    analyze_results_by_category,
    cleanup_gpu_memory,
    get_gpu_info
)


class Llama31Benchmark:
    """Main benchmark class for Llama 3.1 models using vLLM with efficient batching"""
    
    def __init__(self, config: Llama31BenchmarkConfig):
        """Initialize benchmark with configuration"""
        self.config = config
        self.model = None
        self.sampling_params = None
        self.dataset = None
        
        # Setup environment
        self._setup_environment()
        self._validate_hardware()
        
        # Print configuration
        self.config.print_config()
        
        # Clean up GPU memory before starting
        cleanup_gpu_memory()
    
    def _setup_environment(self):
        """Setup environment and load tokens"""
        # Suppress warnings
        if self.config.suppress_warnings:
            warnings.filterwarnings("ignore")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        # Set random seeds
        torch.manual_seed(self.config.random_seed)
        
    def _validate_hardware(self):
        """Validate hardware requirements"""
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA not available! This benchmark requires GPU execution.")
        
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_memory:.1f} GB")
        
        if total_memory < self.config.min_vram_gb:
            raise RuntimeError(f"❌ Insufficient GPU memory: {total_memory:.1f}GB < {self.config.min_vram_gb}GB required")
        
        print(f"✅ GPU has sufficient VRAM for the model")
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory and garbage collect"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @staticmethod
    def print_gpu_memory():
        """Print current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and prepare dataset"""
        print("Loading ITALIC dataset...")
        
        # Load full dataset
        full_dataset = load_italic_dataset(self.config.test_file)
        print(f"Loaded {len(full_dataset)} questions")
        
        # Create subset if needed
        if self.config.max_eval_samples and len(full_dataset) > self.config.max_eval_samples:
            if self.config.use_uniform_subset:
                dataset = create_uniform_subset(full_dataset, self.config.max_eval_samples)
            else:
                dataset = full_dataset[:self.config.max_eval_samples]
            print(f"Using {len(dataset)} questions for evaluation")
        else:
            dataset = full_dataset
            print(f"Using all {len(dataset)} questions for evaluation")
        
        # Display sample question
        if dataset:
            print("\nSample question:")
            sample = dataset[0]
            print(f"Question: {sample['question']}")
            print(f"Options: {sample['options']}")
            print(f"Answer: {sample['answer']}")
            if 'category' in sample:
                print(f"Category: {sample['category']}")
        
        self.dataset = dataset
        return dataset
    
    def load_model(self):
        """Load Llama 3.1 model using vLLM"""
        print(f"🔄 Loading {self.config.model_name} with vLLM...")
        
        # Initialize vLLM engine
        self.model = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_length,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            enforce_eager=False,  # Enable CUDA graphs for better performance
            trust_remote_code=True
        )
        
        # Configure sampling parameters for deterministic generation
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_tokens=self.config.max_new_tokens,
            skip_special_tokens=True,
        )
        
        print(f"✓ vLLM model loaded successfully!")
        print(f"✓ Temperature: {self.config.temperature} (deterministic)" if self.config.temperature == 0.0 else f"✓ Temperature: {self.config.temperature}")
        self.print_gpu_memory()
    
    def generate_batch_responses(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        """Generate responses for a batch of messages using vLLM's efficient batching"""
        
        # Use vLLM's chat method for efficient batch processing
        outputs = self.model.chat(
            batch_messages,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        # Extract generated text from outputs
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text.strip())
        
        return responses
    
    def run_evaluation(self, dataset: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], float, Dict[str, Dict[str, int]]]:
        """Run zero-shot evaluation with efficient vLLM batching"""
        
        if dataset is None:
            dataset = self.dataset
        
        if dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        results = []
        correct = 0
        total = 0
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        print(f"\n🔍 Evaluating {self.config.model_name} on {len(dataset)} questions with batch size {self.config.batch_size}...")
        
        # Process in batches using vLLM's efficient batching
        pbar = tqdm(range(0, len(dataset), self.config.batch_size), desc="Evaluating")
        for batch_start in pbar:
            batch_end = min(batch_start + self.config.batch_size, len(dataset))
            batch_data = dataset[batch_start:batch_end]
            
            try:
                # Prepare batch
                batch_messages = []
                batch_correct_answers = []
                
                for question_data in batch_data:
                    messages, correct_answer = configure_payload_zero_shot(
                        question_data, 
                        self.config.system_message,
                        self.config.query_template
                    )
                    batch_messages.append(messages)
                    batch_correct_answers.append(correct_answer)
                
                # Generate responses for entire batch efficiently
                batch_responses = self.generate_batch_responses(batch_messages)
                
                # Process results
                for i, (question_data, response, correct_answer) in enumerate(zip(batch_data, batch_responses, batch_correct_answers)):
                    predicted = extract_answer_robust(response)
                    is_correct = predicted == correct_answer
                    category = question_data.get('category', 'unknown')
                    
                    category_stats[category]['total'] += 1
                    if is_correct:
                        correct += 1
                        category_stats[category]['correct'] += 1
                    
                    total += 1
                    
                    result = {
                        'index': batch_start + i,
                        'question': question_data['question'],
                        'category': category,
                        'macro_category': question_data.get('macro_category', 'unknown'),
                        'correct_answer': correct_answer,
                        'predicted_answer': predicted,
                        'raw_response': response,
                        'is_correct': is_correct
                    }
                    results.append(result)
                
                # Progress update every 10 batches
                if (batch_start // self.config.batch_size) % 10 == 0:
                    accuracy = correct / total if total > 0 else 0
                    pbar.set_postfix({
                        'Acc': f'{accuracy*100:.1f}%',
                        'Questions': f'{total}/{len(dataset)}'
                    })
                    
                    # Strategic memory cleanup
                    self.clear_gpu_memory()
                    
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Handle individual items in case of batch failure
                for i, question_data in enumerate(batch_data):
                    total += 1
                    category_stats[question_data.get('category', 'unknown')]['total'] += 1
                    
                    result = {
                        'index': batch_start + i,
                        'question': question_data['question'],
                        'category': question_data.get('category', 'unknown'),
                        'macro_category': question_data.get('macro_category', 'unknown'),
                        'correct_answer': question_data['answer'],
                        'predicted_answer': '',
                        'raw_response': f'ERROR: {str(e)}',
                        'is_correct': False
                    }
                    results.append(result)
        
        final_accuracy = correct / total if total > 0 else 0
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        return results, final_accuracy, category_stats
    
    def save_results(self, results: List[Dict[str, Any]], accuracy: float, category_stats: Dict[str, Dict[str, int]]):
        """Save results and summary"""
        
        if not self.config.save_results:
            return
        
        # Get GPU info for filename
        gpu_name, total_memory = get_gpu_info()
        if gpu_name:
            gpu_suffix = gpu_name.lower().replace(' ', '_').replace('nvidia_', '').replace('geforce_', '')
            gpu_suffix = gpu_suffix.split('_')[0]  # Take first part (e.g., rtx3090)
        else:
            gpu_suffix = "cpu"
        
        # Save results CSV
        results_filename = f"{self.config.output_prefix}_{gpu_suffix}_results.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_filename, index=False)
        print(f"\nResults saved to '{results_filename}'")
        
        # Save summary JSON
        if self.config.save_summary:
            summary_filename = f"{self.config.output_prefix}_{gpu_suffix}_summary.json"
            
            overall_accuracy = sum(result['is_correct'] for result in results) / len(results) * 100
            
            summary = {
                'model': self.config.model_name,
                'evaluation_type': 'zero-shot',
                'engine': 'vLLM',
                'quantization': 'none (full precision)',
                'dataset_info': {
                    'test_file': self.config.test_file,
                    'total_questions_tested': len(results),
                    'max_eval_samples': self.config.max_eval_samples
                },
                'generation_config': {
                    'system_message': self.config.system_message,
                    'max_tokens': self.config.max_new_tokens,
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'top_k': self.config.top_k,
                    'repetition_penalty': self.config.repetition_penalty,
                },
                'hardware_config': {
                    'batch_size': self.config.batch_size,
                    'gpu_memory_utilization': self.config.gpu_memory_utilization,
                    'max_num_seqs': self.config.max_num_seqs,
                    'tensor_parallel_size': self.config.tensor_parallel_size,
                    'gpu_name': gpu_name,
                    'total_vram_gb': total_memory
                },
                'results': {
                    'correct_answers': sum(result['is_correct'] for result in results),
                    'accuracy': accuracy,
                    'accuracy_percent': overall_accuracy,
                    'category_results': {cat: {'accuracy': stats['correct']/stats['total'], 
                                              'correct': stats['correct'], 
                                              'total': stats['total']} 
                                        for cat, stats in category_stats.items()}
                }
            }
            
            with open(summary_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to '{summary_filename}'")
    
    def analyze_results_by_category(self, category_stats: Dict[str, Dict[str, int]]) -> float:
        """Analyze and print results by category"""
        
        print(f"\nZERO-SHOT RESULTS BY CATEGORY:")
        print("-" * 60)
        print(f"{'Category':25s} {'Accuracy':>12s} {'Correct/Total':>15s}")
        print("-" * 60)
        
        total_correct = sum(stats['correct'] for stats in category_stats.values())
        total_questions = sum(stats['total'] for stats in category_stats.values())
        overall_accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0
        
        for category, stats in sorted(category_stats.items()):
            accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{category:25s} {accuracy:11.2f}% {stats['correct']:>6d}/{stats['total']:<6d}")
        
        print("-" * 60)
        print(f"{'TOTAL':25s} {overall_accuracy:11.2f}% {total_correct:>6d}/{total_questions:<6d}")
        
        return overall_accuracy
    
    def test_inference(self):
        """Test inference on a single sample"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        print("\n🧪 Testing inference...")
        test_messages, test_answer = configure_payload_zero_shot(
            self.dataset[0], 
            self.config.system_message,
            self.config.query_template
        )
        
        # Test single batch
        test_responses = self.generate_batch_responses([test_messages])
        test_response = test_responses[0]
        test_extracted = extract_answer_robust(test_response)
        
        print(f"Question: {self.dataset[0]['question']}")
        print(f"Expected answer: '{test_answer}'")
        print(f"Raw response: '{test_response}'")
        print(f"Extracted answer: '{test_extracted}'")
        print(f"Correct: {test_extracted == test_answer}")
        
        return test_extracted == test_answer
    
    def run_benchmark(self) -> Tuple[List[Dict[str, Any]], float, Dict[str, Dict[str, int]]]:
        """Run complete benchmark pipeline"""
        
        print(f"\n{'='*60}")
        print("LLAMA 3.1 BENCHMARK (vLLM)")
        print(f"{'='*60}")
        
        # Load dataset
        self.load_dataset()
        
        # Load model
        self.load_model()
        
        # Test inference
        self.test_inference()
        
        # Run evaluation
        print(f"\n{'='*50}")
        print("STARTING EVALUATION")
        print(f"{'='*50}")
        
        results, accuracy, category_stats = self.run_evaluation()
        
        # Analyze results - FIXED: removed the extra 'results' parameter
        overall_accuracy = self.analyze_results_by_category(category_stats)
        
        # Save results
        self.save_results(results, accuracy, category_stats)
        
        # Final cleanup
        print(f"\n🧹 Final GPU cleanup...")
        del self.model
        self.clear_gpu_memory()
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Final GPU memory usage: {final_memory:.2f}GB")
        
        print(f"\n🎉 BENCHMARK COMPLETED!")
        print("=" * 60)
        print(f"📊 Final accuracy: {accuracy:.4f} ({overall_accuracy:.2f}%)")
        print(f"📊 Total questions evaluated: {len(results)}")
        print(f"📊 Batch size used: {self.config.batch_size}")
        print(f"📊 vLLM engine with efficient batching")
        print("=" * 60)
        
        print(f"✅ {self.config.model_name} vLLM benchmark complete! 🚀")
        
        return results, accuracy, category_stats