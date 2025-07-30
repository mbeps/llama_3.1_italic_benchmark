import json
import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Any


def load_italic_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the ITALIC dataset from JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_uniform_subset(
    data: List[Dict[str, Any]], n_samples: int
) -> List[Dict[str, Any]]:
    """
    Create a subset with uniform distribution across categories

    Args:
        data: List of questions from ITALIC dataset
        n_samples: Total number of samples to select

    Returns:
        List of selected questions with uniform category distribution
    """
    # Group questions by category if category field exists
    if not data or "category" not in data[0]:
        # If no category field, just return subset
        if len(data) <= n_samples:
            return data
        return random.sample(data, n_samples)

    # Group questions by category
    category_groups = defaultdict(list)
    for question in data:
        category_groups[question["category"]].append(question)

    # Print category distribution in original dataset
    print(f"\nOriginal dataset category distribution:")
    for category, questions in sorted(category_groups.items()):
        print(f"  {category}: {len(questions)} questions")

    categories = list(category_groups.keys())
    n_categories = len(categories)

    # Calculate samples per category
    base_samples_per_category = n_samples // n_categories
    remainder = n_samples % n_categories

    print(f"\nCreating uniform subset with {n_samples} samples:")
    print(f"  {n_categories} categories")
    print(f"  {base_samples_per_category} samples per category (base)")
    print(f"  {remainder} categories get +1 sample")

    selected_questions = []

    for i, category in enumerate(sorted(categories)):
        # Some categories get one extra sample to handle remainder
        samples_for_category = base_samples_per_category + (1 if i < remainder else 0)

        available_questions = category_groups[category]

        # Ensure we don't request more than available
        actual_samples = min(samples_for_category, len(available_questions))

        if actual_samples < samples_for_category:
            print(
                f"  Warning: {category} only has {len(available_questions)} questions, taking all"
            )

        # Randomly sample from this category
        selected = random.sample(available_questions, actual_samples)
        selected_questions.extend(selected)

        print(
            f"  {category}: selected {actual_samples} / {len(available_questions)} questions"
        )

    # Shuffle the final list to mix categories
    random.shuffle(selected_questions)

    print(f"\nFinal subset: {len(selected_questions)} questions")

    # Verify distribution
    final_category_counts = defaultdict(int)
    for question in selected_questions:
        final_category_counts[question["category"]] += 1

    print("Final category distribution:")
    for category, count in sorted(final_category_counts.items()):
        print(f"  {category}: {count} questions")

    return selected_questions


def format_options(options: List[Dict[str, str]]) -> Tuple[str, str]:
    """Format options list into string and extract letters"""
    formatted_options = "\n".join(
        [f"{list(item.keys())[0]}) {list(item.values())[0]}" for item in options]
    )
    letters = "".join([list(item.keys())[0] for item in options])
    return formatted_options, letters


def extract_answer_robust(output: str) -> str:
    """Extract answer letter using robust pattern matching"""
    if not output or len(output.strip()) == 0:
        return ""

    output = output.strip()

    # Method 1: Look for explicit answer patterns
    explicit_patterns = [
        r"risposta[:\s]*([ABCDE])",  # "risposta: A"
        r"lettera[:\s]*([ABCDE])",  # "lettera: A"
        r"opzione[:\s]*([ABCDE])",  # "opzione: A"
    ]

    for pattern in explicit_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Method 2: Look for formatted answers
    format_patterns = [
        r"\b([ABCDE])\)",  # A), B), C)
        r"\b([ABCDE])\.",  # A., B., C.
        r"\b([ABCDE])\b",  # A, B, C as standalone
    ]

    for pattern in format_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Method 3: Simple fallback - first valid letter found
    VALID_ANSWERS = "ABCDE"
    for letter in VALID_ANSWERS:
        if letter.upper() in output.upper():
            return letter

    return ""


def configure_payload_zero_shot(
    question_data: Dict[str, Any], system_message: str, query_template: str
) -> Tuple[List[Dict[str, str]], str]:
    """Configure the message payload for zero-shot evaluation"""
    question = question_data["question"]
    options = question_data["options"]
    answer = question_data["answer"]

    options_str, merged_letters = format_options(options)

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": query_template.format(
                question=question,
                options=options_str,
                merged_letters=merged_letters,
            ),
        },
    ]

    return messages, answer


def analyze_results_by_category(
    results: List[Dict[str, Any]], category_stats: Dict[str, Dict[str, int]]
) -> float:
    """Analyze results by category and return overall accuracy"""

    print(f"\nZERO-SHOT RESULTS BY CATEGORY:")
    print("-" * 60)
    print(f"{'Category':25s} {'Accuracy':>12s} {'Correct/Total':>15s}")
    print("-" * 60)

    our_total_correct = sum(result["is_correct"] for result in results)
    our_total_questions = len(results)
    our_total_accuracy = our_total_correct / our_total_questions * 100

    for category, stats in sorted(category_stats.items()):
        our_accuracy = (
            stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )
        print(
            f"{category:25s} {our_accuracy:11.2f}% {stats['correct']:>6d}/{stats['total']:<6d}"
        )

    print("-" * 60)
    print(
        f"{'TOTAL':25s} {our_total_accuracy:11.2f}% {our_total_correct:>6d}/{our_total_questions:<6d}"
    )

    return our_total_accuracy


def cleanup_gpu_memory():
    """Clean up GPU memory and garbage collection"""
    import torch
    import gc

    print("🧹 Clearing GPU memory...")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU cache cleared")

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Show memory status
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {total_memory:.1f}GB"
        )
        print(f"Available: {total_memory - memory_reserved:.2f}GB")
    else:
        print("❌ CUDA not available")

    print("✅ Memory cleanup completed!")


def get_gpu_info():
    """Get GPU information"""
    import torch

    if not torch.cuda.is_available():
        return None, 0

    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return gpu_name, total_memory
