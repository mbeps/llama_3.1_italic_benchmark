# Llama 3.1 ITALIC Benchmark

This repository contains tooling and evaluation code to benchmark non-reasoning Llama 3.1 models models on the ITALIC multiple-choice dataset. 

## Requirements

- Python 3.10 to 3.12
- GPU recommended (CUDA-enabled) for reasonable performance when running vLLM

Note: adjust your PyTorch/CUDA installation to match your local GPU runtime if you plan to run parts of the code that require PyTorch. The project primarily uses vLLM for inference.

## Stack

- [vLLM](https://github.com/vllm-project/vllm): efficient inference engine used to run LLMs.
- [PEFT (LoRA)](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning utilities (LoRA) for merging and evaluating adapters.
- [pandas](https://pandas.pydata.org/): data inspection and tabulation.
- [tqdm](https://github.com/tqdm/tqdm): progress bars for loops and batch processing.
- [Jupyter](https://jupyter.org/): interactive notebooks for experimentation and visualization.
- [python-dotenv](https://github.com/theskumar/python-dotenv): load environment variables from a .env file.
- [Transformers](https://github.com/huggingface/transformers): model and tokenizer utilities (optional; useful when working outside vLLM).
- [PyTorch](https://pytorch.org/): core deep learning library (install a wheel that matches your CUDA runtime if you need GPU-enabled PyTorch).
- [datasets](https://github.com/huggingface/datasets): dataset utilities and I/O.
- [scikit-learn](https://scikit-learn.org/): evaluation utilities and metrics.
- [TRL (trl)](https://github.com/lvwerra/trl): training utilities for SFT / policy learning.

## Dependencies

Install the primary runtime dependencies (example):

```bash
pip install pandas tqdm vllm transformers peft python-dotenv accelerate einops safetensors jupyter 
```

Recommended additional packages depending on your workflow and whether you run PyTorch-based tooling:

- torch (install a wheel that matches your CUDA runtime)
- transformers
- datasets
- scikit-learn
- trl

## Set Up

Two simple ways to set up the project: pip+virtualenv or Poetry. The minimal instructions below use the packages from `dependences.txt`.

### 1) pip + virtualenv

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install core dependencies (adjust PyTorch wheel to match your CUDA version if you need GPU-enabled PyTorch):

```bash
pip install --upgrade pip
pip install vllm python-dotenv pandas tqdm jupyter peft
```

### 2) Poetry (optional)

If you prefer Poetry, create a virtual environment and add the same dependencies. Example:

```bash
poetry init --no-interaction
poetry add vllm python-dotenv pandas tqdm jupyter peft
# add torch/transformers as needed depending on your CUDA and workflow
```

## Configuration

Place environment values (for example HF tokens or custom settings) in a `.env` file at the repo root. The benchmark loader uses `python-dotenv` to load these values.

## Saving results

Benchmark output and summaries are written to the `results/` folder. The `run_benchmark()` flow saves a detailed CSV (`*_results.csv`) and a JSON summary (`*_summary.json`).

## References
- [ITALIC: An Italian Culture-Aware Natural Language Benchmark](https://aclanthology.org/2025.naacl-long.68.pdf)