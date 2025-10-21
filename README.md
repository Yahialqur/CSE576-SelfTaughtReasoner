# CSE576-SelfTaughtReasoner

An implementation and comparison of Self-Taught Reasoner (STaR) methodology for improving large language model performance on mathematical reasoning tasks. This project demonstrates how self-supervised learning techniques can significantly enhance model reasoning capabilities on the GSM8K dataset.

## Overview

This repository implements three approaches for solving grade-school math word problems:

1. **Zero-Shot Chain-of-Thought (Zero-Shot CoT)**: Baseline evaluation using prompt-based reasoning
2. **Supervised Fine-Tuning (SFT)**: Traditional fine-tuning on labeled training data
3. **Self-Taught Reasoner (STaR)**: Self-improving methodology that generates and learns from its own reasoning chains

## Benchmark Results

| Approach | Accuracy | Correct/Total |
|----------|----------|---------------|
| Zero-Shot CoT | 56.10% | 740/1319 |
| Vanilla SFT | 64.14% | 846/1319 |
| STaR | 79.68% | 1051/1319 |

The STaR approach achieves a **41% relative improvement** over the zero-shot baseline and a **24% relative improvement** over vanilla supervised fine-tuning.

## Project Structure

```
CSE576-SelfTaughtReasoner/
├── zero_shot_COT/           # Zero-Shot CoT baseline
│   ├── zero_shot_COT.py     # Evaluation script
│   └── zs_cot_results.csv   # Results (56.10% accuracy)
│
├── sft/                      # Supervised Fine-Tuning
│   ├── sft.py               # Training script
│   ├── sft_benchmark.py     # Evaluation script
│   └── sft_benchmark_results.csv  # Results (64.14% accuracy)
│
├── star/                     # Self-Taught Reasoner
│   ├── star.py              # STaR dataset generation
│   ├── star_tuner.py        # Training script
│   ├── star_benchmark.py    # Evaluation script
│   ├── star_dataset_final.csv  # Generated training data
│   └── star_benchmark_results.csv  # Results (79.68% accuracy)
│
└── requirements.txt         # Python dependencies
```

## Methodology

### Zero-Shot Chain-of-Thought (Baseline)

Evaluates the base Llama 3.2-3B-Instruct model using zero-shot prompting with instructions to think step-by-step. No fine-tuning is performed.

**Script**: [zero_shot_COT/zero_shot_COT.py](zero_shot_COT/zero_shot_COT.py)

### Supervised Fine-Tuning (SFT)

Fine-tunes the base model on GSM8K training data using standard supervised learning:
- 3 epochs, batch size 8, learning rate 2e-5
- Cosine scheduler with warmup
- Gradient checkpointing and BF16 precision for memory optimization

**Scripts**:
- Training: [sft/sft.py](sft/sft.py)
- Evaluation: [sft/sft_benchmark.py](sft/sft_benchmark.py)

### Self-Taught Reasoner (STaR)

Implements a self-supervised learning approach with two key phases:

1. **Rationale Generation**: For problems the model solves correctly, generate and save the reasoning chain
2. **Rationalization**: For problems the model fails, provide hints (ground truth answer) and generate rationalized solutions

The model is then fine-tuned on this augmented dataset of self-generated reasoning chains.

**Scripts**:
- Dataset Generation: [star/star.py](star/star.py)
- Training: [star/star_tuner.py](star/star_tuner.py)
- Evaluation: [star/star_benchmark.py](star/star_benchmark.py)

**Training Configuration**:
- 3 epochs, batch size 16, gradient accumulation steps 4
- Learning rate 2e-5
- Enhanced memory optimization with TF32 precision

## Dataset

This project uses **GSM8K** (Grade School Math 8K):
- Training set: ~1,300 examples
- Test set: ~1,300 examples
- Format: Math word problems with step-by-step solutions and answers marked as `#### <number>`

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.9.0
- Transformers 4.57.1
- Datasets 4.2.0
- Pandas 2.3.3
- CUDA 12.8 (for GPU support)

## Usage

### 1. Run Zero-Shot CoT Baseline

```bash
cd zero_shot_COT
python zero_shot_COT.py
```

### 2. Train and Evaluate SFT Model

```bash
cd sft
python sft.py                # Train the model
python sft_benchmark.py      # Evaluate on test set
```

### 3. Train and Evaluate STaR Model

```bash
cd star
python star.py               # Generate STaR dataset
python star_tuner.py         # Fine-tune on STaR dataset
python star_benchmark.py     # Evaluate on test set
```

## Model

All approaches use **Llama 3.2-3B-Instruct** (`meta-llama/Llama-3.2-3B-Instruct`) as the base model from HuggingFace.

## Implementation Details

- **Answer Extraction**: Regex-based extraction looking for `#### <number>` format with fallback to the last number in the response
- **Memory Optimization**: Gradient checkpointing, BF16/TF32 precision, efficient batch processing
- **GPU Optimization**: Model compilation with `torch.compile()`, optimized dataloader workers
- **Batch Inference**: Benchmarking scripts use batched inference for efficiency

## Results Analysis

The results demonstrate the effectiveness of self-supervised learning for reasoning tasks:

- **Zero-Shot CoT → SFT**: +8.04% improvement shows benefits of task-specific fine-tuning
- **SFT → STaR**: +15.54% improvement demonstrates the power of learning from self-generated reasoning chains
- **Overall**: STaR achieves 79.68% accuracy, approaching state-of-the-art performance for this model size

## License

This is an educational project for CSE576 (Natural Language Processing).

## Acknowledgments

- Based on the STaR methodology from "STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning" (Zelikman et al.)
- Uses the GSM8K dataset from "Training Verifiers to Solve Math Word Problems" (Cobbe et al.)
- Built with HuggingFace Transformers and Meta's Llama 3.2 model
