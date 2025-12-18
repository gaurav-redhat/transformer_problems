# Problem 8: High Compute & Power Cost

## Problem

Expensive FLOPs. Not edge-friendly. High electricity bills for training.

Training and running large Transformers requires massive computational resources. A single training run for GPT-3 scale models can cost millions of dollars in compute and consume significant energy.

## Solutions

| Solution | Description |
|----------|-------------|
| **Quantization (INT8/INT4)** | Reduce precision of weights and activations |
| **Lightweight Transformers** | Efficient architectures like MobileBERT, TinyBERT |
| **Operator Fusion** | Combine multiple operations to reduce memory bandwidth |
| **Pruning** | Remove unimportant weights or attention heads |

## References

- [LLM.int8()](https://arxiv.org/abs/2208.07339) - 8-bit Matrix Multiplication for Transformers at Scale
- [MobileBERT](https://arxiv.org/abs/2004.02984) - A Compact Task-Agnostic BERT
- [Movement Pruning](https://arxiv.org/abs/2005.07683) - Adaptive Sparsity by Fine-Tuning

