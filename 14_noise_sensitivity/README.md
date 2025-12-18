# Problem 14: Sensitivity to Noise Tokens

## Problem

Global attention attends to irrelevant tokens. Wastes capacity on noise.

The global attention mechanism allows every token to attend to every other token, including irrelevant or noisy tokens. This can distract the model and waste computational resources.

## Solutions

| Solution | Description |
|----------|-------------|
| **Sparse Attention** | Limit attention to relevant token subsets |
| **Token Pruning** | Remove less important tokens during inference |
| **Attention Masking** | Learn to mask out irrelevant attention patterns |
| **Gating Mechanisms** | Learn to gate attention based on relevance |

## References

- [Sparse Transformers](https://arxiv.org/abs/1904.10509) - Generating Long Sequences with Sparse Transformers
- [DynamicViT](https://arxiv.org/abs/2106.02034) - Efficient Vision Transformers with Dynamic Token Sparsification
- [Token Merging](https://arxiv.org/abs/2210.09461) - Your ViT is Secretly a Hybrid

