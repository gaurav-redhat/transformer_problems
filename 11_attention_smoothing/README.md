# Problem 11: Attention Over-Smoothing

## Problem

Token representations become too similar in deep layers. Loss of information.

As representations pass through many Transformer layers, they tend to converge to similar values (rank collapse). This over-smoothing effect reduces the model's ability to distinguish between tokens.

## Solutions

| Solution | Description |
|----------|-------------|
| **Residual Scaling** | Scale residual connections to prevent representation collapse |
| **DropHead** | Randomly drop attention heads during training |
| **Attention Temperature Control** | Adjust softmax temperature to sharpen or smooth attention |
| **Skip Connections** | Additional skip connections across multiple layers |

## References

- [DeepNet](https://arxiv.org/abs/2203.00555) - Scaling Transformers to 1,000 Layers
- [Understanding Attention](https://arxiv.org/abs/2006.16362) - What Do Vision Transformers Learn?
- [Rank Collapse](https://arxiv.org/abs/2103.03404) - Do Vision Transformers See Like CNNs?

