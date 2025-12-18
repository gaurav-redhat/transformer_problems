# Problem 10: Training Instability

## Problem

Gradient explosion/vanishing in deep Transformers. Training diverges or stalls.

Deep Transformers are prone to training instabilities, especially at larger scales. Loss spikes, divergence, and gradient issues are common challenges that require careful hyperparameter tuning.

## Solutions

| Solution | Description |
|----------|-------------|
| **Pre-LayerNorm** | Apply layer normalization before attention/FFN instead of after |
| **Learning Rate Warmup** | Gradually increase learning rate at the start of training |
| **AdamW Optimizer** | Adam with decoupled weight decay regularization |
| **Gradient Clipping** | Clip gradients to prevent explosion |

## References

- [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) - On Layer Normalization in the Transformer Architecture
- [AdamW](https://arxiv.org/abs/1711.05101) - Decoupled Weight Decay Regularization
- [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) - Chinchilla Scaling Laws

