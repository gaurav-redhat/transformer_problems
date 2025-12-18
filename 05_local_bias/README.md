# Problem 5: No Local Inductive Bias

## Problem

No bias toward local patterns. Hurts vision and audio tasks where locality matters.

Unlike CNNs which have built-in locality through convolutions, Transformers treat all positions equally. This lack of inductive bias means they need more data to learn local patterns that are crucial for images and audio.

## Solutions

| Solution | Description |
|----------|-------------|
| **CNN + Transformer** | Hybrid architectures combining convolutional and attention layers |
| **Swin Transformer** | Shifted window attention for hierarchical local-global modeling |
| **Conformer** | Combines convolution and self-attention for speech recognition |
| **Hierarchical ViT** | Multi-scale vision transformers with local attention at early layers |

## References

- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Hierarchical Vision Transformer using Shifted Windows
- [Conformer](https://arxiv.org/abs/2005.08100) - Convolution-augmented Transformer for Speech Recognition
- [CoAtNet](https://arxiv.org/abs/2106.04803) - Marrying Convolution and Attention

