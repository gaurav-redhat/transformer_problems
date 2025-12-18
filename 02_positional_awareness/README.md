# Problem 2: No Positional Awareness

## Problem

Transformer has no inherent notion of token order. 'Dog bites man' = 'Man bites dog'.

Unlike RNNs which process sequences step-by-step, the self-attention mechanism is permutation-invariant. Without explicit positional information, the model cannot distinguish between different orderings of the same tokens.

## Solutions

| Solution | Description |
|----------|-------------|
| **Sinusoidal Positional Encoding** | Fixed encodings using sine/cosine functions at different frequencies |
| **Learnable PE** | Position embeddings learned during training |
| **Relative PE** | Encodes relative distances between tokens rather than absolute positions |
| **RoPE** | Rotary Position Embedding - encodes position through rotation matrices |
| **ALiBi** | Attention with Linear Biases - adds linear bias based on distance |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Sinusoidal PE
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [ALiBi](https://arxiv.org/abs/2108.12409) - Train Short, Test Long

