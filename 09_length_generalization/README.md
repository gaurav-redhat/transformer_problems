# Problem 9: Poor Length Generalization

## Problem

Fails when sequence length exceeds training length. Positional encodings break.

Models trained on sequences up to length N often fail catastrophically on sequences longer than N. Absolute positional encodings extrapolate poorly to unseen positions.

## Solutions

| Solution | Description |
|----------|-------------|
| **Relative Position Encoding** | Encode relative distances instead of absolute positions |
| **RoPE Scaling** | Scale rotary embeddings to handle longer sequences |
| **ALiBi** | Linear attention biases that naturally extrapolate |
| **Length Extrapolation** | Training techniques for better length generalization |

## References

- [ALiBi](https://arxiv.org/abs/2108.12409) - Train Short, Test Long
- [RoPE Scaling](https://arxiv.org/abs/2306.15595) - Extending Context Window of LLMs via Position Interpolation
- [YaRN](https://arxiv.org/abs/2309.00071) - Efficient Context Window Extension

