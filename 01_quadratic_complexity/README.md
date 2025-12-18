# Problem 1: Quadratic Complexity O(N²)

## Problem

Self-attention computes an N×N matrix, causing memory and compute explosion for long sequences.

The self-attention mechanism requires computing attention scores between every pair of tokens in a sequence. For a sequence of length N, this results in O(N²) time and space complexity, making it impractical for very long sequences.

## Solutions

| Solution | Description |
|----------|-------------|
| **Sparse Attention** | Only compute attention for a subset of token pairs using fixed or learned patterns |
| **Longformer** | Combines local sliding window attention with global attention for selected tokens |
| **Linformer** | Projects keys and values to lower dimensions, reducing complexity to O(N) |
| **Performer** | Uses random feature approximation (FAVOR+) for linear attention |
| **FlashAttention** | IO-aware exact attention algorithm that reduces memory reads/writes |
| **Transformer-XL** | Segment-level recurrence with relative positional encodings |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Longformer](https://arxiv.org/abs/2004.05150) - Long Document Transformer
- [Linformer](https://arxiv.org/abs/2006.04768) - Self-Attention with Linear Complexity
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Fast and Memory-Efficient Exact Attention

