# Problem 7: High Memory Footprint

## Problem

Q, K, V tensors and KV cache consume large memory. GPU OOM errors common.

The attention mechanism requires storing Query, Key, and Value matrices for all layers. During inference, the KV cache grows linearly with sequence length, often causing out-of-memory errors on GPUs.

## Solutions

| Solution | Description |
|----------|-------------|
| **FlashAttention** | Tiled attention computation that minimizes memory transfers |
| **KV Cache Optimization** | Techniques like PagedAttention, grouped-query attention |
| **Gradient Checkpointing** | Trade compute for memory by recomputing activations |
| **Memory-Efficient Attention** | Various implementations that reduce peak memory usage |

## References

- [FlashAttention](https://arxiv.org/abs/2205.14135) - Fast and Memory-Efficient Exact Attention
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for LLM Serving
- [Gradient Checkpointing](https://arxiv.org/abs/1604.06174) - Training Deep Nets with Sublinear Memory Cost

