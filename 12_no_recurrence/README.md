# Problem 12: No Recurrence / Streaming

## Problem

Hard to use for streaming or online inference. Must process full sequence.

Standard Transformers require the entire input sequence to compute attention, making them unsuitable for streaming applications where data arrives continuously.

## Solutions

| Solution | Description |
|----------|-------------|
| **Transformer-XL** | Segment-level recurrence for processing sequences in chunks |
| **Chunk-based Attention** | Process input in fixed-size chunks with cross-chunk attention |
| **Streaming Transformers** | Architectures designed for online/streaming inference |
| **Delta Attention** | Incremental attention updates for streaming scenarios |

## References

- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Attentive Language Models Beyond a Fixed-Length Context
- [Streaming Transformer](https://arxiv.org/abs/2010.11395) - Streaming Transformer for Hardware Efficient Voice Conversion
- [Emformer](https://arxiv.org/abs/2010.10759) - Efficient Memory Transformer for Streaming Speech Recognition

