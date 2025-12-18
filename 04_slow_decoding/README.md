# Problem 4: Slow Autoregressive Decoding

## Problem

Decoder generates one token at a time causing high latency. Sequential bottleneck.

Autoregressive generation requires sequential token-by-token generation, where each new token depends on all previous tokens. This creates a fundamental latency bottleneck that cannot be parallelized.

## Solutions

| Solution | Description |
|----------|-------------|
| **KV Cache** | Cache key-value pairs to avoid recomputation during generation |
| **Speculative Decoding** | Use smaller draft model to propose multiple tokens, verify with large model |
| **Non-Autoregressive Models** | Generate all tokens in parallel (with quality trade-offs) |
| **Distillation** | Train smaller, faster models to mimic larger ones |

## References

- [Speculative Decoding](https://arxiv.org/abs/2211.17192) - Fast Inference from Transformers via Speculative Decoding
- [Non-Autoregressive Translation](https://arxiv.org/abs/1711.02281) - Non-Autoregressive Neural Machine Translation

