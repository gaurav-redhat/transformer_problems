# Problem 17: Hardware Inefficiency

[← Back to Main](../README.md) | [← Previous](../16_dense_inputs/README.md) | [Next →](../18_realtime_deployment/README.md)

---

![Problem 17](./problem.png)

## What's the Problem?

GPUs are insanely fast at matrix multiplication — A100s can do 312 TFLOPS. But during attention, they often sit idle at 30-50% utilization.

Why? Memory bandwidth. The GPU can compute faster than it can read/write data. Attention requires reading Q, K, V matrices from memory, computing scores, writing them back, reading again for softmax... lots of memory traffic.

This is called being **memory-bound** (vs. **compute-bound**).

## The Memory Hierarchy

```
SRAM (on-chip): ~20MB, ~19 TB/s bandwidth
HBM (GPU memory): ~80GB, ~2 TB/s bandwidth
```

SRAM is fast but tiny. HBM is big but slow (relatively). Standard attention constantly shuffles data between them.

## The Attention Bottleneck

Standard attention:
1. Load Q, K from HBM → SRAM
2. Compute QK^T → Write to HBM
3. Load QK^T from HBM → SRAM  
4. Compute softmax → Write to HBM
5. Load attention weights and V → SRAM
6. Compute output → Write to HBM

That N×N attention matrix gets written and read multiple times. For long sequences, this dominates runtime.

## How Do We Fix It?

| Approach | What It Does |
|----------|--------------|
| **FlashAttention** | Fuse all attention operations, never materialize N×N matrix |
| **Fused Kernels** | Combine multiple operations into single GPU kernel |
| **xFormers** | Library of memory-efficient transformer components |
| **Custom CUDA** | Hand-written kernels optimized for specific hardware |

## FlashAttention: The Fix

FlashAttention restructures the computation:
1. Process attention in tiles that fit in SRAM
2. Compute softmax incrementally (online softmax)
3. Never write the full N×N matrix to HBM

Same math, same output, but 2-4x faster and uses way less memory. It's now the default in most frameworks.

## The Broader Lesson

Algorithmic improvements alone aren't enough. You need **IO-aware algorithms** that account for hardware realities. The "optimal" algorithm on paper might be slower than a "worse" one that plays nice with the memory hierarchy.

## Learn More

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Even faster
- [xFormers](https://github.com/facebookresearch/xformers)

---

[← Back to Main](../README.md) | [← Previous](../16_dense_inputs/README.md) | [Next →](../18_realtime_deployment/README.md)
