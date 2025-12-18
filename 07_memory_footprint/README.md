# Problem 7: High Memory Footprint

[← Back to Main](../README.md) | [← Previous](../06_data_hungry/README.md) | [Next →](../08_compute_cost/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/07_memory_footprint/demo.ipynb)

---

![Problem 7](./problem.png)

## What's the Problem?

Ever seen "CUDA out of memory" when running a transformer? Welcome to the club.

Here's what's eating your GPU memory:
- Model weights (billions of parameters × 2-4 bytes each)
- Activations (saved for backprop)
- The attention matrix (N² elements per layer per head)
- KV cache during inference (grows with sequence length)

A 7B parameter model needs ~14GB just for weights in fp16. Add activations and attention matrices during training, and you're looking at 40-80GB easily.

## The KV Cache Problem

During inference, the KV cache grows linearly with sequence length:
```
KV cache size = 2 × layers × heads × d_head × seq_len × batch_size
```

For a 70B model with 128K context? That's ~40GB just for the KV cache. Per request.

## Why This Matters

- **Training**: Limits batch size, which hurts convergence
- **Inference**: Limits concurrent users, increases cost
- **Edge deployment**: Forget about it

## How Do We Fix It?

| Approach | What It Does |
|----------|--------------|
| **FlashAttention** | Computes attention without materializing the full N² matrix |
| **Gradient Checkpointing** | Don't save all activations — recompute them during backward pass |
| **PagedAttention** | Manage KV cache like virtual memory (used in vLLM) |
| **GQA/MQA** | Share KV heads across query heads — smaller cache |
| **Quantization** | 4-bit weights use 4x less memory than 16-bit |

## FlashAttention: The Game Changer

Traditional attention:
1. Compute QK^T (N² matrix)
2. Store it
3. Apply softmax
4. Multiply by V

FlashAttention:
1. Process in tiles that fit in SRAM
2. Never materialize full N² matrix in HBM
3. Same exact result, fraction of the memory

This is now the default in most frameworks.

## Learn More

- [FlashAttention](https://arxiv.org/abs/2205.14135) — The paper
- [vLLM](https://arxiv.org/abs/2309.06180) — PagedAttention for serving

---

[← Back to Main](../README.md) | [← Previous](../06_data_hungry/README.md) | [Next →](../08_compute_cost/README.md)
