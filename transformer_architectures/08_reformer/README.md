# Reformer

[← Back](../README.md) | [← Prev: Performer](../07_performer/README.md) | [Next: Longformer →](../09_longformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb)

---

![Architecture](architecture.png)

Reformer tackles two problems at once: the O(N²) attention **and** the memory explosion from storing activations. It uses LSH (locality-sensitive hashing) for fast attention and reversible layers to avoid storing activations.

Together, they let you process 64K tokens on a single GPU.

---

## Problem 1: O(N²) attention

Standard attention: every query compares to every key. Most of those comparisons are pointless - the softmax makes most weights near zero.

Reformer's insight: **similar queries probably have similar keys**. So hash them into buckets and only compute attention within buckets.

---

## LSH Attention

LSH (locality-sensitive hashing) is a hash function where similar items collide:
```
P(hash(x) = hash(y)) ∝ similarity(x, y)
```

For attention, we use angular LSH:
```
hash(x) = sign(x · r)  (r is a random vector)
```

The algorithm:
1. Hash all queries and keys
2. Sort by hash bucket
3. Only compute attention within buckets

If buckets are size √N on average, you get O(N × √N) = O(N^1.5) instead of O(N²).

---

## Problem 2: Activation memory

During training, you need to store activations at every layer for backprop:
```
Memory = O(N × L × d)   where L = number of layers
```

For 64K tokens and 12 layers, that's gigabytes of activation memory. Gradient checkpointing helps, but you're still bounded.

---

## Reversible layers

Reformer uses **reversible residual networks**. The idea: if you can compute outputs from inputs, can you compute inputs from outputs?

Split the hidden state into two parts (x₁, x₂):
```
Forward:
y₁ = x₁ + Attention(x₂)
y₂ = x₂ + FFN(y₁)

Backward (reconstruct inputs):
x₂ = y₂ - FFN(y₁)
x₁ = y₁ - Attention(x₂)
```

You only store the **final layer's activations**. During backprop, you reconstruct earlier layers on the fly.

Memory drops from O(N × L × d) to O(N × d). A factor of L savings.

---

## Code

LSH hashing:

```python
def lsh_hash(x, n_hashes, n_buckets):
    d = x.shape[-1]
    projections = torch.randn(n_hashes, d, n_buckets // 2)
    
    dots = torch.einsum('...d,hdb->...hb', x, projections)
    buckets = (dots > 0).int()
    
    # Combine hash values
    powers = 2 ** torch.arange(n_buckets // 2)
    return (buckets * powers).sum(dim=-1)
```

Reversible layer:

```python
class ReversibleBlock(nn.Module):
    def __init__(self, attn, ffn):
        self.attn = attn
        self.ffn = ffn
    
    def forward(self, x1, x2):
        y1 = x1 + self.attn(x2)
        y2 = x2 + self.ffn(y1)
        return y1, y2
    
    def backward_pass(self, y1, y2):
        x2 = y2 - self.ffn(y1)
        x1 = y1 - self.attn(x2)
        return x1, x2
```

---

## The tradeoffs

**LSH attention:**
- Pro: O(N log N) instead of O(N²)
- Con: May miss important key-value pairs (hash collision isn't perfect)
- Con: Multiple hash rounds needed for reliability

**Reversible layers:**
- Pro: Huge memory savings
- Con: Extra compute during backprop (recomputation)
- Con: More complex implementation

---

## Shared Q/K

One more trick: Reformer shares queries and keys (Q = K). This makes LSH bucketing more meaningful - you're hashing queries and keys with the same function, so they naturally end up in the same buckets.

This loses some expressivity but makes the whole thing work better.

---

## In practice

Reformer was important research, but it's complex to implement correctly. In practice:

- **For long sequences**: People now use FlashAttention or Longformer
- **For memory**: Gradient checkpointing is simpler and works well

Still, understanding Reformer teaches you about efficient attention design.

---

## Papers

- [Reformer](https://arxiv.org/abs/2001.04451) (2020) - Original
- [RevNets](https://arxiv.org/abs/1707.04585) (2017) - Reversible networks

---

## Try it

The notebook implements LSH attention, builds reversible layers, compares memory usage, and visualizes bucket assignments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb)
