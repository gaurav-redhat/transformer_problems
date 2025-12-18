# Transformer-XL

[← Back](../README.md) | [← Prev: ViT](../04_vision_transformer/README.md) | [Next: Sparse Transformer →](../06_sparse_transformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb)

---

![Architecture](architecture.png)

Standard transformers have a hard context limit. Whatever fits in the window is all the model can see - nothing before that exists. Transformer-XL fixes this by caching hidden states between segments, so information can flow across the boundary.

---

## The problem

When you process long documents with a standard transformer, you split into chunks:

```
Segment 1: [tokens 1-512]   → process → forget everything
Segment 2: [tokens 513-1024] → process → no idea what happened in segment 1
```

This is called **context fragmentation**. The model can't learn dependencies across segment boundaries - it doesn't know they exist.

---

## The solution: memory

Transformer-XL caches the hidden states from the previous segment:

```
Segment 1: [tokens 1-512]   → process → save hidden states
Segment 2: [tokens 513-1024] → process with access to segment 1's states
```

When computing attention in segment 2, the keys and values include the cached states from segment 1. So token 513 can attend to token 512, even though they're in different segments.

---

## How it works

For each layer:
1. Cache hidden states from the previous segment (with gradient detached)
2. Concatenate cached states with current states
3. Query from current, key/value from concatenated
4. Save current states for next segment

```python
def forward(self, x, memory=None):
    if memory is not None:
        cat = torch.cat([memory, x], dim=1)  # Extended context
    else:
        cat = x
    
    Q = self.W_q(x)       # Query from current only
    K = self.W_k(cat)     # Key from memory + current
    V = self.W_v(cat)     # Value from memory + current
    
    return attention(Q, K, V)
```

---

## Relative positional encoding

Here's a problem: if we use absolute positions, position 0 in segment 2 is the same as position 0 in segment 1. That's wrong - they're different tokens.

Transformer-XL uses **relative positions** instead. Instead of "this is position 5", it encodes "this token is 3 positions after that token".

```
A_ij = q_i · k_j + q_i · R_{i-j} + u · k_j + v · R_{i-j}
       ^^^^^^^   ^^^^^^^^^^^^^   ^^^^^^^   ^^^^^^^^^^^
       content   relative pos    global    global pos
```

This is more complex than absolute encoding, but it works across segment boundaries.

---

## Effective context

With memory length M and segment length L:

| Layers | Memory | Effective Context |
|--------|--------|-------------------|
| 1 | L | L + L = 2L |
| 4 | L | L + 4L = 5L |
| 6 | L | L + 6L = 7L |

The context grows with the number of layers. With 4 layers and 512 segment length, you get ~2500 tokens of effective context.

---

## The speedup at inference

During training, Transformer-XL is similar speed to standard (you still compute everything). But at inference, it's **much faster**.

Standard transformer: Process the entire context from scratch for each new token.

Transformer-XL: Reuse cached states. Only compute for new tokens.

The paper reports ~1800x speedup on evaluation. That's not a typo.

---

## Code

```python
class TransformerXL(nn.Module):
    def forward(self, x, memories=None):
        if memories is None:
            memories = [None] * self.n_layers
        
        new_memories = []
        for layer, mem in zip(self.layers, memories):
            new_memories.append(x.detach())  # Cache for next segment
            x = layer(x, memory=mem)
        
        return x, new_memories
```

---

## Why it matters

Transformer-XL introduced two ideas that are everywhere now:

1. **Recurrence through caching** - GPT and other models use KV cache for fast inference
2. **Relative positional encoding** - Led to RoPE and ALiBi, used in modern LLMs

Even if you don't use Transformer-XL directly, its ideas are in every production model.

---

## Papers

- [Transformer-XL](https://arxiv.org/abs/1901.02860) (2019) - Original
- [Compressive Transformer](https://arxiv.org/abs/1911.05507) (2019) - Compressed memory

---

## Try it

The notebook implements Transformer-XL with memory, compares context length vs standard transformer, and shows the evaluation speedup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb)
