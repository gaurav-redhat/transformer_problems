# Transformer-XL: Beyond Fixed-Length Context

[← Back to Architectures](../README.md) | [← Previous: ViT](../04_vision_transformer/README.md) | [Next: Sparse Transformer →](../06_sparse_transformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb)

---

![Architecture](architecture.png)

## What is it?

**Transformer-XL** (2019) solves the **fixed context length** problem. Standard transformers process fixed-length segments independently, losing information across segment boundaries. Transformer-XL introduces **segment-level recurrence** to capture longer dependencies.

## The Problem

Standard Transformer:
```
Segment 1: [tokens 1-512]  → process → forget
Segment 2: [tokens 513-1024] → process → no memory of segment 1!
```

This causes **context fragmentation** - the model can't learn dependencies across segments.

## The Solution: Segment-Level Recurrence

Transformer-XL:
```
Segment 1: [tokens 1-512]  → process → cache hidden states
Segment 2: [tokens 513-1024] → process with access to cached states
```

Hidden states from the previous segment are **cached** and used as extended context.

## Architecture

```
Segment t-1                    Segment t (current)
    ↓                              ↓
[Hidden states]  ----CACHE--->  [Attention]
    ↓                              ↓
  (stop gradient)           Attends to both memory and current
```

### Key Components

1. **Memory Cache**: Store hidden states from previous segment
2. **Extended Context**: Current segment attends to memory + self
3. **Relative Position Encoding**: Since absolute positions don't work across segments

## The Math

### Attention with Memory

For layer n, segment τ:

```
h̃_τ^(n-1) = [SG(m_τ^(n-1)) ∘ h_τ^(n-1)]    (concatenate memory + current)

q_τ^n = h_τ^(n-1) × W_q                      (query from current only)
k_τ^n = h̃_τ^(n-1) × W_k                     (key from memory + current)
v_τ^n = h̃_τ^(n-1) × W_v                     (value from memory + current)
```

SG = stop gradient (don't backprop through memory)

### Relative Positional Encoding

Absolute positions don't work across segments (position 0 in segment 2 ≠ position 0 in segment 1).

Instead, encode **relative positions** (how far apart are two tokens):

```
A_ij = q_i^T k_j + q_i^T W_R R_{i-j} + u^T k_j + v^T R_{i-j}
       ├─────┘   ├───────────────┘   ├─────┘   ├───────────┘
       content   position-key        global    global position
```

Where:
- R_{i-j} = sinusoidal encoding of relative distance
- u, v = learnable bias vectors

## Complexity

| Aspect | Standard | Transformer-XL |
|--------|----------|----------------|
| Memory per segment | O(L²) | O(L²) |
| Effective context | L | L × N_segments |
| Evaluation speed | Normal | ~1800x faster (with cache) |

The 1800x speedup comes from not recomputing hidden states for cached tokens.

## Code Highlights

```python
class TransformerXLAttention(nn.Module):
    def forward(self, h, memory=None):
        # Concatenate memory with current hidden states
        if memory is not None:
            cat = torch.cat([memory, h], dim=1)
        else:
            cat = h
        
        # Query from current, Key/Value from memory+current
        Q = self.W_q(h)
        K = self.W_k(cat)  # Extended context!
        V = self.W_v(cat)
        
        # Compute attention with relative positions
        attn = self.relative_attention(Q, K, V, self.R)
        return attn

class TransformerXL(nn.Module):
    def forward(self, x, memories=None):
        if memories is None:
            memories = [None] * self.n_layers
        
        new_memories = []
        for layer, mem in zip(self.layers, memories):
            # Cache current hidden state for next segment
            new_memories.append(h.detach())  # Stop gradient
            h = layer(h, memory=mem)
        
        return self.head(h), new_memories
```

## Key Findings

1. **Context matters**: Longer context = better perplexity on language modeling
2. **Relative positions work**: Better than absolute for long sequences
3. **Memory is efficient**: 1800x speedup at evaluation time
4. **State-of-the-art** (at the time): Best results on WikiText-103, enwik8

## Effective Context Length

| Model | Effective Context |
|-------|-------------------|
| Standard Transformer | 512 tokens |
| Transformer-XL | 512 × N segments |

With memory length M and segment length L:
- Effective context = L + (N-1) × M
- With 4 layers and M=L: context grows ~900 tokens

## Transformer-XL vs Standard

| Aspect | Standard | Transformer-XL |
|--------|----------|----------------|
| Context | Fixed | Extended via memory |
| Position encoding | Absolute | Relative |
| Segment boundary | No information flow | Cached hidden states |
| Evaluation | Recompute everything | Reuse cached states |

## Key Papers

- [Transformer-XL](https://arxiv.org/abs/1901.02860) (2019) - Original
- [Compressive Transformer](https://arxiv.org/abs/1911.05507) (2019) - Compressed memory

## Try It

Run the notebook to:
1. Build Transformer-XL with memory
2. Compare vs standard transformer
3. Visualize extended context
4. See the evaluation speedup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb)

