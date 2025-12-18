# Sparse Transformer

[← Back to Architectures](../README.md) | [← Previous: Transformer-XL](../05_transformer_xl/README.md) | [Next: Performer →](../07_performer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb)

---

![Architecture](architecture.png)

## What is it?

**Sparse Transformer** (OpenAI, 2019) addresses the O(N²) attention bottleneck by using **sparse attention patterns**. Instead of every token attending to every other token, tokens attend to a carefully chosen subset.

## The Problem

Standard attention: O(N²)
```
N = 1024  → 1M attention computations
N = 4096  → 16M attention computations
N = 16384 → 268M attention computations
```

This quadratic scaling makes long sequences prohibitively expensive.

## The Solution: Sparse Patterns

Instead of dense attention, use patterns where each token only attends to O(√N) other tokens:

```
Total: O(N × √N) = O(N^1.5)
```

## Attention Patterns

### 1. Local (Sliding Window)

Each token attends to nearby tokens within a window:

```
Token 5 attends to: [3, 4, 5, 6, 7]  (window size = 5)
```

Good for: Local context, syntax, nearby relationships

### 2. Strided

Each token attends to every k-th token:

```
Token 8 attends to: [0, 4, 8]  (stride = 4)
```

Good for: Long-range dependencies, periodic patterns

### 3. Fixed

Attend to specific positions (e.g., first few tokens):

```
All tokens attend to: [0, 1, 2, 3] (summary tokens)
```

Good for: Global information aggregation

### 4. Combined (Factorized)

The Sparse Transformer uses **factorized attention** - alternating between patterns:

```
Layer 1: Local attention (window)
Layer 2: Strided attention
Layer 3: Local attention
Layer 4: Strided attention
...
```

## The Math

### Standard Dense Attention

```
A_ij = 1 for all i, j  (attend to everything)
Complexity: O(N²)
```

### Sparse Attention

```
A_ij = 1 only if j ∈ S(i)  (attend to subset)
|S(i)| = O(√N)
Complexity: O(N√N)
```

### Factorized Pattern

Split positions into two patterns p = (p₁, p₂) where:
- p₁: Local pattern (attend to previous √N tokens)
- p₂: Strided pattern (attend to every √N-th token)

```
A^(1)_ij = 1 if floor(j/√N) = floor(i/√N)    (same block)
A^(2)_ij = 1 if j mod √N = i mod √N           (same column)
```

## Complexity Comparison

| Sequence Length | Dense O(N²) | Sparse O(N√N) | Speedup |
|-----------------|-------------|---------------|---------|
| 1024 | 1M | 32K | 32x |
| 4096 | 16M | 256K | 64x |
| 16384 | 268M | 2M | 128x |

## Code Highlights

```python
def sparse_attention_mask(seq_len, window_size, stride):
    """Create sparse attention mask (local + strided)."""
    mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        # Local: attend to nearby tokens
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
        
        # Strided: attend to every stride-th token
        for j in range(0, i + 1, stride):
            mask[i, j] = 1
    
    return mask

class SparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, window_size, stride):
        super().__init__()
        # ... projection layers ...
        
        # Pre-compute sparse mask
        mask = sparse_attention_mask(seq_len, window_size, stride)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        attn_scores = Q @ K.T / sqrt(d_k)
        
        # Apply sparse mask
        attn_scores = attn_scores.masked_fill(self.mask == 0, -inf)
        attn = softmax(attn_scores)
        
        return attn @ V
```

## Visualizing Sparse Patterns

```
Dense (Full):        Local:              Strided:            Combined:
■ ■ ■ ■ ■ ■ ■ ■     ■ ■ □ □ □ □ □ □     ■ □ □ □ ■ □ □ □     ■ ■ □ □ ■ □ □ □
■ ■ ■ ■ ■ ■ ■ ■     ■ ■ ■ □ □ □ □ □     □ ■ □ □ □ ■ □ □     ■ ■ ■ □ □ ■ □ □
■ ■ ■ ■ ■ ■ ■ ■     □ ■ ■ ■ □ □ □ □     □ □ ■ □ □ □ ■ □     □ ■ ■ ■ □ □ ■ □
■ ■ ■ ■ ■ ■ ■ ■     □ □ ■ ■ ■ □ □ □     □ □ □ ■ □ □ □ ■     □ □ ■ ■ ■ □ □ ■
■ ■ ■ ■ ■ ■ ■ ■     □ □ □ ■ ■ ■ □ □     ■ □ □ □ ■ □ □ □     ■ □ □ ■ ■ ■ □ □
■ ■ ■ ■ ■ ■ ■ ■     □ □ □ □ ■ ■ ■ □     □ ■ □ □ □ ■ □ □     □ ■ □ □ ■ ■ ■ □
■ ■ ■ ■ ■ ■ ■ ■     □ □ □ □ □ ■ ■ ■     □ □ ■ □ □ □ ■ □     □ □ ■ □ □ ■ ■ ■
■ ■ ■ ■ ■ ■ ■ ■     □ □ □ □ □ □ ■ ■     □ □ □ ■ □ □ □ ■     □ □ □ ■ □ □ ■ ■
```

## Key Findings

1. **Works for generation**: Can generate images, music, long text
2. **Scalable**: Trained on sequences up to 16K
3. **Factorization works**: Two simple patterns combine well
4. **Used in production**: Foundation for GPT-3's sparse attention layers

## Applications

- **Image generation**: Trained on 64×64 images (4096 tokens)
- **Audio generation**: Raw audio samples (up to 16384)
- **Language modeling**: Long documents

## Key Papers

- [Sparse Transformers](https://arxiv.org/abs/1904.10509) (2019) - Original
- [Image GPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) (2020) - Uses sparse attention

## Try It

Run the notebook to:
1. Build different sparse patterns
2. Compare dense vs sparse attention
3. Visualize patterns
4. Measure speedup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb)

