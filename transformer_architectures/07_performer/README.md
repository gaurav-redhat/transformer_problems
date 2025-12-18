# Performer

[← Back](../README.md) | [← Prev: Sparse Transformer](../06_sparse_transformer/README.md) | [Next: Reformer →](../08_reformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb)

---

![Architecture](architecture.png)

Most "efficient attention" methods skip computations (sparse) or make approximations (Reformer's LSH). Performer does something cleverer: it rewrites the attention equation so that you can compute it in a different order - and that order happens to be O(N) instead of O(N²).

---

## The key insight

Standard attention:
```
Attention = softmax(QK^T) × V
```

The problem is QK^T - that's an N×N matrix. You have to compute it before multiplying by V.

But what if we could decompose softmax as:
```
softmax(q·k) ≈ φ(q) · φ(k)
```

Then:
```
Attention ≈ φ(Q) × (φ(K)^T × V)
```

Now you can compute **K^T × V first** (that's a d×d matrix, much smaller than N×N), then multiply by φ(Q).

The order of operations changes from O(N²) to O(N). That's the whole trick.

---

## How the decomposition works

Performer uses **random features** to approximate the softmax kernel. The math is deep, but the intuition:

1. Sample random vectors ω₁, ω₂, ..., ωₘ
2. Define φ(x) = exp(-||x||²/2) × [exp(ω₁·x), exp(ω₂·x), ...]
3. Then φ(q)·φ(k) ≈ exp(q·k)

This is called **FAVOR+** (Fast Attention Via positive Orthogonal Random features).

---

## The complexity breakdown

| Operation | Standard | Performer |
|-----------|----------|-----------|
| QK^T | O(N² × d) | — |
| φ(K)^T × V | — | O(N × d × m) |
| φ(Q) × (K^TV) | — | O(N × d × m) |
| **Total** | **O(N² × d)** | **O(N × d × m)** |

When N >> d and m ≈ d: **huge savings**.

Example with N=16384, d=64, m=64:
- Standard: 17 billion operations
- Performer: 67 million operations
- **256× speedup**

---

## Code

The random feature map:

```python
def random_feature_map(x, omega):
    # x: (B, N, d), omega: (d, m) random projections
    projection = x @ omega  # (B, N, m)
    norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2
    features = torch.exp(projection - norm_sq) / math.sqrt(omega.size(1))
    return features
```

FAVOR+ attention:

```python
def favor_attention(Q, K, V, omega):
    Q_prime = random_feature_map(Q, omega)  # (B, N, m)
    K_prime = random_feature_map(K, omega)  # (B, N, m)
    
    # THE TRICK: compute K'V first
    KV = K_prime.transpose(-2, -1) @ V       # (B, m, d)
    K_sum = K_prime.sum(dim=1, keepdim=True).T  # (B, m, 1)
    
    numerator = Q_prime @ KV                  # (B, N, d)
    denominator = Q_prime @ K_sum             # (B, N, 1)
    
    return numerator / (denominator + 1e-6)
```

---

## The catch

It's an **approximation**. The more random features (larger m), the better the approximation, but the more compute. You're trading accuracy for speed.

In practice, m ≈ d works well for most tasks. But if you need exact attention, Performer isn't it.

---

## Causal (GPT-style) Performer

For autoregressive models, you need causal attention. Performer handles this with **prefix sums**:

```python
def causal_favor(Q_prime, K_prime, V):
    outputs = []
    KV_cumsum = 0
    K_cumsum = 0
    
    for t in range(N):
        KV_cumsum += K_prime[:, t:t+1].T @ V[:, t:t+1]
        K_cumsum += K_prime[:, t:t+1].T
        
        out_t = Q_prime[:, t:t+1] @ KV_cumsum / (Q_prime[:, t:t+1] @ K_cumsum + eps)
        outputs.append(out_t)
    
    return torch.cat(outputs, dim=1)
```

You maintain running sums as you go. Still O(N).

---

## Performer vs other methods

| Method | Complexity | Exact? | Practical? |
|--------|------------|--------|------------|
| Standard | O(N²) | Yes | For short sequences |
| Sparse | O(N√N) | No (misses pairs) | Yes, widely used |
| Performer | O(N) | No (approximation) | Sometimes |
| FlashAttention | O(N²) | Yes | Yes, the current standard |

Performer was exciting when it came out. Then FlashAttention showed you can make exact attention fast through better memory management. In practice, most people now use FlashAttention instead.

---

## Papers

- [Performer](https://arxiv.org/abs/2009.14794) (2020) - Original
- [Random Features for Kernels](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) - The theory behind it

---

## Try it

The notebook implements FAVOR+ from scratch, compares approximation quality vs exact attention, and measures speedup on long sequences.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb)
