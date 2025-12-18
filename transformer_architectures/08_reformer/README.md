<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Reformer-795548?style=for-the-badge" alt="Reformer"/>
  <img src="https://img.shields.io/badge/Complexity-O(N_log_N)-green?style=for-the-badge" alt="Complexity"/>
  <img src="https://img.shields.io/badge/Method-LSH_+_Reversible-orange?style=for-the-badge" alt="Method"/>
</p>

<h1 align="center">08. Reformer</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../07_performer/README.md">← Prev</a> •
  <a href="../09_longformer/README.md">Next: Longformer →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

Reformer tackles **two problems at once**:

| Problem | Solution |
|---------|----------|
| O(N²) attention | **LSH Attention** — hash similar tokens together |
| Memory for backprop | **Reversible Layers** — don't store activations |

> *Together, they let you process **64K tokens** on a single GPU.*

---

## 🔍 LSH Attention

### The Insight
> *Similar queries probably have similar keys. Hash them into buckets, attend within buckets.*

### LSH (Locality-Sensitive Hashing)
```
P(hash(x) = hash(y)) ∝ similarity(x, y)
```

### The Algorithm
```
1. Hash all queries and keys
2. Sort by bucket
3. Attend ONLY within buckets
4. Multiple rounds for coverage
```

### Complexity
```
O(N × bucket_size) ≈ O(N log N)
```

---

## 🔄 Reversible Layers

### The Problem
```
Standard: Store activations at EVERY layer
Memory = O(N × L × d)  where L = layers
```

### The Solution
Split into two streams (x₁, x₂):

```
Forward:
y₁ = x₁ + Attention(x₂)
y₂ = x₂ + FFN(y₁)

Backward (reconstruct!):
x₂ = y₂ - FFN(y₁)
x₁ = y₁ - Attention(x₂)
```

### Memory Savings

| Method | Activation Memory |
|--------|:-----------------:|
| Standard | O(N × L × d) |
| Reversible | O(N × d) |

> *Save a factor of L (number of layers)!*

---

## 💻 Code

### LSH Hashing
```python
def lsh_hash(x, n_hashes, n_buckets):
    d = x.shape[-1]
    projections = torch.randn(n_hashes, d, n_buckets // 2)
    dots = torch.einsum('...d,hdb->...hb', x, projections)
    buckets = (dots > 0).int()
    powers = 2 ** torch.arange(n_buckets // 2)
    return (buckets * powers).sum(dim=-1)
```

### Reversible Block
```python
class ReversibleBlock(nn.Module):
    def __init__(self, attn, ffn):
        self.attn, self.ffn = attn, ffn
    
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

## 📊 What Reformer Saves

| Technique | What It Saves | How |
|-----------|:-------------:|-----|
| LSH Attention | Compute | Only attend within buckets |
| Reversible | Memory | Recompute on backward |
| Chunked FFN | Memory | Process in chunks |
| Shared Q/K | Parameters | Q = K for LSH |

---

## ⚠️ The Tradeoffs

| Pro | Con |
|-----|-----|
| O(N log N) attention | May miss important keys |
| Huge memory savings | Extra compute on backward |
| 64K tokens on 1 GPU | Complex implementation |

---

## 🤔 In Practice

Reformer was important research, but it's complex. Today:

| For Long Sequences | Use |
|-------------------|-----|
| Documents | FlashAttention or Longformer |
| Memory | Gradient checkpointing |

Still, understanding Reformer teaches you about efficient attention design.

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [Reformer](https://arxiv.org/abs/2001.04451) | 2020 |
| [RevNets](https://arxiv.org/abs/1707.04585) | 2017 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Implement LSH attention • Build reversible layers • Compare memory usage</sub>
</p>
