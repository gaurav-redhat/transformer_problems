<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Performer-00BCD4?style=for-the-badge" alt="Performer"/>
  <img src="https://img.shields.io/badge/Complexity-O(N)-green?style=for-the-badge" alt="Complexity"/>
  <img src="https://img.shields.io/badge/Method-FAVOR+-orange?style=for-the-badge" alt="Method"/>
</p>

<h1 align="center">07. Performer</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../06_sparse_transformer/README.md">← Prev</a> •
  <a href="../08_reformer/README.md">Next: Reformer →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

Most efficient attention methods skip computations or use heuristics. Performer does something clever:

> *Rewrite the attention equation so you can compute it in a different order — O(N) instead of O(N²).*

---

## 🔑 The Key Insight

**Standard attention:**
```
Attention = softmax(QK^T) × V
```
↳ Must compute QK^T first → N×N matrix 🚫

**What if we could decompose softmax?**
```
softmax(q·k) ≈ φ(q) · φ(k)
```

**Then:**
```
Attention ≈ φ(Q) × (φ(K)^T × V)
```
↳ Compute K^T × V first → d×d matrix ✅

---

## 📊 The Numbers

| Operation | Standard | Performer |
|-----------|:--------:|:---------:|
| QK^T | O(N² × d) | — |
| φ(K)^T × V | — | O(N × d × m) |
| **Total** | **O(N² × d)** | **O(N × d × m)** |

### Example

```
N = 16,384  d = 64  m = 64

Standard:  16,384² × 64 = 17B operations
Performer: 16,384 × 64² = 67M operations

Speedup: ~256×
```

---

## 🎲 Random Feature Map (FAVOR+)

```python
def random_feature_map(x, omega):
    # x: (B, N, d), omega: (d, m) random projections
    projection = x @ omega  # (B, N, m)
    norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2
    features = torch.exp(projection - norm_sq) / sqrt(m)
    return features
```

**FAVOR+** = Fast Attention Via positive Orthogonal Random features

---

## 💻 Code

```python
def favor_attention(Q, K, V, omega):
    Q_prime = random_feature_map(Q, omega)  # (B, N, m)
    K_prime = random_feature_map(K, omega)  # (B, N, m)
    
    # THE TRICK: compute K'V first!
    KV = K_prime.transpose(-2, -1) @ V       # (B, m, d)
    K_sum = K_prime.sum(dim=1, keepdim=True).T
    
    numerator = Q_prime @ KV                  # (B, N, d)
    denominator = Q_prime @ K_sum
    
    return numerator / (denominator + 1e-6)
```

---

## ⚖️ The Tradeoff

| Aspect | Performer | Standard |
|--------|:---------:|:--------:|
| Complexity | O(N) | O(N²) |
| Exact? | ❌ Approximation | ✅ Exact |
| Accuracy | Depends on m | Perfect |
| Memory | Low | High |

> 💡 *More random features (larger m) = better approximation but more compute*

---

## 🆚 vs Other Methods

| Method | Complexity | Exact? |
|--------|:----------:|:------:|
| Standard | O(N²) | ✅ |
| Sparse | O(N√N) | ❌ (misses pairs) |
| **Performer** | **O(N)** | ❌ (approx) |
| FlashAttention | O(N²) | ✅ |

> ⚠️ *FlashAttention is now preferred — exact attention that's also fast.*

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [Performer](https://arxiv.org/abs/2009.14794) | 2020 |
| [Random Features for Kernels](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) | 2007 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Implement FAVOR+ • Compare approximation quality • Measure speedup</sub>
</p>
