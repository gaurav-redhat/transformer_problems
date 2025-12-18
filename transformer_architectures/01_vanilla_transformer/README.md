<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Vanilla_Transformer-4285F4?style=for-the-badge" alt="Vanilla"/>
  <img src="https://img.shields.io/badge/Type-Encoder--Decoder-lightgrey?style=for-the-badge" alt="Type"/>
  <img src="https://img.shields.io/badge/Complexity-O(N²)-red?style=for-the-badge" alt="Complexity"/>
</p>

<h1 align="center">01. Vanilla Transformer</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../02_bert/README.md">Next: BERT →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

This is where it all started. The 2017 "Attention Is All You Need" paper that replaced RNNs and LSTMs with pure attention. Every modern language model — BERT, GPT, LLaMA, Claude — builds on this foundation.

> *If you only understand one architecture, make it this one.*

---

## 🏗️ Architecture

```
Input → Embedding → [Encoder × 6] → [Decoder × 6] → Output
```

<table>
<tr>
<td width="50%" valign="top">

### Encoder
1. **Self-Attention** — every token sees every token
2. **Add & Norm** — residual + layer norm
3. **Feed-Forward** — two linear layers
4. **Add & Norm**

</td>
<td width="50%" valign="top">

### Decoder
1. **Masked Self-Attention** — only see past
2. **Add & Norm**
3. **Cross-Attention** — look at encoder
4. **Add & Norm**
5. **Feed-Forward**
6. **Add & Norm**

</td>
</tr>
</table>

---

## ➗ The Math

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

| Symbol | Meaning |
|:------:|---------|
| **Q** | Query — "What am I looking for?" |
| **K** | Key — "What do I contain?" |
| **V** | Value — "What should I return?" |
| **√d_k** | Scaling — prevents softmax saturation |

### Multi-Head Attention

```
MultiHead = Concat(head₁, ..., head_h) × W_O
```

Different heads learn different relationships (syntax, semantics, coreference).

---

## 📊 Numbers

| Parameter | Value |
|-----------|:-----:|
| Model dimension | 512 |
| FFN dimension | 2048 |
| Attention heads | 8 |
| Layers | 6 |
| **Total params** | **~65M** |

---

## ⚠️ The O(N²) Problem

| Sequence | Attention Ops |
|:--------:|:-------------:|
| 1K | 1M |
| 4K | 16M |
| 16K | 256M |

This is why we have Longformer, Performer, etc. — they fix this quadratic scaling.

---

## 💻 Code

```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    return torch.matmul(F.softmax(scores, dim=-1), V)
```

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) | 2018 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Build from scratch • Train on tiny dataset • Visualize attention</sub>
</p>
