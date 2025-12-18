# Vanilla Transformer

[← Back](../README.md) | [Next: BERT →](../02_bert/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb)

---

![Architecture](architecture.png)

This is where it all started. The 2017 "Attention Is All You Need" paper that replaced RNNs and LSTMs with pure attention. Every modern language model - BERT, GPT, LLaMA, Claude - is built on this foundation.

If you only understand one architecture, make it this one.

---

## How it works

The transformer has two parts: an **encoder** (reads the input) and a **decoder** (generates the output). The original was built for translation - encoder reads "Hello", decoder outputs "Bonjour".

```
Input → Embedding → [Encoder × 6] → [Decoder × 6] → Output
```

Each encoder layer does:
1. Self-attention (every token looks at every other token)
2. Feed-forward network (process each position)
3. Residual connections + layer norm (keep gradients flowing)

The decoder is similar, but with two key differences:
- **Masked self-attention**: Can only see past tokens (no peeking at the future)
- **Cross-attention**: Looks at the encoder output

---

## The attention mechanism

This is the heart of the transformer. The idea: let each token decide what to pay attention to.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

In plain English:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What should I return?"

The √d_k is just to keep the softmax from saturating. Without it, large dot products make everything either 0 or 1.

**Multi-head attention** runs several attention operations in parallel - different heads learn different things (one might track syntax, another semantics, another coreference).

---

## Positional encoding

Attention has no sense of order. "Dog bites man" and "Man bites dog" look identical to it. We fix this by adding position information:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

The sin/cos pattern lets the model learn relative positions - position 5 relative to position 3 always has the same pattern.

---

## The numbers

Original paper configuration:

| What | Value |
|------|-------|
| Model dimension | 512 |
| Feed-forward dimension | 2048 |
| Attention heads | 8 |
| Layers | 6 |
| Total parameters | ~65M |

---

## Why O(N²) matters

The QK^T multiplication creates an N×N attention matrix. Double your sequence length, quadruple your compute. This is why we have Longformer, Performer, and all the efficient variants - they're trying to fix this.

| Sequence | Attention ops |
|----------|---------------|
| 1K tokens | 1M |
| 4K tokens | 16M |
| 16K tokens | 256M |

---

## Code

The core attention is surprisingly simple:

```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    return torch.matmul(F.softmax(scores, dim=-1), V)
```

---

## Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) - The original
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Excellent walkthrough

---

## Try it

The notebook builds a transformer from scratch, trains it on a tiny translation task, and visualizes what the attention heads are looking at.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb)
