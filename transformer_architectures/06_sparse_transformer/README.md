# Sparse Transformer

[← Back](../README.md) | [← Prev: Transformer-XL](../05_transformer_xl/README.md) | [Next: Performer →](../07_performer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb)

---

![Architecture](architecture.png)

The O(N²) problem gets bad fast. 4K tokens = 16M attention operations. 16K tokens = 256M. OpenAI's Sparse Transformer asks: do we really need every token to attend to every other token? What if we only compute attention for pairs that matter?

---

## The insight

Most attention weights are close to zero anyway. The model learns to ignore most token pairs. So why compute them at all?

Sparse Transformer uses **patterns** - predefined rules for which tokens can attend to which. Instead of O(N²), you get O(N√N).

---

## The patterns

**1. Local (sliding window)**

Each token attends to nearby tokens:
```
Token 5 attends to: [3, 4, 5, 6, 7]  (window = 5)
```
Good for: syntax, local context

**2. Strided**

Each token attends to every k-th token:
```
Token 8 attends to: [0, 4, 8]  (stride = 4)
```
Good for: long-range dependencies

**3. Combined (factorized)**

Alternate between patterns across layers:
```
Layer 1: Local attention
Layer 2: Strided attention
Layer 3: Local attention
...
```

Information flows locally in some layers, globally in others. After a few layers, any two tokens can communicate through the network.

---

## Visualizing the patterns

```
Dense:              Local:              Strided:            Combined:
■ ■ ■ ■ ■ ■ ■ ■     ■ ■ □ □ □ □ □ □     ■ □ □ □ ■ □ □ □     ■ ■ □ □ ■ □ □ □
■ ■ ■ ■ ■ ■ ■ ■     ■ ■ ■ □ □ □ □ □     □ ■ □ □ □ ■ □ □     ■ ■ ■ □ □ ■ □ □
■ ■ ■ ■ ■ ■ ■ ■     □ ■ ■ ■ □ □ □ □     □ □ ■ □ □ □ ■ □     □ ■ ■ ■ □ □ ■ □
■ ■ ■ ■ ■ ■ ■ ■     □ □ ■ ■ ■ □ □ □     □ □ □ ■ □ □ □ ■     □ □ ■ ■ ■ □ □ ■
...                 ...                 ...                 ...
```

Dense = compute everything. Sparse = compute only where it's 1.

---

## The numbers

| Sequence | Dense O(N²) | Sparse O(N√N) | Speedup |
|----------|-------------|---------------|---------|
| 1K | 1M | 32K | 32× |
| 4K | 16M | 256K | 64× |
| 16K | 256M | 2M | 128× |

The longer the sequence, the more you save.

---

## Code

Creating a sparse mask:

```python
def sparse_mask(seq_len, window_size, stride):
    mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        # Local: nearby tokens
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
        
        # Strided: every stride-th token
        for j in range(0, i + 1, stride):
            mask[i, j] = 1
    
    return mask
```

Using it in attention:

```python
def sparse_attention(Q, K, V, mask):
    scores = Q @ K.T / sqrt(d_k)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    return softmax(scores) @ V
```

---

## Where it's used

- **GPT-3** uses sparse attention in some layers
- **Image generation** (ImageGPT) - 64×64 images = 4096 tokens
- **Audio** - Raw audio needs very long sequences

The paper trained on sequences up to 16K tokens - impossible with dense attention at the time.

---

## The tradeoff

Sparse attention is an approximation. You're betting that the tokens you skip don't matter. For most tasks, this works. But if your task genuinely needs all-pairs attention, you'll lose quality.

Modern approach: Use FlashAttention instead - it's exact attention but fast. Sparse attention was more important before FlashAttention existed.

---

## Papers

- [Sparse Transformers](https://arxiv.org/abs/1904.10509) (2019) - Original
- [ImageGPT](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) (2020) - Uses sparse attention

---

## Try it

The notebook builds different sparse patterns, visualizes them, compares to dense attention, and measures the speedup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb)
