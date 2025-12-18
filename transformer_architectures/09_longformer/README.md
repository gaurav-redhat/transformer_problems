# Longformer

[← Back](../README.md) | [← Prev: Reformer](../08_reformer/README.md) | [Next: Switch Transformer →](../10_switch_transformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb)

---

![Architecture](architecture.png)

Longformer is probably the most practical of the efficient attention methods. The idea is simple: **local attention for most tokens, global attention for special tokens**. No random features, no LSH, no complicated approximations. Just a sensible pattern that works.

---

## The insight

Think about how you read a long document. Most words only matter in context of nearby words. But some tokens - like the title, or the question you're trying to answer - need to see everything.

Longformer does exactly this:
- **Local**: Each token attends to a window of neighbors
- **Global**: A few special tokens attend to everything (and everything attends to them)

---

## Sliding window attention

Each token attends to w tokens around it:

```
Token i attends to: [i-w/2, ..., i, ..., i+w/2]
```

With w=512 and N=4096:
- Standard: 4096² = 16M operations
- Sliding window: 4096 × 512 = 2M operations
- **8× speedup**

And it's still exact - no approximation. You're just computing attention for pairs that matter.

---

## Global attention

Some tokens need full context:
- **[CLS]** for classification - needs to see the whole document
- **Question tokens** for QA - need to find the answer anywhere
- **Section markers** for summarization

For these tokens, attention is bidirectional and global:
```
Global tokens see: everything
Everything sees: global tokens
```

---

## Task-specific globals

| Task | Which tokens are global |
|------|-------------------------|
| Classification | [CLS] |
| Question Answering | Question tokens |
| Summarization | [CLS] + paragraph starts |
| NER | None (all local) |

You choose what's global based on your task. More globals = more compute but better long-range reasoning.

---

## The math

```
A_ij = 1 if:
  - |i - j| ≤ w/2  (within window)
  - OR i is global
  - OR j is global

Complexity: O(N × (w + g))
```

Where g = number of global tokens (usually tiny compared to N).

---

## Code

```python
def sliding_window_mask(seq_len, window_size):
    mask = torch.zeros(seq_len, seq_len)
    half_w = window_size // 2
    
    for i in range(seq_len):
        start = max(0, i - half_w)
        end = min(seq_len, i + half_w + 1)
        mask[i, start:end] = 1
    
    return mask

def add_global(mask, global_indices):
    for idx in global_indices:
        mask[idx, :] = 1  # Global sees all
        mask[:, idx] = 1  # All see global
    return mask
```

---

## Dilated attention

For even larger receptive fields without more compute, use dilation:

```
Standard window (d=1): [3, 4, 5, 6, 7]
Dilated window (d=2):  [0, 2, 4, 6, 8]
```

Lower layers use tight windows (local patterns), higher layers use dilated windows (global patterns).

---

## Model sizes

| Model | Layers | Hidden | Window | Max Length |
|-------|--------|--------|--------|------------|
| Longformer-base | 12 | 768 | 512 | 4096 |
| Longformer-large | 24 | 1024 | 512 | 4096 |

4096 tokens is enough for most documents. For longer, you can extend with gradient checkpointing.

---

## When to use Longformer

**Good for:**
- Long document classification
- Question answering over long context
- Summarization
- Legal, medical, scientific documents

**Not needed for:**
- Short texts (< 512 tokens) - just use BERT
- When you actually need all-pairs attention
- Tasks where local context doesn't help

---

## Longformer vs alternatives

| Method | Complexity | Exact? | Simple? |
|--------|------------|--------|---------|
| Standard | O(N²) | Yes | Yes |
| Sparse | O(N√N) | Patterns only | Medium |
| Performer | O(N) | Approximation | Complex |
| Reformer | O(N log N) | LSH approximation | Complex |
| **Longformer** | **O(N × w)** | **Yes (within pattern)** | **Yes** |
| FlashAttention | O(N²) | Yes | Yes (library) |

Longformer wins on simplicity. You don't need special kernels or approximations - it's just masked attention with a sensible pattern.

---

## Related: BigBird

Google's BigBird (2020) is similar but adds random attention - each token attends to some random tokens as well. In theory this helps with worst-case coverage. In practice, Longformer works just as well and is simpler.

---

## Papers

- [Longformer](https://arxiv.org/abs/2004.05150) (2020) - Original
- [BigBird](https://arxiv.org/abs/2007.14062) (2020) - Google's variant

---

## Try it

The notebook implements sliding window attention, adds global attention, compares to full attention, and trains on a long document task.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb)
