<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Longformer-607D8B?style=for-the-badge" alt="Longformer"/>
  <img src="https://img.shields.io/badge/Complexity-O(N)-green?style=for-the-badge" alt="Complexity"/>
  <img src="https://img.shields.io/badge/Method-Sliding_Window-blue?style=for-the-badge" alt="Method"/>
</p>

<h1 align="center">09. Longformer</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../08_reformer/README.md">← Prev</a> •
  <a href="../10_switch_transformer/README.md">Next: Switch →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

Longformer is probably the most **practical** efficient attention method.

> *Local attention for most tokens + Global attention for special tokens.*

No random features. No LSH. Just a sensible pattern that works.

---

## 🎯 The Insight

Think about how you read:
- Most words only matter in context of **nearby words**
- But some tokens (title, question) need to see **everything**

<table>
<tr>
<td align="center" width="50%">

### 🔲 Local (Sliding Window)
Each token attends to w neighbors

```
Token i → [i-w/2, ..., i, ..., i+w/2]
```

**Complexity: O(N × w)**

</td>
<td align="center" width="50%">

### 🌍 Global
Special tokens see everything

```
[CLS] → [all tokens]
[all tokens] → [CLS]
```

**For: classification, questions**

</td>
</tr>
</table>

---

## 📊 The Speedup

```
N = 4096, w = 512

Standard:  4096² = 16M operations
Longformer: 4096 × 512 = 2M operations

Speedup: 8×
```

And it's **exact** — no approximation!

---

## 🎯 Task-Specific Globals

| Task | Global Tokens |
|------|:-------------:|
| Classification | `[CLS]` |
| Question Answering | Question tokens |
| Summarization | `[CLS]` + paragraph starts |
| NER | None (all local) |

---

## 📐 The Math

```
A_ij = 1 if:
  - |i - j| ≤ w/2       (within window)
  - OR i is global
  - OR j is global

Complexity: O(N × (w + g))
```

Where g = number of global tokens (usually tiny).

---

## 🆚 Longformer vs BERT

| Model | Max Length | Memory (4K) |
|-------|:----------:|:-----------:|
| BERT-base | 512 | OOM ❌ |
| Longformer-base | 4096 | ~3GB ✅ |
| Longformer-large | 4096 | ~8GB ✅ |

---

## 💻 Code

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

## ✅ When to Use

| ✅ Good For | ❌ Not Needed |
|------------|--------------|
| Long document classification | Short texts (< 512) |
| Long-form QA | Tasks needing full attention |
| Summarization | |
| Legal/medical docs | |

---

## 🆚 vs Other Methods

| Method | Complexity | Exact? | Simple? |
|--------|:----------:|:------:|:-------:|
| Standard | O(N²) | ✅ | ✅ |
| Sparse | O(N√N) | ❌ | ⚠️ |
| Performer | O(N) | ❌ | ❌ |
| Reformer | O(N log N) | ❌ | ❌ |
| **Longformer** | **O(N × w)** | **✅** | **✅** |

> 💡 *Longformer wins on simplicity — just masked attention with a sensible pattern.*

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [Longformer](https://arxiv.org/abs/2004.05150) | 2020 |
| [BigBird](https://arxiv.org/abs/2007.14062) | 2020 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Implement sliding window • Add global attention • Compare vs full attention</sub>
</p>
