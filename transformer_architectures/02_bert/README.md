<p align="center">
  <img src="https://img.shields.io/badge/Architecture-BERT-34A853?style=for-the-badge" alt="BERT"/>
  <img src="https://img.shields.io/badge/Type-Encoder--Only-lightgrey?style=for-the-badge" alt="Type"/>
  <img src="https://img.shields.io/badge/Direction-Bidirectional-blue?style=for-the-badge" alt="Direction"/>
</p>

<h1 align="center">02. BERT</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../01_vanilla_transformer/README.md">← Prev</a> •
  <a href="../03_gpt/README.md">Next: GPT →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

BERT changed how we think about NLP. Instead of training a model for each task, you pretrain once on tons of text, then fine-tune for anything.

> *The key insight: let the model see **both directions**.*

```
GPT:   "The cat sat on the ___"  → only sees left
BERT:  "The cat sat on the ___"  → sees left AND right
```

---

## 🏗️ Architecture

```
[CLS] Token₁ Token₂ ... Token_N [SEP]
              ↓
Token + Segment + Position Embeddings
              ↓
    Transformer Encoder × 12
              ↓
[CLS] → Classification | Tokens → NER/QA
```

---

## 🎯 Pre-training

<table>
<tr>
<td width="50%" valign="top">

### 🎭 Masked Language Model (MLM)

Mask 15% of tokens, predict them:
- 80% → `[MASK]`
- 10% → random word
- 10% → unchanged

```
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

</td>
<td width="50%" valign="top">

### 🔗 Next Sentence Prediction (NSP)

Predict if sentence B follows A:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
                  ↓
          IsNext / NotNext
```

*(Later dropped in RoBERTa)*

</td>
</tr>
</table>

---

## 📊 Model Sizes

| Model | Layers | Hidden | Heads | Params |
|-------|:------:|:------:|:-----:|:------:|
| BERT-base | 12 | 768 | 12 | **110M** |
| BERT-large | 24 | 1024 | 16 | **340M** |

---

## ✅ Best For

| ✅ Good | ❌ Not For |
|---------|-----------|
| Text classification | Text generation |
| Named Entity Recognition | Translation |
| Question answering | Chatbots |
| Sentiment analysis | |
| Embeddings | |

---

## 👨‍👩‍👧‍👦 The BERT Family

| Model | Year | What's Different |
|-------|:----:|------------------|
| **RoBERTa** | 2019 | More data, no NSP |
| **ALBERT** | 2019 | Parameter sharing |
| **DistilBERT** | 2019 | 40% smaller, 97% performance |
| **DeBERTa** | 2020 | Disentangled attention ⭐ |

> 💡 *Today, I'd use DeBERTa for encoder tasks*

---

## 💻 Code

```python
# BERT: No causal mask - all tokens see all tokens
class BertAttention(nn.Module):
    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        attn = softmax(Q @ K.T / sqrt(d_k))  # No mask!
        return attn @ V

# MLM loss: only on masked positions
def mlm_loss(logits, labels, mask):
    return cross_entropy(logits[mask], labels[mask])
```

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 |
| [RoBERTa](https://arxiv.org/abs/1907.11692) | 2019 |
| [DeBERTa](https://arxiv.org/abs/2006.03654) | 2020 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Build BERT from scratch • MLM pretraining • Visualize bidirectional attention</sub>
</p>
