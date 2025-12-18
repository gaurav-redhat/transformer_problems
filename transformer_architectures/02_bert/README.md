# BERT

[← Back](../README.md) | [← Prev: Vanilla](../01_vanilla_transformer/README.md) | [Next: GPT →](../03_gpt/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb)

---

![Architecture](architecture.png)

BERT changed how we think about NLP. Instead of training a model for each task, you pretrain once on tons of text, then fine-tune for anything - classification, QA, NER, whatever.

The key insight: let the model see **both directions**. GPT only sees left context. BERT sees everything.

---

## Why it matters

Before BERT (2018), we trained separate models for each task. After BERT, we realized you could:

1. Pretrain on massive unlabeled text (cheap, scales)
2. Fine-tune on your small labeled dataset (expensive, limited)

This is now the default approach for almost everything in NLP.

---

## Bidirectional attention

Here's the difference from GPT:

```
GPT reads:   "The cat sat on the ___"  → only sees left
BERT reads:  "The cat sat on the ___"  → sees left AND right
```

BERT can use "mat." at the end to help fill in "the ___" - GPT can't.

This makes BERT great for understanding tasks but useless for generation (it would cheat by looking ahead).

---

## How it trains

Two pretraining objectives:

**1. Masked Language Modeling (MLM)**

Randomly mask 15% of tokens, predict them:
- 80% replaced with [MASK]
- 10% replaced with random word
- 10% unchanged

```
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

The random/unchanged tokens prevent the model from only paying attention to [MASK].

**2. Next Sentence Prediction (NSP)**

Given two sentences, predict if B follows A. Turns out this doesn't help much - later models (RoBERTa) dropped it.

---

## Architecture

BERT is just the encoder half of the original transformer:

```
[CLS] + Tokens + [SEP]
        ↓
Token + Segment + Position embeddings
        ↓
Transformer Encoder × 12 (or 24)
        ↓
[CLS] → classification
Tokens → token-level tasks
```

No decoder. No causal mask. Every token sees every other token.

---

## Model sizes

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

---

## What BERT is good at

Works great:
- Text classification (spam, sentiment, topic)
- Named Entity Recognition
- Question answering (extractive - find the answer in context)
- Semantic similarity
- Sentence embeddings

Doesn't work:
- Text generation (use GPT)
- Translation (use encoder-decoder)
- Anything where you need to produce text

---

## The BERT family

BERT spawned a bunch of variants:

| Model | What's different |
|-------|------------------|
| RoBERTa | More data, no NSP, dynamic masking |
| ALBERT | Parameter sharing, smaller |
| DistilBERT | Distillation, 40% smaller, 97% performance |
| DeBERTa | Disentangled attention, currently best encoder |

If I were starting a project today, I'd use DeBERTa or a fine-tuned BERT variant from HuggingFace.

---

## Code

BERT attention is vanilla self-attention without a causal mask:

```python
class BertAttention(nn.Module):
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # No mask - all tokens see all tokens
        attn = softmax(Q @ K.T / sqrt(d_k))
        return attn @ V
```

The MLM loss only computes on masked positions:

```python
def mlm_loss(logits, labels, mask):
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    return cross_entropy(masked_logits, masked_labels)
```

---

## Papers

- [BERT](https://arxiv.org/abs/1810.04805) (2018) - Original
- [RoBERTa](https://arxiv.org/abs/1907.11692) (2019) - Better training recipe
- [DeBERTa](https://arxiv.org/abs/2006.03654) (2020) - Current state of the art

---

## Try it

The notebook implements BERT from scratch, shows the MLM pretraining, and visualizes the bidirectional attention.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb)
