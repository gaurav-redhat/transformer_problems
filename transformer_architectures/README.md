# 🤖 Transformer Architectures: A Complete Guide

<p align="center">
  <img src="banner.png" alt="Transformer Architectures" width="100%"/>
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb)

## 📚 Overview

This repository contains implementations, explanations, and training code for **10 different Transformer architectures**. Each implementation includes:

- 📊 **Architecture diagrams** - Visual explanations
- 💻 **PyTorch implementations** - Clean, documented code
- 🎯 **Training notebooks** - Train on tiny datasets in Google Colab
- 📖 **Detailed explanations** - How and why each architecture works

---

## 🗺️ Transformer Family Tree

```
                        ┌─────────────────────────────────────────┐
                        │     Original Transformer (2017)          │
                        │     "Attention Is All You Need"          │
                        └─────────────────┬───────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │  ENCODER-ONLY   │         │ ENCODER-DECODER │         │  DECODER-ONLY   │
    │     (BERT)      │         │      (T5)       │         │     (GPT)       │
    └────────┬────────┘         └─────────────────┘         └────────┬────────┘
             │                                                        │
    ┌────────┴────────┐                                     ┌────────┴────────┐
    │  Vision (ViT)   │                                     │  GPT-2, GPT-3   │
    │  RoBERTa, ALBERT│                                     │  LLaMA, Mistral │
    └─────────────────┘                                     └─────────────────┘

                        ┌─────────────────────────────────────────┐
                        │         EFFICIENT TRANSFORMERS           │
                        └─────────────────┬───────────────────────┘
                                          │
    ┌─────────────┬─────────────┬─────────┴─────────┬─────────────┬─────────────┐
    │             │             │                   │             │             │
    ▼             ▼             ▼                   ▼             ▼             ▼
┌───────┐   ┌─────────┐   ┌──────────┐       ┌──────────┐   ┌─────────┐   ┌───────┐
│Sparse │   │Performer│   │Longformer│       │ Reformer │   │Trans-XL │   │Switch │
│  O(N) │   │  O(N)   │   │  O(N)    │       │O(NlogN)  │   │ O(N)    │   │ MoE   │
└───────┘   └─────────┘   └──────────┘       └──────────┘   └─────────┘   └───────┘
```

---

## 📁 Architectures

| # | Architecture | Type | Attention | Key Innovation | Colab |
|---|--------------|------|-----------|----------------|-------|
| 01 | [Vanilla Transformer](./01_vanilla_transformer/) | Encoder-Decoder | O(N²) | Self-attention mechanism | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| 02 | [BERT](./02_bert/) | Encoder | O(N²) | Bidirectional, MLM pretraining | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| 03 | [GPT](./03_gpt/) | Decoder | O(N²) | Autoregressive, causal mask | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| 04 | [Vision Transformer](./04_vision_transformer/) | Encoder | O(N²) | Patches as tokens for images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |
| 05 | [Transformer-XL](./05_transformer_xl/) | Decoder | O(N²) | Segment-level recurrence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| 06 | [Sparse Transformer](./06_sparse_transformer/) | Any | O(N√N) | Sparse attention patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| 07 | [Performer](./07_performer/) | Any | O(N) | FAVOR+ random features | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| 08 | [Reformer](./08_reformer/) | Any | O(N log N) | LSH attention, reversible | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| 09 | [Longformer](./09_longformer/) | Encoder | O(N) | Local + global attention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |
| 10 | [Switch Transformer](./10_switch_transformer/) | Any | O(N²) | Mixture of Experts (MoE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

---

## 🔑 Key Concepts

### Attention Complexity Comparison

| Architecture | Time | Memory | Max Context | Best For |
|--------------|------|--------|-------------|----------|
| Vanilla | O(N²) | O(N²) | 512-2K | General |
| BERT/GPT | O(N²) | O(N²) | 512-4K | NLU/NLG |
| Transformer-XL | O(N²) | O(N) | 4K+ | Long sequences |
| Sparse | O(N√N) | O(N√N) | 8K+ | Structured data |
| Performer | O(N) | O(N) | 64K+ | Very long |
| Longformer | O(N) | O(N) | 16K | Documents |
| Reformer | O(N log N) | O(N) | 64K | Memory-limited |

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)
Click any Colab badge above to start training immediately!

### Local Setup
```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems/transformer_architectures
pip install torch matplotlib numpy
```

---

## 📊 What Each Notebook Contains

1. **Architecture Diagram** - Visual representation
2. **Math Explanation** - Key equations
3. **PyTorch Implementation** - From scratch
4. **Training Loop** - On tiny dataset
5. **Visualization** - Attention patterns, loss curves
6. **Comparison** - vs other architectures

---

## 🎯 Learning Path

**Beginner:**
1. Start with [Vanilla Transformer](./01_vanilla_transformer/) - understand the basics
2. Move to [GPT](./03_gpt/) - see autoregressive generation
3. Try [BERT](./02_bert/) - understand bidirectional

**Intermediate:**
4. [Vision Transformer](./04_vision_transformer/) - apply to images
5. [Transformer-XL](./05_transformer_xl/) - handle longer sequences

**Advanced:**
6. [Sparse Transformer](./06_sparse_transformer/) - efficient attention
7. [Performer](./07_performer/) - linear complexity
8. [Longformer](./09_longformer/) - document understanding
9. [Reformer](./08_reformer/) - memory efficiency
10. [Switch Transformer](./10_switch_transformer/) - scaling with MoE

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [BERT](https://arxiv.org/abs/1810.04805) - Bidirectional Encoder
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Language Models
- [ViT](https://arxiv.org/abs/2010.11929) - Vision Transformer
- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Long Context
- [Sparse Transformers](https://arxiv.org/abs/1904.10509) - OpenAI
- [Performer](https://arxiv.org/abs/2009.14794) - FAVOR+
- [Reformer](https://arxiv.org/abs/2001.04451) - Efficient Transformer
- [Longformer](https://arxiv.org/abs/2004.05150) - Long Document
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - MoE

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📄 License

MIT License

