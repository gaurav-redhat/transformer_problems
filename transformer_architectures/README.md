# Transformer Architectures

<p align="center">
  <img src="banner.png" alt="Transformer Architectures" width="100%"/>
</p>

Ever tried to understand how BERT differs from GPT? Or why Longformer can handle 16K tokens while vanilla attention dies at 4K?

This is my attempt to implement 10 transformer variants from scratch, train them on tiny datasets, and actually *see* how they work.

---

## What's Here

Each folder has:
- A **diagram** showing the architecture
- **PyTorch code** you can actually read (no 500-line base classes)
- A **Colab notebook** that trains on a tiny dataset in ~2 minutes
- Visualizations of attention patterns

The goal: understand *why* each architecture exists, not just *what* it does.

---

## The Architectures

### The Classics

| | Architecture | What It Does | Try It |
|--|--------------|--------------|--------|
| 01 | [Vanilla Transformer](./01_vanilla_transformer/) | The OG. Encoder-decoder with self-attention. Still the foundation of everything. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| 02 | [BERT](./02_bert/) | Encoder-only. Looks at text bidirectionally. Great for understanding, bad for generating. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| 03 | [GPT](./03_gpt/) | Decoder-only. Predicts next token. The architecture behind ChatGPT, Claude, etc. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| 04 | [Vision Transformer](./04_vision_transformer/) | Treats image patches like words. Surprisingly simple and it works. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |

### The Long-Context Ones

Standard attention is O(N²). Double your sequence, quadruple your compute. These architectures fix that:

| | Architecture | The Trick | Complexity | Try It |
|--|--------------|-----------|------------|--------|
| 05 | [Transformer-XL](./05_transformer_xl/) | Cache hidden states across segments | O(N²) per segment | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| 06 | [Sparse Transformer](./06_sparse_transformer/) | Only attend to some tokens (local + strided) | O(N√N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| 07 | [Performer](./07_performer/) | Approximate softmax with random features | O(N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| 08 | [Reformer](./08_reformer/) | LSH to find similar queries, reversible layers | O(N log N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| 09 | [Longformer](./09_longformer/) | Sliding window + global tokens for [CLS] | O(N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |

### The Scaling One

| | Architecture | The Trick | Try It |
|--|--------------|-----------|--------|
| 10 | [Switch Transformer](./10_switch_transformer/) | MoE: route each token to 1 of N experts. Trillion params, same FLOPs. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

---

## How They Evolved

```
2017: Vanilla Transformer ("Attention Is All You Need")
      │
      ├── Encoder-only → BERT (2018) → RoBERTa, ALBERT, DistilBERT
      │                      └── Vision Transformer (2020)
      │
      ├── Decoder-only → GPT (2018) → GPT-2 → GPT-3 → ChatGPT
      │                                         └── LLaMA, Mistral, Claude
      │
      └── "Attention is O(N²), let's fix that"
              │
              ├── Transformer-XL (2019) - recurrence across segments
              ├── Sparse Transformer (2019) - attend to subset
              ├── Longformer (2020) - local + global
              ├── Reformer (2020) - LSH attention  
              ├── Performer (2020) - random features
              └── Switch Transformer (2021) - MoE
```

---

## Quick Comparison

When to use what:

| Need | Use | Why |
|------|-----|-----|
| Text classification | BERT | Bidirectional context |
| Text generation | GPT | Autoregressive, battle-tested |
| Image classification | ViT | If you have enough data |
| Long documents (4K-16K) | Longformer | Simple, works well |
| Very long (64K+) | Performer or Reformer | Linear complexity |
| Massive scale | Switch Transformer | More params, same compute |

---

## Get Started

Easiest way: click a Colab badge above and run. Everything installs in the notebook.

Or locally:

```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems/transformer_architectures
pip install torch matplotlib numpy

# Then open any notebook
jupyter notebook 01_vanilla_transformer/demo.ipynb
```

---

## Where to Start

**New to transformers?**
1. [Vanilla Transformer](./01_vanilla_transformer/) - the foundation
2. [GPT](./03_gpt/) - see autoregressive generation in action
3. [BERT](./02_bert/) - understand bidirectional attention

**Want to handle long sequences?**
- Start with [Longformer](./09_longformer/) - most practical
- Then [Sparse Transformer](./06_sparse_transformer/) - understand the patterns
- [Performer](./07_performer/) if you need true O(N)

**Building something big?**
- [Switch Transformer](./10_switch_transformer/) - MoE is the future

---

## Papers

The actual papers if you want to go deeper:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) - Start here
- [BERT](https://arxiv.org/abs/1810.04805) (2018)
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- [ViT](https://arxiv.org/abs/2010.11929) (2020)
- [Transformer-XL](https://arxiv.org/abs/1901.02860) (2019)
- [Sparse Transformers](https://arxiv.org/abs/1904.10509) (2019)
- [Performer](https://arxiv.org/abs/2009.14794) (2020)
- [Reformer](https://arxiv.org/abs/2001.04451) (2020)
- [Longformer](https://arxiv.org/abs/2004.05150) (2020)
- [Switch Transformer](https://arxiv.org/abs/2101.03961) (2021)

---

## Contributing

Found a bug? Have a cleaner implementation? PRs welcome.

## License

MIT - do whatever you want with it.
