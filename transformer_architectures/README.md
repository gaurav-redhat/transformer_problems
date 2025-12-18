# Transformer Architectures

<p align="center">
  <img src="banner.png" alt="Transformer Architectures" width="100%"/>
</p>

I got tired of reading papers that explain attention with walls of math but never show you how it actually works. So I implemented 10 transformer variants from scratch — not production code, just readable PyTorch that trains on tiny datasets in a couple minutes.

The goal: understand *why* each architecture exists, not just memorize the equations.

---

## What's here

Each folder has:
- A diagram showing the architecture
- PyTorch code you can actually read (no 500-line base classes)
- A Colab notebook that trains on a tiny dataset
- Visualizations of what the attention is doing

Click any Colab badge to start training immediately.

---

## The architectures

### The classics — understand these first

| | What | The idea | Run it |
|--|------|----------|--------|
| 01 | [**Vanilla Transformer**](./01_vanilla_transformer/) | The 2017 original. Encoder-decoder, self-attention, the works | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| 02 | [**BERT**](./02_bert/) | Encoder only, sees both directions. The king of NLU tasks | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| 03 | [**GPT**](./03_gpt/) | Decoder only, predicts next token. This is ChatGPT's architecture | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| 04 | [**Vision Transformer**](./04_vision_transformer/) | Treat image patches as tokens. Turns out CNNs aren't necessary | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |

### The efficient ones — when O(N²) hurts

Standard attention compares every token to every other token. That's quadratic. These architectures get around it.

| | What | The trick | Complexity | Run it |
|--|------|-----------|------------|--------|
| 05 | [**Transformer-XL**](./05_transformer_xl/) | Cache hidden states between segments | O(N²)/segment | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| 06 | [**Sparse Transformer**](./06_sparse_transformer/) | Only attend to some tokens (local + strided) | O(N√N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| 07 | [**Performer**](./07_performer/) | Random features to approximate softmax attention | O(N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| 08 | [**Reformer**](./08_reformer/) | LSH to find similar tokens + reversible layers | O(N log N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| 09 | [**Longformer**](./09_longformer/) | Sliding window for local, global tokens for [CLS] | O(N) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |

### Scaling up — more parameters, same compute

| | What | The trick | Run it |
|--|------|-----------|--------|
| 10 | [**Switch Transformer**](./10_switch_transformer/) | Route tokens to different expert FFNs. 1.6T params, same FLOPs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

---

## How they evolved

```
2017: Vanilla Transformer
      │
      └── split into:
          │
          ├── BERT (2018) — encoder only, bidirectional
          │   └── RoBERTa, ALBERT, DeBERTa...
          │
          └── GPT (2018) — decoder only, autoregressive  
              └── GPT-2 → GPT-3 → ChatGPT → GPT-4

2019: "Attention is O(N²), let's fix that"
      │
      ├── Transformer-XL — recurrence across segments
      └── Sparse Transformer — attend to subset

2020: Long sequence boom
      │
      ├── Longformer — sliding window + global
      ├── Performer — random features (FAVOR+)
      ├── Reformer — LSH hashing
      └── ViT — images as patches

2021: Scale everything
      │
      └── Switch Transformer — 1.6T params via MoE

2022-2024: Modern LLMs
      │
      ├── FlashAttention (exact attention, but fast)
      ├── LLaMA, Mistral, Mixtral (open weights)
      ├── Mamba (no attention at all!)
      └── DeepSeek-V3 ($5.5M to train 671B)
```

---

## What I'd actually use in 2025

Honest opinions:

| Task | Just use this |
|------|--------------|
| Text classification | Fine-tune BERT or DeBERTa |
| Chatbot / generation | Mistral 7B or LLaMA 3.1 |
| Long documents | Native long-context models (Gemini, Claude) |
| Images | ViT or DINOv2 |
| Scale without $$$ | Mixtral or DeepSeek |
| Embeddings | E5-mistral or GTE |

The "efficient" architectures (Performer, Reformer) were cool research but FlashAttention basically solved the problem differently — it's exact attention that just runs fast. Most production systems use that now.

---

## Running locally

```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems/transformer_architectures
pip install torch matplotlib numpy
jupyter notebook
```

Or just click the Colab badges. They work.

---

## If you want the papers

**Start with these:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the original
- [BERT](https://arxiv.org/abs/1810.04805) (2018) — bidirectional pretraining
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)

**Efficient attention:**
- [Longformer](https://arxiv.org/abs/2004.05150) (2020) — the practical one
- [FlashAttention](https://arxiv.org/abs/2205.14135) (2022) — the one everyone uses

**Modern stuff:**
- [LLaMA](https://arxiv.org/abs/2302.13971) (2023) — open LLM
- [Mamba](https://arxiv.org/abs/2312.00752) (2023) — no attention, SSM instead

---

## Contributing

Found a bug? Think an explanation is wrong? PRs welcome.

## License

MIT
