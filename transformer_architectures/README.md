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

## How They Evolved (2017-2025)

```
2017: Vanilla Transformer ("Attention Is All You Need")
      │
      ├── Encoder-only ─────────────────────────────────────────────────────────
      │   └── BERT (2018) → RoBERTa → ALBERT → DeBERTa (2020)
      │         └── Vision Transformer (2020) → DeiT → Swin → ViT-22B
      │
      ├── Decoder-only (The LLM Era) ───────────────────────────────────────────
      │   └── GPT (2018) → GPT-2 (2019) → GPT-3 (2020)
      │         │
      │         ├── 2022: ChatGPT, InstructGPT
      │         ├── 2023: GPT-4, LLaMA, LLaMA-2, Mistral-7B, Phi-1.5
      │         ├── 2024: LLaMA-3, Mistral Large, Mixtral 8x22B, Phi-3, Qwen-2
      │         └── 2025: DeepSeek-V3, LLaMA-3.3, Qwen-2.5
      │
      ├── Efficient Attention ──────────────────────────────────────────────────
      │   ├── Sparse/Longformer (2020) - sparse patterns
      │   ├── FlashAttention (2022) - IO-aware exact attention
      │   ├── FlashAttention-2 (2023) - 2x faster
      │   ├── Ring Attention (2023) - distributed long context
      │   └── FlashAttention-3 (2024) - Hopper GPUs
      │
      ├── MoE (Mixture of Experts) ─────────────────────────────────────────────
      │   ├── Switch Transformer (2021) - top-1 routing
      │   ├── Mixtral 8x7B (2023) - open MoE that works
      │   ├── Mixtral 8x22B (2024)
      │   └── DeepSeek-MoE (2024) - fine-grained experts
      │
      └── Post-Transformer / Alternatives ──────────────────────────────────────
          ├── RWKV (2023) - linear attention RNN
          ├── Mamba (2023) - State Space Models
          ├── RetNet (2023) - retention mechanism
          ├── Mamba-2 (2024) - structured state space duality
          └── Jamba (2024) - Mamba + Transformer hybrid
```

---

## Quick Comparison

What to use in 2025:

| Need | Use | Why |
|------|-----|-----|
| Text classification | BERT / DeBERTa | Still great for NLU tasks |
| Text generation (small) | Phi-3, Mistral 7B | Best quality per parameter |
| Text generation (large) | LLaMA-3.1, Qwen-2.5 | Open, good performance |
| Coding | DeepSeek-Coder, Qwen-Coder | Specialized training |
| Long context (128K+) | LLaMA-3.1, Qwen-2.5 | Native long context + RoPE |
| Very long (1M+) | Ring Attention + any model | Distributed context |
| Images | ViT, Swin, DINOv2 | Depends on task |
| Multimodal | LLaVA, Qwen-VL | Vision + Language |
| MoE (efficiency) | Mixtral, DeepSeek-V3 | More params, same FLOPs |
| Non-attention | Mamba, RWKV | Linear complexity, good for long sequences |

### Model Size Reality Check (2025)

| Model | Size | Context | Training Cost | Open? |
|-------|------|---------|---------------|-------|
| Phi-3-mini | 3.8B | 128K | ~$1M | Yes |
| Mistral 7B | 7B | 32K | ~$2M | Yes |
| LLaMA-3.1-8B | 8B | 128K | Part of $100M+ | Yes |
| Mixtral 8x7B | 46B (12B active) | 32K | ~$10M | Yes |
| LLaMA-3.1-70B | 70B | 128K | Part of $100M+ | Yes |
| DeepSeek-V3 | 671B (37B active) | 128K | $5.5M | Yes |
| LLaMA-3.1-405B | 405B | 128K | Part of $100M+ | Yes |
| GPT-4 | ~1.8T (rumored MoE) | 128K | ~$100M | No |

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
- Start with [Longformer](./09_longformer/) - understand the concept
- In production: use models with native long context (LLaMA-3.1, Qwen-2.5)
- For 1M+ tokens: look into Ring Attention or Mamba

**Building something big?**
- [Switch Transformer](./10_switch_transformer/) - understand MoE
- In production: Mixtral 8x7B or DeepSeek-V3 for efficiency

**Want to stay current?**
- Follow: [@kaborka](https://x.com/kaborka), [@_akhaliq](https://x.com/_akhaliq), [@ylaboratory](https://x.com/ylaboratory)
- Read: [The Transformer Family V2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) by Lilian Weng

---

## Papers & Code (2017-2025)

### The Foundations (2017-2020)

| Year | Paper | What It Did | Code |
|------|-------|-------------|------|
| 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Started everything | [tensor2tensor](https://github.com/tensorflow/tensor2tensor) |
| 2018 | [BERT](https://arxiv.org/abs/1810.04805) | Bidirectional pretraining | [google-research/bert](https://github.com/google-research/bert) |
| 2018 | [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Decoder-only pretraining | [openai/finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm) |
| 2019 | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Scaled up, emergent abilities | [openai/gpt-2](https://github.com/openai/gpt-2) |
| 2019 | [Transformer-XL](https://arxiv.org/abs/1901.02860) | Segment-level recurrence | [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl) |
| 2019 | [Sparse Transformers](https://arxiv.org/abs/1904.10509) | Sparse attention patterns | [openai/sparse_attention](https://github.com/openai/sparse_attention) |
| 2020 | [Reformer](https://arxiv.org/abs/2001.04451) | LSH attention | [google/trax](https://github.com/google/trax) |
| 2020 | [Longformer](https://arxiv.org/abs/2004.05150) | Sliding window + global | [allenai/longformer](https://github.com/allenai/longformer) |
| 2020 | [Performer](https://arxiv.org/abs/2009.14794) | FAVOR+ linear attention | [google-research/performer](https://github.com/google-research/google-research/tree/master/performer) |
| 2020 | [GPT-3](https://arxiv.org/abs/2005.14165) | 175B params, few-shot learning | Closed |
| 2020 | [ViT](https://arxiv.org/abs/2010.11929) | Images as patches | [google-research/vision_transformer](https://github.com/google-research/vision_transformer) |

### The LLM Era (2021-2022)

| Year | Paper | What It Did | Code |
|------|-------|-------------|------|
| 2021 | [Switch Transformer](https://arxiv.org/abs/2101.03961) | MoE with top-1 routing | [google/flaxformer](https://github.com/google/flaxformer) |
| 2021 | [RoPE](https://arxiv.org/abs/2104.09864) | Rotary position embeddings | Used in LLaMA, Mistral |
| 2022 | [InstructGPT](https://arxiv.org/abs/2203.02155) | RLHF for alignment | Closed |
| 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware exact attention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| 2022 | [Chinchilla](https://arxiv.org/abs/2203.15556) | Compute-optimal scaling | Closed |
| 2022 | [PaLM](https://arxiv.org/abs/2204.02311) | 540B params, pathways | Closed |

### Open Source Revolution (2023)

| Year | Paper | What It Did | Code |
|------|-------|-------------|------|
| 2023 | [LLaMA](https://arxiv.org/abs/2302.13971) | Open 7B-65B models | [facebookresearch/llama](https://github.com/facebookresearch/llama) |
| 2023 | [LLaMA-2](https://arxiv.org/abs/2307.09288) | Improved + chat versions | [meta-llama/llama](https://github.com/meta-llama/llama) |
| 2023 | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 2x faster, better parallelism | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| 2023 | [Mistral 7B](https://arxiv.org/abs/2310.06825) | Best 7B model, GQA + sliding window | [mistralai/mistral-src](https://github.com/mistralai/mistral-src) |
| 2023 | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | Open MoE, beats LLaMA-2 70B | [mistralai/mistral-src](https://github.com/mistralai/mistral-src) |
| 2023 | [Phi-1.5](https://arxiv.org/abs/2309.05463) | "Textbooks are all you need" | [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) |
| 2023 | [Ring Attention](https://arxiv.org/abs/2310.01889) | Distributed million-token context | [lhao499/RingAttention](https://github.com/lhao499/RingAttention) |
| 2023 | [Mamba](https://arxiv.org/abs/2312.00752) | State Space Models, no attention | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| 2023 | [RWKV](https://arxiv.org/abs/2305.13048) | Linear attention RNN | [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) |
| 2023 | [RetNet](https://arxiv.org/abs/2307.08621) | Retention mechanism | [microsoft/torchscale](https://github.com/microsoft/torchscale) |

### Scaling & Efficiency (2024)

| Year | Paper | What It Did | Code |
|------|-------|-------------|------|
| 2024 | [LLaMA-3](https://ai.meta.com/blog/meta-llama-3/) | 8B/70B, 15T tokens | [meta-llama/llama3](https://github.com/meta-llama/llama3) |
| 2024 | [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/) | Larger MoE | [HuggingFace](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) |
| 2024 | [Phi-3](https://arxiv.org/abs/2404.14219) | 3.8B beats Mixtral 8x7B | [microsoft/Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| 2024 | [Qwen-2](https://arxiv.org/abs/2407.10671) | Alibaba's best open model | [QwenLM/Qwen2](https://github.com/QwenLM/Qwen2) |
| 2024 | [Mamba-2](https://arxiv.org/abs/2405.21060) | Structured state space duality | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| 2024 | [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Hopper GPU optimizations | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| 2024 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | MLA + DeepSeekMoE | [deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) |
| 2024 | [Jamba](https://arxiv.org/abs/2403.19887) | Mamba + Transformer hybrid | [ai21labs/Jamba](https://huggingface.co/ai21labs/Jamba-v0.1) |

### Latest (2024-2025)

| Year | Paper | What It Did | Code |
|------|-------|-------------|------|
| 2024 | [LLaMA-3.1](https://ai.meta.com/blog/meta-llama-3-1/) | 405B, 128K context | [meta-llama/llama-models](https://github.com/meta-llama/llama-models) |
| 2024 | [Qwen-2.5](https://qwenlm.github.io/blog/qwen2.5/) | 72B rivals GPT-4 | [QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5) |
| 2024 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B MoE, $5.5M training | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) |
| 2024 | [LLaMA-3.3](https://ai.meta.com/blog/llama-3-3/) | 70B matches 405B quality | [meta-llama/llama-models](https://github.com/meta-llama/llama-models) |

### Key Techniques to Know

| Technique | Paper | Why It Matters |
|-----------|-------|----------------|
| RoPE | [Su et al. 2021](https://arxiv.org/abs/2104.09864) | Position encoding that extrapolates to longer sequences |
| GQA | [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) | Grouped-query attention: balance between MHA and MQA |
| SwiGLU | [Shazeer 2020](https://arxiv.org/abs/2002.05202) | Better FFN activation (used in LLaMA) |
| ALiBi | [Press et al. 2022](https://arxiv.org/abs/2108.12409) | No position embeddings, just attention bias |
| KV Cache | - | Caching keys/values for fast autoregressive generation |
| Speculative Decoding | [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192) | Draft model + verify for 2-3x speedup |

---

## Contributing

Found a bug? Have a cleaner implementation? PRs welcome.

## License

MIT - do whatever you want with it.
