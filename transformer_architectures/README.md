# Transformer Architectures

<p align="center">
  <img src="banner.png" alt="Transformer Architectures" width="100%"/>
</p>

<p align="center">
  <strong>10 transformer variants, from scratch, with training code you can run in 2 minutes.</strong>
</p>

<p align="center">
  <a href="#the-classics">Classics</a> •
  <a href="#efficient-attention">Efficient</a> •
  <a href="#scaling">Scaling</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#papers">Papers</a>
</p>

---

## At a Glance

| # | Architecture | Type | Complexity | Best For | Demo |
|:-:|--------------|------|:----------:|----------|:----:|
| 01 | [**Vanilla Transformer**](./01_vanilla_transformer/) | Enc-Dec | O(N²) | Foundation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| 02 | [**BERT**](./02_bert/) | Encoder | O(N²) | Understanding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| 03 | [**GPT**](./03_gpt/) | Decoder | O(N²) | Generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| 04 | [**Vision Transformer**](./04_vision_transformer/) | Encoder | O(N²) | Images | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |
| 05 | [**Transformer-XL**](./05_transformer_xl/) | Decoder | O(N²)/seg | Long context | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| 06 | [**Sparse Transformer**](./06_sparse_transformer/) | Any | O(N√N) | Very long | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| 07 | [**Performer**](./07_performer/) | Any | O(N) | Linear attention | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| 08 | [**Reformer**](./08_reformer/) | Any | O(N log N) | Memory efficient | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| 09 | [**Longformer**](./09_longformer/) | Encoder | O(N) | Documents | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |
| 10 | [**Switch Transformer**](./10_switch_transformer/) | Any | O(N²) | Trillion scale | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

---

<h2 id="the-classics">🏛️ The Classics (Foundation)</h2>

These are the architectures you need to understand first.

### [01. Vanilla Transformer](./01_vanilla_transformer/)
> The 2017 paper that started it all: "Attention Is All You Need"

- **Architecture**: Encoder-Decoder with self-attention
- **Key equation**: `Attention(Q,K,V) = softmax(QK^T/√d)V`
- **Used in**: Machine translation, seq2seq tasks

### [02. BERT](./02_bert/)
> Encoder-only. Bidirectional. Great for understanding text.

- **Key insight**: Mask 15% of tokens, predict them (MLM)
- **Sees**: Both left AND right context
- **Used in**: Classification, NER, QA, embeddings

### [03. GPT](./03_gpt/)
> Decoder-only. The architecture behind ChatGPT, Claude, LLaMA.

- **Key insight**: Just predict the next token. Scale it up.
- **Sees**: Only left context (causal mask)
- **Used in**: Text generation, chatbots, code completion

### [04. Vision Transformer](./04_vision_transformer/)
> "An image is worth 16x16 words"

- **Key insight**: Split image into patches, treat as tokens
- **224×224 image** → 196 patches → transformer → classification
- **Used in**: Image classification, object detection

---

<h2 id="efficient-attention">⚡ Efficient Attention (Breaking O(N²))</h2>

Standard attention is O(N²). These architectures fix that.

### [05. Transformer-XL](./05_transformer_xl/)
> Segment-level recurrence for longer context

- **Problem**: Fixed context window
- **Solution**: Cache hidden states from previous segments
- **Complexity**: O(N²) per segment, but extended context

### [06. Sparse Transformer](./06_sparse_transformer/)
> Only attend to some tokens

- **Pattern**: Local window + strided (every k-th token)
- **Complexity**: O(N√N) instead of O(N²)
- **Used in**: GPT-3's sparse attention layers

### [07. Performer](./07_performer/)
> True linear attention via random features

- **Trick**: Approximate softmax with `φ(Q)(φ(K)^T V)`
- **Complexity**: O(N) — compute K^T V first!
- **Trade-off**: Approximation (not exact attention)

### [08. Reformer](./08_reformer/)
> LSH attention + reversible layers

- **LSH**: Hash similar queries into buckets, attend within buckets
- **Reversible**: Recompute activations during backprop (save memory)
- **Complexity**: O(N log N)

### [09. Longformer](./09_longformer/)
> Sliding window + global attention for special tokens

- **Local**: Each token attends to window of nearby tokens
- **Global**: [CLS] and question tokens see everything
- **Complexity**: O(N × window_size)
- **Used in**: Long document QA, summarization

---

<h2 id="scaling">🚀 Scaling (More Parameters, Same FLOPs)</h2>

### [10. Switch Transformer](./10_switch_transformer/)
> Mixture of Experts: trillion parameters, constant compute

- **Idea**: Route each token to 1 of N expert FFNs
- **Result**: N× more parameters, same FLOPs per token
- **Scale**: Switch-C has 1.6T parameters!
- **Descendants**: Mixtral, DeepSeek-V3

---

<h2 id="quick-start">🚀 Quick Start</h2>

**Fastest way**: Click any Colab badge above. Everything installs automatically.

**Local setup**:
```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems/transformer_architectures
pip install torch matplotlib numpy
jupyter notebook
```

**Learning path**:

| Level | Start Here |
|-------|------------|
| **Beginner** | [Vanilla Transformer](./01_vanilla_transformer/) → [GPT](./03_gpt/) → [BERT](./02_bert/) |
| **Intermediate** | [ViT](./04_vision_transformer/) → [Longformer](./09_longformer/) |
| **Advanced** | [Performer](./07_performer/) → [Switch Transformer](./10_switch_transformer/) |

---

## 📊 Evolution Timeline

```
2017 ─── Vanilla Transformer
          │
2018 ─────┼── BERT (encoder)
          └── GPT (decoder)
          
2019 ─────┼── Transformer-XL (memory)
          └── Sparse Transformer (O(N√N))
          
2020 ─────┼── GPT-3 (175B)
          ├── ViT (images)
          ├── Longformer (documents)
          ├── Performer (O(N))
          └── Reformer (LSH)
          
2021 ─────┼── Switch Transformer (MoE)
          └── RoPE (position encoding)
          
2022 ─────┼── ChatGPT
          └── FlashAttention (IO-aware)
          
2023 ─────┼── LLaMA, Mistral, Mixtral
          ├── Mamba (no attention!)
          └── FlashAttention-2
          
2024 ─────┼── LLaMA-3, Qwen-2, Phi-3
          ├── DeepSeek-V3 ($5.5M training!)
          └── FlashAttention-3
```

---

## 🔧 What to Use in 2025

| Task | Recommended |
|------|-------------|
| Classification | BERT, DeBERTa |
| Generation (small) | Phi-3, Mistral 7B |
| Generation (large) | LLaMA-3.1, Qwen-2.5 |
| Long documents | Longformer, or native long-context models |
| Images | ViT, Swin, DINOv2 |
| Efficiency at scale | Mixtral, DeepSeek-V3 (MoE) |
| Non-attention | Mamba, RWKV |

---

<h2 id="papers">📚 Key Papers</h2>

<details>
<summary><strong>Foundations (2017-2020)</strong></summary>

| Paper | Year | Innovation |
|-------|------|------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Bidirectional pretraining |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Scaled LM |
| [Transformer-XL](https://arxiv.org/abs/1901.02860) | 2019 | Segment recurrence |
| [Sparse Transformers](https://arxiv.org/abs/1904.10509) | 2019 | O(N√N) |
| [Longformer](https://arxiv.org/abs/2004.05150) | 2020 | Sliding window |
| [Performer](https://arxiv.org/abs/2009.14794) | 2020 | FAVOR+ |
| [ViT](https://arxiv.org/abs/2010.11929) | 2020 | Image patches |
</details>

<details>
<summary><strong>LLM Era (2021-2023)</strong></summary>

| Paper | Year | Innovation |
|-------|------|------------|
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 2021 | MoE top-1 |
| [RoPE](https://arxiv.org/abs/2104.09864) | 2021 | Rotary PE |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | IO-aware |
| [LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | Open LLM |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | GQA + SWA |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | SSM |
</details>

<details>
<summary><strong>Current (2024-2025)</strong></summary>

| Paper | Year | Innovation |
|-------|------|------------|
| [LLaMA-3](https://ai.meta.com/blog/meta-llama-3/) | 2024 | 15T tokens |
| [Phi-3](https://arxiv.org/abs/2404.14219) | 2024 | Small & mighty |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 | 671B MoE, $5.5M |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | 2024 | Hopper GPUs |
</details>

<details>
<summary><strong>Key Techniques</strong></summary>

| Technique | Why It Matters |
|-----------|----------------|
| **RoPE** | Position encoding that extrapolates |
| **GQA** | Balance between MHA and MQA |
| **SwiGLU** | Better FFN activation |
| **FlashAttention** | Fast exact attention |
| **KV Cache** | Fast autoregressive generation |
</details>

---

## Contributing

Found a bug? Have a cleaner implementation? PRs welcome.

## License

MIT
