<p align="center">
  <img src="banner.png" alt="Transformer Architectures" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architectures-10-blue?style=for-the-badge" alt="Architectures"/>
  <img src="https://img.shields.io/badge/PyTorch-From_Scratch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Training-~2_min-green?style=for-the-badge" alt="Training"/>
</p>

<h1 align="center">🤖 Transformer Architectures</h1>

<p align="center">
  <strong>10 transformer variants implemented from scratch with training code</strong>
</p>

<p align="center">
  <em>I got tired of papers that explain attention with walls of math but never show you how it actually works.<br/>
  So I implemented everything from scratch — readable PyTorch that trains on tiny datasets in ~2 minutes.</em>
</p>

---

## 📋 What's Here

<table>
<tr>
<td>📊</td>
<td><strong>Architecture diagrams</strong></td>
<td>Visual explanation of each model</td>
</tr>
<tr>
<td>💻</td>
<td><strong>PyTorch code</strong></td>
<td>Clean, readable, no 500-line base classes</td>
</tr>
<tr>
<td>🔥</td>
<td><strong>Colab notebooks</strong></td>
<td>Train in browser, see it working</td>
</tr>
<tr>
<td>👁️</td>
<td><strong>Visualizations</strong></td>
<td>Attention patterns, loss curves</td>
</tr>
</table>

---

## 🏛️ The Classics

> *Understand these first — everything else builds on them*

<table>
<tr>
<td align="center" width="25%">
<a href="./01_vanilla_transformer/">
<img src="https://img.shields.io/badge/01-Vanilla-4285F4?style=for-the-badge" alt="Vanilla"/><br/>
<strong>Vanilla Transformer</strong><br/>
<sub>The 2017 original</sub><br/><br/>
<img src="https://img.shields.io/badge/Enc--Dec-O(N²)-lightgrey?style=flat-square"/>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb">
<img src="https://img.shields.io/badge/▶_Train_It-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</a>
</td>
<td align="center" width="25%">
<a href="./02_bert/">
<img src="https://img.shields.io/badge/02-BERT-34A853?style=for-the-badge" alt="BERT"/><br/>
<strong>BERT</strong><br/>
<sub>Bidirectional encoder</sub><br/><br/>
<img src="https://img.shields.io/badge/Encoder-O(N²)-lightgrey?style=flat-square"/>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb">
<img src="https://img.shields.io/badge/▶_Train_It-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</a>
</td>
<td align="center" width="25%">
<a href="./03_gpt/">
<img src="https://img.shields.io/badge/03-GPT-EA4335?style=for-the-badge" alt="GPT"/><br/>
<strong>GPT</strong><br/>
<sub>ChatGPT's architecture</sub><br/><br/>
<img src="https://img.shields.io/badge/Decoder-O(N²)-lightgrey?style=flat-square"/>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb">
<img src="https://img.shields.io/badge/▶_Train_It-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</a>
</td>
<td align="center" width="25%">
<a href="./04_vision_transformer/">
<img src="https://img.shields.io/badge/04-ViT-FBBC04?style=for-the-badge" alt="ViT"/><br/>
<strong>Vision Transformer</strong><br/>
<sub>Images as patches</sub><br/><br/>
<img src="https://img.shields.io/badge/Encoder-O(N²)-lightgrey?style=flat-square"/>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb">
<img src="https://img.shields.io/badge/▶_Train_It-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</a>
</td>
</tr>
</table>

---

## ⚡ Efficient Attention

> *Standard attention is O(N²). These architectures fix that.*

<table>
<tr>
<td align="center" width="20%">
<a href="./05_transformer_xl/">
<img src="https://img.shields.io/badge/05-XL-9C27B0?style=for-the-badge"/><br/>
<strong>Transformer-XL</strong><br/>
<sub>Memory caching</sub><br/>
<code>O(N²)/seg</code>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb">
<img src="https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab"/>
</a>
</td>
<td align="center" width="20%">
<a href="./06_sparse_transformer/">
<img src="https://img.shields.io/badge/06-Sparse-FF5722?style=for-the-badge"/><br/>
<strong>Sparse</strong><br/>
<sub>Attend to subset</sub><br/>
<code>O(N√N)</code>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb">
<img src="https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab"/>
</a>
</td>
<td align="center" width="20%">
<a href="./07_performer/">
<img src="https://img.shields.io/badge/07-Performer-00BCD4?style=for-the-badge"/><br/>
<strong>Performer</strong><br/>
<sub>Random features</sub><br/>
<code>O(N)</code>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb">
<img src="https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab"/>
</a>
</td>
<td align="center" width="20%">
<a href="./08_reformer/">
<img src="https://img.shields.io/badge/08-Reformer-795548?style=for-the-badge"/><br/>
<strong>Reformer</strong><br/>
<sub>LSH hashing</sub><br/>
<code>O(N log N)</code>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb">
<img src="https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab"/>
</a>
</td>
<td align="center" width="20%">
<a href="./09_longformer/">
<img src="https://img.shields.io/badge/09-Longformer-607D8B?style=for-the-badge"/><br/>
<strong>Longformer</strong><br/>
<sub>Sliding window</sub><br/>
<code>O(N)</code>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb">
<img src="https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab"/>
</a>
</td>
</tr>
</table>

---

## 🚀 Scaling Up

> *More parameters, same compute*

<table>
<tr>
<td align="center">
<a href="./10_switch_transformer/">
<img src="https://img.shields.io/badge/10-Switch_Transformer-E91E63?style=for-the-badge"/><br/><br/>
<strong>Mixture of Experts</strong><br/>
<sub>1.6 trillion parameters, constant FLOPs</sub><br/><br/>
<code>Route each token to 1 of N expert FFNs</code>
</a>
<br/><br/>
<a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb">
<img src="https://img.shields.io/badge/▶_Train_It-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</a>
</td>
</tr>
</table>

---

## 📊 Evolution Timeline

```
2017 ─── Vanilla Transformer ("Attention Is All You Need")
          │
2018 ────┼── BERT (encoder-only, bidirectional)
          └── GPT (decoder-only, autoregressive)
          
2019 ────┼── Transformer-XL (memory across segments)
          └── Sparse Transformer (O(N√N))
          
2020 ────┼── ViT (images as patches)
          ├── Longformer (sliding window)
          ├── Performer (random features)
          └── Reformer (LSH + reversible)
          
2021 ─── Switch Transformer (MoE at scale)

2022-2024 ─── FlashAttention, LLaMA, Mamba, Mixtral, DeepSeek
```

---

## 🎯 What to Use in 2025

| Task | My Recommendation |
|------|-------------------|
| 📝 Classification | BERT / DeBERTa |
| 💬 Chatbot | Mistral 7B / LLaMA 3.1 |
| 📄 Long documents | Native long-context (Gemini, Claude) |
| 🖼️ Images | ViT / DINOv2 |
| 💰 Scale on budget | Mixtral / DeepSeek (MoE) |
| 🔋 Efficiency | Mamba / RWKV |

---

## 🚀 Quick Start

<table>
<tr>
<td width="50%" valign="top">

### ☁️ Easiest: Colab
Click any **▶ Colab** badge above.
- No setup needed
- GPU included
- ~2 min to train

</td>
<td width="50%" valign="top">

### 💻 Local Setup
```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems/transformer_architectures
pip install torch matplotlib numpy
jupyter notebook
```

</td>
</tr>
</table>

---

## 🎓 Learning Path

| Level | Start Here |
|-------|------------|
| 🟢 **Beginner** | Vanilla → GPT → BERT |
| 🟡 **Intermediate** | ViT → Longformer |
| 🔴 **Advanced** | Performer → Switch Transformer |

---

## 📚 Papers

<details>
<summary><strong>🏛️ Foundations (2017-2020)</strong></summary>

| Paper | Year | Key Innovation |
|-------|:----:|----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Bidirectional pretraining |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Scaled LM |
| [Transformer-XL](https://arxiv.org/abs/1901.02860) | 2019 | Segment recurrence |
| [Sparse Transformers](https://arxiv.org/abs/1904.10509) | 2019 | O(N√N) |
| [ViT](https://arxiv.org/abs/2010.11929) | 2020 | Image patches |
| [Longformer](https://arxiv.org/abs/2004.05150) | 2020 | Sliding window |
| [Performer](https://arxiv.org/abs/2009.14794) | 2020 | FAVOR+ |

</details>

<details>
<summary><strong>🚀 Modern Era (2021-2024)</strong></summary>

| Paper | Year | Key Innovation |
|-------|:----:|----------------|
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 2021 | MoE top-1 |
| [RoPE](https://arxiv.org/abs/2104.09864) | 2021 | Rotary PE |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | IO-aware attention |
| [LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | Open LLM |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | State space model |
| [Mixtral](https://arxiv.org/abs/2401.04088) | 2024 | Open MoE |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 | 671B MoE, $5.5M |

</details>

---

<p align="center">
  <strong>🤝 Contributing</strong> — Found a bug? PRs welcome!<br/>
  <strong>📄 License</strong> — MIT
</p>
