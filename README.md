<p align="center">
  <img src="https://img.shields.io/badge/Problems-18-red?style=for-the-badge" alt="Problems"/>
  <img src="https://img.shields.io/badge/Architectures-10-blue?style=for-the-badge" alt="Architectures"/>
  <img src="https://img.shields.io/badge/Colab_Notebooks-15-orange?style=for-the-badge" alt="Notebooks"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🔥 Transformer Problems</h1>

<p align="center">
  <strong>18 limitations of transformer architectures — and how researchers are fixing them</strong>
</p>

<p align="center">
  <em>Why is ChatGPT so slow? Why can't it remember what you said 10 minutes ago?<br/>Why did it cost $4.6 million to train GPT-3?</em>
</p>

---

## 📖 What's Inside

| Section | Description |
|---------|-------------|
| 🔴 [**Problems**](#-the-problems) | 18 documented limitations with infographics |
| 🔵 [**Architectures**](#-transformer-architectures) | 10 implementations from scratch |
| 🟢 [**Notebooks**](#-quick-links) | Runnable Colab demos |

---

## 🔴 The Problems

### ⚡ Efficiency Issues

> *The O(N²) problem — where everything gets expensive*

| # | Problem | What Happens | Solution | Demo |
|:-:|---------|--------------|----------|:----:|
| 01 | [**Quadratic Complexity**](./01_quadratic_complexity/) | Double sequence → 4× compute | Sparse/Linear attention | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/01_quadratic_complexity/demo.ipynb) |
| 07 | [**Memory Footprint**](./07_memory_footprint/) | KV cache fills GPU memory | FlashAttention, GQA | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/07_memory_footprint/demo.ipynb) |
| 08 | [**Compute Cost**](./08_compute_cost/) | GPT-3 training: $4.6M | Efficient architectures | — |
| 17 | [**Hardware Inefficiency**](./17_hardware_inefficiency/) | GPU sits idle waiting | Kernel fusion | — |

### 🏗️ Architecture Gaps

> *What transformers don't "know" by default*

| # | Problem | What Happens | Solution | Demo |
|:-:|---------|--------------|----------|:----:|
| 02 | [**No Positional Awareness**](./02_positional_awareness/) | "Dog bites man" = "Man bites dog" | RoPE, ALiBi | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/02_positional_awareness/demo.ipynb) |
| 05 | [**No Local Bias**](./05_local_bias/) | Doesn't know nearby pixels matter | Local attention | — |
| 11 | [**Attention Smoothing**](./11_attention_smoothing/) | Deep layers → tokens look same | Skip connections | — |
| 12 | [**No Recurrence**](./12_no_recurrence/) | Needs all tokens upfront | SSMs, Mamba | — |

### 📈 Scaling Walls

> *What breaks when you go bigger*

| # | Problem | What Happens | Solution | Demo |
|:-:|---------|--------------|----------|:----:|
| 03 | [**Fixed Context**](./03_fixed_context/) | Hard limit on memory | Longer context, RAG | — |
| 06 | [**Data Hungry**](./06_data_hungry/) | ViT needs 100× more than CNN | Better pretraining | — |
| 09 | [**Length Generalization**](./09_length_generalization/) | Trained on 4K, breaks at 8K | RoPE, ALiBi | — |
| 13 | [**Model Size**](./13_model_size/) | 175B params = 350GB | LoRA, MoE | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/13_model_size/demo.ipynb) |

### 🔧 Training Challenges

> *Why large-scale training is tricky*

| # | Problem | What Happens | Solution | Demo |
|:-:|---------|--------------|----------|:----:|
| 10 | [**Training Instability**](./10_training_instability/) | Loss spikes, divergence | Pre-LN, careful LR | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/10_training_instability/demo.ipynb) |
| 14 | [**Noise Sensitivity**](./14_noise_sensitivity/) | Wastes capacity on padding | Better masking | — |
| 15 | [**Interpretability**](./15_interpretability/) | Attention ≠ explanation | Probing, mechanistic | — |

### 🚀 Deployment Blockers

> *Getting transformers to production*

| # | Problem | What Happens | Solution | Demo |
|:-:|---------|--------------|----------|:----:|
| 04 | [**Slow Decoding**](./04_slow_decoding/) | One token at a time | Speculative decoding | — |
| 16 | [**Dense Inputs**](./16_dense_inputs/) | Images → thousands of tokens | Patch merging | — |
| 18 | [**Real-Time**](./18_realtime_deployment/) | Too slow for voice/games | Quantization | — |

---

## 🔵 Transformer Architectures

<p align="center">
  <a href="./transformer_architectures/">
    <img src="./transformer_architectures/banner.png" alt="Transformer Architectures" width="85%"/>
  </a>
</p>

<p align="center">
  <em>10 architectures implemented from scratch — click any Colab badge to train</em>
</p>

### 🏛️ The Classics

| | Architecture | Type | What It Does | Train It |
|:-:|--------------|:----:|--------------|:--------:|
| 01 | [**Vanilla Transformer**](./transformer_architectures/01_vanilla_transformer/) | Enc-Dec | The 2017 original | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| 02 | [**BERT**](./transformer_architectures/02_bert/) | Encoder | Bidirectional understanding | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| 03 | [**GPT**](./transformer_architectures/03_gpt/) | Decoder | ChatGPT's architecture | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| 04 | [**Vision Transformer**](./transformer_architectures/04_vision_transformer/) | Encoder | Images as patches | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |

### ⚡ Efficient Variants

| | Architecture | Complexity | The Trick | Train It |
|:-:|--------------|:----------:|-----------|:--------:|
| 05 | [**Transformer-XL**](./transformer_architectures/05_transformer_xl/) | O(N²)/seg | Cache across segments | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| 06 | [**Sparse**](./transformer_architectures/06_sparse_transformer/) | O(N√N) | Attend to subset | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| 07 | [**Performer**](./transformer_architectures/07_performer/) | O(N) | Random features | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| 08 | [**Reformer**](./transformer_architectures/08_reformer/) | O(N log N) | LSH attention | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| 09 | [**Longformer**](./transformer_architectures/09_longformer/) | O(N) | Sliding window | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |
| 10 | [**Switch**](./transformer_architectures/10_switch_transformer/) | O(N²) | MoE routing | [![Colab](https://img.shields.io/badge/▶-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

<p align="center">
  <a href="./transformer_architectures/"><strong>→ Full Architecture Guide</strong></a>
</p>

---

## 🚀 Quick Start

<table>
<tr>
<td width="50%">

### Run in Colab
Click any **▶ Colab** badge above — everything runs in browser, no setup needed.

</td>
<td width="50%">

### Run Locally
```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems
pip install torch matplotlib numpy
```

</td>
</tr>
</table>

---

## 📚 Key Papers

| Problem | Solution Paper |
|---------|----------------|
| Quadratic attention | [FlashAttention](https://arxiv.org/abs/2205.14135) ⭐ |
| Position encoding | [RoPE](https://arxiv.org/abs/2104.09864) |
| Model size | [LoRA](https://arxiv.org/abs/2106.09685) |
| No recurrence | [Mamba](https://arxiv.org/abs/2312.00752) |
| Slow decoding | [Speculative Decoding](https://arxiv.org/abs/2211.17192) |

---

## 🤝 Contributing

Found an error? Know a better explanation? PRs welcome.

---

<p align="center">
  <strong>MIT License</strong> — do whatever you want with it
</p>
