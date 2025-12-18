# Transformer Problems

<p align="center">
  <strong>18 limitations of transformers — and how researchers are fixing them.</strong>
</p>

<p align="center">
  <a href="#efficiency">Efficiency</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#scaling">Scaling</a> •
  <a href="#training">Training</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#architectures">Architectures</a>
</p>

---

Why are LLMs slow? Why do they forget your conversation? Why does GPT-3 cost $4.6M to train?

This repo explains each problem with infographics, plain English, and the solutions being used today.

---

## At a Glance

| Category | Problems | Key Issue |
|----------|----------|-----------|
| [**Efficiency**](#efficiency) | Quadratic complexity, Memory, Compute, Hardware | O(N²) attention doesn't scale |
| [**Architecture**](#architecture) | Position, Local bias, Smoothing, No recurrence | Missing inductive biases |
| [**Scaling**](#scaling) | Context, Data hunger, Length, Model size | Hard limits on what fits |
| [**Training**](#training) | Instability, Noise, Interpretability | Tricky to train well |
| [**Deployment**](#deployment) | Slow decoding, Dense inputs, Real-time | Too slow for production |

---

<h2 id="efficiency">⚡ Efficiency Issues</h2>

The big one: attention is O(N²). Everything else follows from this.

| # | Problem | What Goes Wrong | Solution | Demo |
|:-:|---------|-----------------|----------|:----:|
| 01 | [**Quadratic Complexity**](./01_quadratic_complexity/) | Double sequence → 4× compute | Sparse/Linear attention | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/01_quadratic_complexity/demo.ipynb) |
| 07 | [**Memory Footprint**](./07_memory_footprint/) | KV cache fills GPU memory | FlashAttention, GQA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/07_memory_footprint/demo.ipynb) |
| 08 | [**Compute Cost**](./08_compute_cost/) | GPT-3: $4.6M to train | Efficient architectures | — |
| 17 | [**Hardware Inefficiency**](./17_hardware_inefficiency/) | GPU waits for memory | Kernel fusion, FlashAttn | — |

---

<h2 id="architecture">🏗️ Architecture Limitations</h2>

What transformers don't "know" by default.

| # | Problem | What Goes Wrong | Solution | Demo |
|:-:|---------|-----------------|----------|:----:|
| 02 | [**No Positional Awareness**](./02_positional_awareness/) | "Dog bites man" = "Man bites dog" | Positional encoding, RoPE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/02_positional_awareness/demo.ipynb) |
| 05 | [**No Local Bias**](./05_local_bias/) | Doesn't know nearby pixels matter | Convolutions, local attention | — |
| 11 | [**Attention Smoothing**](./11_attention_smoothing/) | Deep layers → all tokens same | Skip connections, careful init | — |
| 12 | [**No Recurrence**](./12_no_recurrence/) | Needs all tokens upfront | Streaming, state-space models | — |

---

<h2 id="scaling">📈 Scaling Challenges</h2>

What happens when you try to go bigger.

| # | Problem | What Goes Wrong | Solution | Demo |
|:-:|---------|-----------------|----------|:----:|
| 03 | [**Fixed Context**](./03_fixed_context/) | Can't "remember" beyond window | Longer context, RAG | — |
| 06 | [**Data Hungry**](./06_data_hungry/) | ViT needs 100× more than CNN | Better augmentation, pretraining | — |
| 09 | [**Length Generalization**](./09_length_generalization/) | Trained on 4K, breaks at 8K | RoPE, ALiBi | — |
| 13 | [**Model Size**](./13_model_size/) | 175B params = 350GB | Quantization, LoRA, MoE | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/13_model_size/demo.ipynb) |

---

<h2 id="training">🔧 Training & Quality</h2>

Why training large models is tricky.

| # | Problem | What Goes Wrong | Solution | Demo |
|:-:|---------|-----------------|----------|:----:|
| 10 | [**Training Instability**](./10_training_instability/) | Loss spikes, divergence | Pre-LN, careful LR | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/10_training_instability/demo.ipynb) |
| 14 | [**Noise Sensitivity**](./14_noise_sensitivity/) | Wastes capacity on padding | Better masking | — |
| 15 | [**Interpretability**](./15_interpretability/) | Attention ≠ explanation | Probing, mechanistic interp | — |

---

<h2 id="deployment">🚀 Deployment</h2>

Getting transformers to production.

| # | Problem | What Goes Wrong | Solution | Demo |
|:-:|---------|-----------------|----------|:----:|
| 04 | [**Slow Decoding**](./04_slow_decoding/) | One token at a time | Speculative decoding, batching | — |
| 16 | [**Dense Inputs**](./16_dense_inputs/) | Images → thousands of tokens | Patch merging, early exit | — |
| 18 | [**Real-Time**](./18_realtime_deployment/) | Too slow for voice/games | Distillation, quantization | — |

---

<h2 id="architectures">🤖 Transformer Architectures</h2>

Understand the problems → understand the solutions. Here are 10 architectures implemented from scratch.

<p align="center">
  <a href="./transformer_architectures/">
    <img src="./transformer_architectures/banner.png" alt="Transformer Architectures" width="80%"/>
  </a>
</p>

| Architecture | Type | Complexity | Solves | Demo |
|--------------|------|:----------:|--------|:----:|
| [Vanilla Transformer](./transformer_architectures/01_vanilla_transformer/) | Enc-Dec | O(N²) | Foundation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| [BERT](./transformer_architectures/02_bert/) | Encoder | O(N²) | Understanding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| [GPT](./transformer_architectures/03_gpt/) | Decoder | O(N²) | Generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| [ViT](./transformer_architectures/04_vision_transformer/) | Encoder | O(N²) | Images | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |
| [Transformer-XL](./transformer_architectures/05_transformer_xl/) | Decoder | O(N²)/seg | Fixed context (#3) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| [Sparse](./transformer_architectures/06_sparse_transformer/) | Any | O(N√N) | Quadratic (#1) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| [Performer](./transformer_architectures/07_performer/) | Any | O(N) | Quadratic (#1) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| [Reformer](./transformer_architectures/08_reformer/) | Any | O(N log N) | Memory (#7) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| [Longformer](./transformer_architectures/09_longformer/) | Encoder | O(N) | Fixed context (#3) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |
| [Switch](./transformer_architectures/10_switch_transformer/) | Any | O(N²) | Model size (#13) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

**[→ See all architectures with detailed explanations](./transformer_architectures/)**

---

## Quick Start

**Run a demo**: Click any Colab badge above.

**Explore locally**:
```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems

# View a problem
cat 01_quadratic_complexity/README.md

# Train an architecture
cd transformer_architectures
pip install torch matplotlib numpy
jupyter notebook 01_vanilla_transformer/demo.ipynb
```

**Regenerate infographics**:
```bash
python generate_images.py
```

---

## Key Papers

| Problem | Solution Paper |
|---------|----------------|
| Quadratic attention | [FlashAttention](https://arxiv.org/abs/2205.14135), [Longformer](https://arxiv.org/abs/2004.05150) |
| Position encoding | [RoPE](https://arxiv.org/abs/2104.09864), [ALiBi](https://arxiv.org/abs/2108.12409) |
| Model size | [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314) |
| No recurrence | [Mamba](https://arxiv.org/abs/2312.00752), [RWKV](https://arxiv.org/abs/2305.13048) |
| Slow decoding | [Speculative Decoding](https://arxiv.org/abs/2211.17192) |

---

## Contributing

Found an error? Know a better solution? PRs welcome.

## License

MIT
