# Transformer Problems

A practical guide to the 18 biggest limitations of transformer architectures — and how researchers are fixing them.

If you've ever wondered why LLMs are slow, expensive, or can't remember your conversation from yesterday, you're in the right place.

---

## Why This Exists

Transformers power everything from ChatGPT to image generators. But they come with real limitations that affect how we build and deploy AI systems. This repo documents each problem with:

- Visual explanations (infographics)
- Plain English descriptions (no jargon soup)
- Practical solutions being used today
- Links to the key papers

Whether you're a student trying to understand transformers, an engineer deploying them, or a researcher working on improvements — this is for you.

---

## The Problems

### Efficiency Issues

| # | Problem | One-Line Summary |
|---|---------|------------------|
| 1 | [Quadratic Complexity](./01_quadratic_complexity/) | Attention is O(N²) — double your sequence, quadruple your compute |
| 7 | [Memory Footprint](./07_memory_footprint/) | KV cache and activations eat your GPU memory |
| 8 | [Compute Cost](./08_compute_cost/) | Training GPT-3 cost $4.6 million |
| 17 | [Hardware Inefficiency](./17_hardware_inefficiency/) | GPUs sit idle waiting for memory |

### Architecture Limitations  

| # | Problem | One-Line Summary |
|---|---------|------------------|
| 2 | [No Positional Awareness](./02_positional_awareness/) | "Dog bites man" = "Man bites dog" without PE |
| 5 | [No Local Bias](./05_local_bias/) | Doesn't know nearby pixels matter more |
| 11 | [Attention Smoothing](./11_attention_smoothing/) | Deep layers make all tokens look the same |
| 12 | [No Recurrence](./12_no_recurrence/) | Can't stream — needs all tokens first |

### Scaling Challenges

| # | Problem | One-Line Summary |
|---|---------|------------------|
| 3 | [Fixed Context](./03_fixed_context/) | Hard limit on how much it can "remember" |
| 6 | [Data Hungry](./06_data_hungry/) | Needs 100x more data than CNNs |
| 9 | [Length Generalization](./09_length_generalization/) | Trained on 4K tokens, breaks at 8K |
| 13 | [Model Size](./13_model_size/) | 175B parameters = 350GB of weights |

### Training & Quality

| # | Problem | One-Line Summary |
|---|---------|------------------|
| 10 | [Training Instability](./10_training_instability/) | Loss spikes and divergence at scale |
| 14 | [Noise Sensitivity](./14_noise_sensitivity/) | Wastes capacity attending to padding tokens |
| 15 | [Interpretability](./15_interpretability/) | Attention weights don't explain decisions |

### Deployment

| # | Problem | One-Line Summary |
|---|---------|------------------|
| 4 | [Slow Decoding](./04_slow_decoding/) | Generates one token at a time |
| 16 | [Dense Inputs](./16_dense_inputs/) | Images become thousands of tokens |
| 18 | [Real-Time](./18_realtime_deployment/) | Too slow for voice assistants and games |

---

## Transformer Architectures

Okay, so you know the problems. But how do the actual architectures work?

I've implemented 10 transformer variants from scratch — BERT, GPT, ViT, and the efficient ones like Longformer and Performer. Each one trains on a tiny dataset in Colab so you can see it working.

<p align="center">
  <a href="./transformer_architectures/">
    <img src="./transformer_architectures/banner.png" alt="Transformer Architectures" width="80%"/>
  </a>
</p>

**The Classics:**

| Architecture | What It Is | Run It |
|--------------|------------|--------|
| [Vanilla Transformer](./transformer_architectures/01_vanilla_transformer/) | The original encoder-decoder | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/01_vanilla_transformer/demo.ipynb) |
| [BERT](./transformer_architectures/02_bert/) | Bidirectional encoder (understanding) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/02_bert/demo.ipynb) |
| [GPT](./transformer_architectures/03_gpt/) | Autoregressive decoder (generation) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb) |
| [Vision Transformer](./transformer_architectures/04_vision_transformer/) | Images as patch sequences | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb) |

**The Efficient Ones** (when O(N²) hurts):

| Architecture | Complexity | The Trick | Run It |
|--------------|------------|-----------|--------|
| [Transformer-XL](./transformer_architectures/05_transformer_xl/) | O(N²)/seg | Cache states across segments | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/05_transformer_xl/demo.ipynb) |
| [Sparse](./transformer_architectures/06_sparse_transformer/) | O(N√N) | Only attend to some tokens | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/06_sparse_transformer/demo.ipynb) |
| [Performer](./transformer_architectures/07_performer/) | O(N) | Random feature approximation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/07_performer/demo.ipynb) |
| [Reformer](./transformer_architectures/08_reformer/) | O(N log N) | LSH to find similar queries | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/08_reformer/demo.ipynb) |
| [Longformer](./transformer_architectures/09_longformer/) | O(N) | Sliding window + global tokens | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/09_longformer/demo.ipynb) |
| [Switch](./transformer_architectures/10_switch_transformer/) | O(N²) | MoE: trillion params, same FLOPs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb) |

[**See all architectures →**](./transformer_architectures/)

---

## Quick Links

Jump to any problem:

| # | Problem | Docs | Colab |
|---|---------|------|-------|
| 1 | Quadratic Complexity O(N²) | [📖 README](./01_quadratic_complexity/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/01_quadratic_complexity/demo.ipynb) |
| 2 | No Positional Awareness | [📖 README](./02_positional_awareness/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/02_positional_awareness/demo.ipynb) |
| 3 | Fixed Context Window | [📖 README](./03_fixed_context/) | Coming Soon |
| 4 | Slow Autoregressive Decoding | [📖 README](./04_slow_decoding/) | Coming Soon |
| 5 | No Local Inductive Bias | [📖 README](./05_local_bias/) | Coming Soon |
| 6 | Data-Hungry Architecture | [📖 README](./06_data_hungry/) | Coming Soon |
| 7 | High Memory Footprint | [📖 README](./07_memory_footprint/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/07_memory_footprint/demo.ipynb) |
| 8 | High Compute & Power Cost | [📖 README](./08_compute_cost/) | Coming Soon |
| 9 | Poor Length Generalization | [📖 README](./09_length_generalization/) | Coming Soon |
| 10 | Training Instability | [📖 README](./10_training_instability/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/10_training_instability/demo.ipynb) |
| 11 | Attention Over-Smoothing | [📖 README](./11_attention_smoothing/) | Coming Soon |
| 12 | No Recurrence / Streaming | [📖 README](./12_no_recurrence/) | Coming Soon |
| 13 | Large Model Size | [📖 README](./13_model_size/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/13_model_size/demo.ipynb) |
| 14 | Sensitivity to Noise Tokens | [📖 README](./14_noise_sensitivity/) | Coming Soon |
| 15 | Poor Interpretability | [📖 README](./15_interpretability/) | Coming Soon |
| 16 | Inefficient for Dense Inputs | [📖 README](./16_dense_inputs/) | Coming Soon |
| 17 | Hardware Inefficiency | [📖 README](./17_hardware_inefficiency/) | Coming Soon |
| 18 | Real-Time Deployment | [📖 README](./18_realtime_deployment/) | Coming Soon |

---

## Regenerate Images

Each problem folder has a `problem.png` infographic. To regenerate all of them:

```bash
python generate_images.py
```

Requires Python 3.7+ and matplotlib.

---

## Key Papers

If you want to go deeper, start with these:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Solving the memory problem
- [RoPE](https://arxiv.org/abs/2104.09864) — Better position encoding
- [LoRA](https://arxiv.org/abs/2106.09685) — Training giant models affordably
- [Mamba](https://arxiv.org/abs/2312.00752) — The SSM alternative

---

## Contributing

Found an error? Know a better solution? PRs welcome.

- Add new problems
- Improve explanations
- Fix inaccuracies
- Add more references

---

## License

MIT — use this however you want.
