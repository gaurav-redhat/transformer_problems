# Transformer Problems

Why is ChatGPT so slow? Why can't it remember what you said 10 minutes ago? Why did it cost $4.6 million to train GPT-3?

These aren't bugs — they're fundamental limitations of the transformer architecture. This repo breaks down 18 of them, explains why they exist, and shows how researchers are working around them.

---

## What's in here

I've documented each problem with:
- An infographic (because walls of text are annoying)
- A plain English explanation
- The solutions people are actually using
- Links to papers if you want to go deeper

I've also implemented 10 transformer architectures from scratch with training code you can run in Colab. Because reading about attention is one thing — watching it learn is another.

---

## The Problems

### The O(N²) problem and its friends

This is the big one. Attention compares every token to every other token. Double your sequence length, quadruple your compute. Everything else flows from here.

| # | Problem | TL;DR |
|---|---------|-------|
| [01](./01_quadratic_complexity/) | **Quadratic complexity** | The reason you can't just throw a million tokens at GPT |
| [07](./07_memory_footprint/) | **Memory blowup** | KV cache alone can eat 30GB for a 70B model |
| [08](./08_compute_cost/) | **Training cost** | GPT-3 training: ~$4.6M. GPT-4: probably 10x that |
| [17](./17_hardware_inefficiency/) | **GPU utilization** | Your expensive A100 spends most of its time waiting |

### Things transformers don't "know"

Transformers are surprisingly dumb out of the box. They don't understand order, they don't know nearby things are related, they can't process a stream.

| # | Problem | TL;DR |
|---|---------|-------|
| [02](./02_positional_awareness/) | **No sense of order** | Without position encoding, "dog bites man" = "man bites dog" |
| [05](./05_local_bias/) | **No local bias** | CNNs know nearby pixels matter. Transformers don't |
| [11](./11_attention_smoothing/) | **Attention collapse** | Go deep enough and everything looks the same |
| [12](./12_no_recurrence/) | **No streaming** | Need all tokens before you can start. RNNs didn't have this problem |

### Scaling headaches

| # | Problem | TL;DR |
|---|---------|-------|
| [03](./03_fixed_context/) | **Fixed context** | There's a hard limit on how much it can "see" at once |
| [06](./06_data_hungry/) | **Data hungry** | ViT needs ~100x more data than a CNN to match it |
| [09](./09_length_generalization/) | **Length generalization** | Train on 4K tokens, give it 8K, watch it break |
| [13](./13_model_size/) | **Model size** | 175B parameters = 350GB just for the weights |

### Training is hard

| # | Problem | TL;DR |
|---|---------|-------|
| [10](./10_training_instability/) | **Training instability** | Random loss spikes at scale. Nobody fully understands why |
| [14](./14_noise_sensitivity/) | **Padding waste** | Model spends capacity attending to meaningless [PAD] tokens |
| [15](./15_interpretability/) | **Black box** | Attention weights look meaningful. They're often not |

### Deployment reality

| # | Problem | TL;DR |
|---|---------|-------|
| [04](./04_slow_decoding/) | **Slow generation** | One token at a time. Can't parallelize the output |
| [16](./16_dense_inputs/) | **Images are expensive** | A 224×224 image becomes 196 tokens. A 1024×1024? 4096 tokens |
| [18](./18_realtime_deployment/) | **Too slow for real-time** | Try putting this in a video game or voice assistant |

---

## The Architectures

Understanding problems is half the battle. Here's how different architectures solve them.

<p align="center">
  <a href="./transformer_architectures/">
    <img src="./transformer_architectures/banner.png" alt="Transformer Architectures" width="80%"/>
  </a>
</p>

I've implemented these from scratch with runnable training code:

| Architecture | What it fixes | The trick |
|--------------|---------------|-----------|
| [Vanilla Transformer](./transformer_architectures/01_vanilla_transformer/) | — | The starting point. Understand this first |
| [BERT](./transformer_architectures/02_bert/) | — | Encoder-only. Sees both directions |
| [GPT](./transformer_architectures/03_gpt/) | — | Decoder-only. The ChatGPT architecture |
| [ViT](./transformer_architectures/04_vision_transformer/) | — | Images as patch sequences |
| [Transformer-XL](./transformer_architectures/05_transformer_xl/) | Fixed context | Caches hidden states across segments |
| [Sparse Transformer](./transformer_architectures/06_sparse_transformer/) | O(N²) | Only attends to some tokens |
| [Performer](./transformer_architectures/07_performer/) | O(N²) | Approximates attention in O(N) |
| [Reformer](./transformer_architectures/08_reformer/) | Memory | LSH + reversible layers |
| [Longformer](./transformer_architectures/09_longformer/) | Fixed context | Sliding window + global tokens |
| [Switch Transformer](./transformer_architectures/10_switch_transformer/) | Model size | Routes tokens to different experts |

Each folder has a Colab notebook. Click, run, watch it train.

[**→ Full architecture guide**](./transformer_architectures/)

---

## Running the code

Most demos run in Colab (click the badges). If you want to run locally:

```bash
git clone https://github.com/gaurav-redhat/transformer_problems.git
cd transformer_problems
pip install torch matplotlib numpy
```

To regenerate the infographics:
```bash
python generate_images.py
```

---

## Papers worth reading

If you want to go deeper:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — where it all started
- [FlashAttention](https://arxiv.org/abs/2205.14135) — clever memory tricks that actually work
- [RoPE](https://arxiv.org/abs/2104.09864) — position encoding that doesn't break at longer sequences
- [LoRA](https://arxiv.org/abs/2106.09685) — fine-tuning without fine-tuning all the weights
- [Mamba](https://arxiv.org/abs/2312.00752) — what if we didn't use attention at all?

---

## Contributing

Found something wrong? Know a better explanation? PRs welcome.

## License

MIT — do whatever you want with it.
