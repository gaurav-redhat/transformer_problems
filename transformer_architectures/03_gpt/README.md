# GPT

[← Back](../README.md) | [← Prev: BERT](../02_bert/README.md) | [Next: ViT →](../04_vision_transformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb)

---

![Architecture](architecture.png)

This is it. The architecture behind ChatGPT, Claude, LLaMA, Mistral - pretty much every chatbot and code assistant you've used. The idea is absurdly simple: **predict the next token**. Do that well enough, at scale, and you get intelligence.

---

## The core idea

Train on massive text with one objective:

```
Input:  "The cat sat on the"
Output: "mat" (predict what comes next)
```

That's it. No fancy objectives, no task-specific heads. Just next token prediction, scaled up with more data and parameters.

Turns out this simple objective, applied to enough text, learns grammar, facts, reasoning, and code. Nobody fully understands why it works this well.

---

## Architecture

GPT is the decoder half of the original transformer:

```
Tokens
   ↓
Token Embedding + Position Embedding
   ↓
Decoder Block × N (with causal mask)
   ↓
Linear → Softmax
   ↓
Next Token Probabilities
```

No encoder. Just stack decoder layers and let it learn.

---

## The causal mask

This is the key difference from BERT. Each token can only see itself and past tokens:

```
Position:    1  2  3  4
Token 1:     ✓  ✗  ✗  ✗
Token 2:     ✓  ✓  ✗  ✗
Token 3:     ✓  ✓  ✓  ✗
Token 4:     ✓  ✓  ✓  ✓
```

When predicting token 4, the model only sees tokens 1-3. This lets it generate text autoregressively without cheating.

---

## Why decoder-only won

The original transformer had encoder + decoder. BERT used encoder-only. But for generation, decoder-only (GPT-style) won because:

1. **Simpler**: One stack, not two
2. **Unified**: Same architecture for training and generation
3. **Scalable**: No cross-attention overhead
4. **Emergent abilities**: Scale seems to unlock new capabilities

Every major LLM today is decoder-only.

---

## The evolution

| Model | Year | Params | What changed |
|-------|------|--------|--------------|
| GPT-1 | 2018 | 117M | Proved the concept |
| GPT-2 | 2019 | 1.5B | Zero-shot capabilities |
| GPT-3 | 2020 | 175B | In-context learning, few-shot |
| GPT-4 | 2023 | ~1.8T? | MoE, multimodal, RLHF |

The progression: same architecture, more scale, better data, alignment tuning.

---

## Generation

Once trained, you generate by sampling:

```python
def generate(model, prompt, max_tokens, temperature=1.0):
    for _ in range(max_tokens):
        logits = model(prompt)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        prompt = torch.cat([prompt, next_token], dim=1)
    return prompt
```

Temperature controls randomness:
- Low (0.1): Predictable, repetitive
- High (1.5): Creative, sometimes nonsense
- Medium (0.7): Usually what you want

---

## GPT vs BERT

| | GPT | BERT |
|---|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (left only) | Bidirectional |
| Training | Next token prediction | Masked language model |
| Good at | Generation | Understanding |

Use BERT for classification. Use GPT for generation. Or just use GPT for everything (modern LLMs are surprisingly good at classification too).

---

## Code

The causal attention is simple - just mask future positions:

```python
class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        # Lower triangular = can only see past
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        return attn @ V
```

---

## Papers

- [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- [GPT-3](https://arxiv.org/abs/2005.14165) (2020)
- [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) - The RLHF paper

---

## Try it

The notebook builds GPT from scratch, trains a character-level language model, and lets you generate text with different temperatures.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb)
