<p align="center">
  <img src="https://img.shields.io/badge/Architecture-GPT-EA4335?style=for-the-badge" alt="GPT"/>
  <img src="https://img.shields.io/badge/Type-Decoder--Only-lightgrey?style=for-the-badge" alt="Type"/>
  <img src="https://img.shields.io/badge/Direction-Autoregressive-blue?style=for-the-badge" alt="Direction"/>
</p>

<h1 align="center">03. GPT</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../02_bert/README.md">← Prev</a> •
  <a href="../04_vision_transformer/README.md">Next: ViT →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

This is it. The architecture behind ChatGPT, Claude, LLaMA, Mistral — pretty much every chatbot you've used.

> *The idea is absurdly simple: **predict the next token**.*

```
Input:  "The cat sat on the"
Output: "mat"
```

Scale this up with more data and parameters → emergent abilities.

---

## 🏗️ Architecture

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

---

## 🎭 The Causal Mask

> *Each token can only see itself and past tokens*

```
Position:    1  2  3  4
Token 1:     ✓  ✗  ✗  ✗
Token 2:     ✓  ✓  ✗  ✗
Token 3:     ✓  ✓  ✓  ✗
Token 4:     ✓  ✓  ✓  ✓
```

This prevents cheating during training and enables generation.

---

## 📈 The Evolution

| Model | Year | Params | Context | Milestone |
|-------|:----:|:------:|:-------:|-----------|
| GPT-1 | 2018 | 117M | 512 | Proved the concept |
| GPT-2 | 2019 | 1.5B | 1024 | Zero-shot abilities |
| GPT-3 | 2020 | 175B | 2048 | In-context learning |
| GPT-4 | 2023 | ~1.8T? | 128K | Multimodal, RLHF |

---

## ➗ The Math

### Autoregressive Probability

```
P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... 
```

### Loss Function

```
L = -∑ log P(xₜ | x₁, ..., xₜ₋₁)
```

Just cross-entropy on next token prediction.

---

## 🆚 GPT vs BERT

| | GPT | BERT |
|:-:|:---:|:----:|
| **Architecture** | Decoder-only | Encoder-only |
| **Attention** | Causal (→) | Bidirectional (↔) |
| **Training** | Next token | Masked LM |
| **Best for** | Generation | Understanding |

---

## 💻 Code

```python
class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        # Lower triangular = can only see past
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        attn = (Q @ K.T) / math.sqrt(d_k)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        return F.softmax(attn, dim=-1) @ V

# Generation
def generate(model, prompt, max_tokens, temperature=1.0):
    for _ in range(max_tokens):
        logits = model(prompt)[:, -1, :] / temperature
        next_token = torch.multinomial(F.softmax(logits, -1), 1)
        prompt = torch.cat([prompt, next_token], dim=1)
    return prompt
```

---

## 🌡️ Temperature

| Value | Effect |
|:-----:|--------|
| **0.1** | Predictable, repetitive |
| **0.7** | Balanced (usually best) |
| **1.5** | Creative, sometimes nonsense |

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 2018 |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Build GPT from scratch • Train char-level LM • Generate text</sub>
</p>
