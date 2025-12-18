# GPT: Generative Pre-trained Transformer

[← Back to Architectures](../README.md) | [← Previous: BERT](../02_bert/README.md) | [Next: ViT →](../04_vision_transformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb)

---

![Architecture](architecture.png)

## What is it?

**GPT** is a **decoder-only** transformer that generates text autoregressively (one token at a time). It's the architecture behind ChatGPT, Claude, LLaMA, and most modern LLMs.

## The Key Insight

Train on massive text data with one simple objective: **predict the next token**.

```
Input:  "The cat sat on the"
Output: "mat" (or whatever comes next)
```

Scale this up with more data and parameters → emergent abilities.

## Architecture

```
Token1 Token2 Token3 ... TokenN
              ↓
    Token Embedding + Position Embedding
              ↓
    Decoder Block × N (with causal mask)
              ↓
    Linear → Softmax
              ↓
    Next Token Probabilities
```

### Decoder Block
1. **Masked Self-Attention** - Can only see past tokens
2. **Layer Norm** (Pre-LN in GPT-2+)
3. **Feed Forward Network**
4. **Layer Norm**
5. **Residual connections**

## The Causal Mask

The key difference from BERT: **causal masking**.

```
Query\Key  T1  T2  T3  T4
   T1      ✓   ✗   ✗   ✗
   T2      ✓   ✓   ✗   ✗
   T3      ✓   ✓   ✓   ✗
   T4      ✓   ✓   ✓   ✓
```

Each token can only attend to itself and previous tokens.

## The Math

### Autoregressive Language Modeling

```
P(x) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
     = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)
```

### Loss Function

```
L = -∑ log P(xₜ | x₁, ..., xₜ₋₁)
```

Just cross-entropy on next token prediction.

### Causal Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k + M) × V

where M[i,j] = 0 if j ≤ i, else -∞
```

## GPT Evolution

| Model | Year | Parameters | Context | Training Data |
|-------|------|------------|---------|---------------|
| GPT-1 | 2018 | 117M | 512 | BookCorpus |
| GPT-2 | 2019 | 1.5B | 1024 | WebText (40GB) |
| GPT-3 | 2020 | 175B | 2048 | 300B tokens |
| GPT-4 | 2023 | ~1.8T (MoE) | 8K-128K | Unknown |

## Code Highlights

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        # ... projections ...
        
        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        
        # Attention with causal mask
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        return attn @ V

# Generation
@torch.no_grad()
def generate(model, prompt, max_tokens, temperature=1.0):
    for _ in range(max_tokens):
        logits = model(prompt)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        prompt = torch.cat([prompt, next_token], dim=1)
    return prompt
```

## Key Innovations Over Versions

### GPT-1 (2018)
- Showed pre-training + fine-tuning works

### GPT-2 (2019)
- Pre-LN (LayerNorm before attention)
- Larger scale, zero-shot capabilities

### GPT-3 (2020)
- In-context learning
- Few-shot prompting
- Emergent abilities at scale

### GPT-4 (2023)
- Likely MoE architecture
- Multimodal (vision)
- RLHF alignment

## What GPT is Good For

✅ **Generation tasks**:
- Text completion
- Chatbots
- Code generation
- Creative writing
- Summarization (with prompting)

❌ **Less ideal for**:
- Pure classification (use BERT)
- When you need bidirectional context

## GPT vs BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (unidirectional) | Bidirectional |
| Pre-training | Next token prediction | MLM + NSP |
| Best for | Generation | Understanding |

## Key Papers

- [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- [GPT-3](https://arxiv.org/abs/2005.14165) (2020)
- [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) - RLHF

## Try It

Run the notebook to:
1. Build GPT from scratch
2. Train a character-level language model
3. Generate text with different temperatures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/03_gpt/demo.ipynb)

