# Problem 10: Training Instability

[← Back to Main](../README.md) | [← Previous](../09_length_generalization/README.md) | [Next →](../11_attention_smoothing/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/10_training_instability/demo.ipynb)

---

![Problem 10](./problem.png)

## What's the Problem?

You're 3 weeks into training a 70B model. Everything looks good. Then suddenly — loss spikes to infinity. Training diverges. Three weeks of compute, wasted.

Training instability gets worse with scale. The bigger your model, the more likely you'll hit:
- Loss spikes (sudden jumps in loss)
- Gradient explosion (gradients become NaN)
- Training divergence (loss stops decreasing)

This is why training frontier models requires teams of engineers monitoring 24/7.

## Why Does This Happen?

Several culprits:

1. **Gradient explosion/vanishing**: In deep networks, gradients can compound through layers — either exploding or shrinking to zero

2. **Attention entropy collapse**: Attention can become too "sharp" (all weight on one token) or too "flat" (uniform attention)

3. **Bad data batches**: A single weird batch can throw off the entire training run

4. **Learning rate sensitivity**: Too high and you diverge, too low and you don't learn

## How Do We Fix It?

| Technique | What It Does |
|-----------|--------------|
| **Pre-LayerNorm** | Normalize *before* attention/FFN, not after — much more stable |
| **Learning Rate Warmup** | Start with tiny LR, gradually increase — prevents early divergence |
| **Gradient Clipping** | Cap gradient magnitude — prevents explosion |
| **AdamW** | Adam + proper weight decay — more stable than vanilla Adam |
| **BF16 Training** | Better numerical range than FP16, fewer overflows |

## Pre-LN vs Post-LN

Original transformer (Post-LN):
```
output = LayerNorm(x + Attention(x))
```

Modern transformers (Pre-LN):
```
output = x + Attention(LayerNorm(x))
```

This small change makes training dramatically more stable for deep models.

## The Real World

Big labs have dedicated infrastructure for this:
- Checkpoint frequently
- Monitor gradient norms
- Automatic rollback on loss spikes
- Learning rate scheduling based on loss

Training 100B+ models is as much engineering as it is research.

## Learn More

- [Pre-LN Paper](https://arxiv.org/abs/2002.04745) — Why it works
- [PaLM Technical Report](https://arxiv.org/abs/2204.02311) — Google's training notes

---

[← Back to Main](../README.md) | [← Previous](../09_length_generalization/README.md) | [Next →](../11_attention_smoothing/README.md)
