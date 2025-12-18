# Problem 9: Poor Length Generalization

[← Back to Main](../README.md) | [← Previous](../08_compute_cost/README.md) | [Next →](../10_training_instability/README.md)

---

![Problem 9](./problem.png)

## What's the Problem?

Train a model on sequences up to 2048 tokens. Test it on 4096 tokens. Watch it fall apart.

This isn't just "worse performance" — models can completely break down when they see positions they've never encountered during training. The output becomes nonsensical.

This matters because:
- You can't always predict input length at deployment
- Users want to paste long documents
- Code files vary wildly in length

## Why Does This Happen?

The culprit is usually positional encoding. If you trained with learned position embeddings for positions 1-2048, what happens at position 3000? That embedding was never learned.

Even sinusoidal encodings, which theoretically exist for all positions, behave differently at unseen positions because the model never learned how to use them.

## The Core Issue

Position encodings effectively tell the model "here's what position 500 looks like." But if the model only saw positions 1-2048 during training, it has no idea what to do with position 3000's encoding.

## How Do We Fix It?

| Approach | How It Helps |
|----------|--------------|
| **Relative Position Encoding** | Only encodes distances, not absolute positions — distance of 10 is the same whether it's pos 5→15 or 5000→5010 |
| **RoPE Scaling** | Interpolate positions — treat position 4000 as if it were position 2000 |
| **ALiBi** | Linear bias based on distance — naturally extrapolates |
| **YaRN** | Smart interpolation that preserves high-frequency information |

## ALiBi: The Simple Solution

Instead of adding position embeddings, ALiBi just subtracts a penalty from attention scores based on distance:
```
attention_score = q·k - m × distance
```

No learned parameters for position at all. Distance of 100 works the same whether you're at the start or end of a 1M token sequence.

## Learn More

- [ALiBi Paper](https://arxiv.org/abs/2108.12409) — Train short, test long
- [YaRN](https://arxiv.org/abs/2309.00071) — Better RoPE scaling

---

[← Back to Main](../README.md) | [← Previous](../08_compute_cost/README.md) | [Next →](../10_training_instability/README.md)
