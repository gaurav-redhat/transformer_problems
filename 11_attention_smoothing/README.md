# Problem 11: Attention Over-Smoothing

[← Back to Main](../README.md) | [← Previous](../10_training_instability/README.md) | [Next →](../12_no_recurrence/README.md)

---

![Problem 11](./problem.png)

## What's the Problem?

Stack enough transformer layers and something weird happens: all your token representations start looking the same. Token 1, token 50, token 500 — by layer 24, they're nearly identical vectors.

This is called **over-smoothing** or **rank collapse**. The repeated attention operations act like a low-pass filter, averaging out the distinctive features of each token.

The result? Your deep model might not be much better than a shallow one because the extra layers aren't adding useful information.

## Why Does This Happen?

Attention computes weighted averages of value vectors. When you stack this operation:
- Layer 1: Each token is a mix of its neighbors
- Layer 5: Each token is a mix of mixes
- Layer 20: Everyone is mixed with everyone — everything converges to the mean

It's like repeatedly averaging colors — eventually everything becomes gray.

## Visualizing the Problem

Early layers:
```
Token 1: [0.8, 0.2, -0.5, ...]  ← distinct
Token 2: [-0.3, 0.9, 0.1, ...]  ← distinct
Token 3: [0.1, -0.4, 0.7, ...]  ← distinct
```

Deep layers:
```
Token 1: [0.2, 0.3, 0.1, ...]   ← similar
Token 2: [0.2, 0.3, 0.1, ...]   ← similar
Token 3: [0.2, 0.2, 0.1, ...]   ← similar
```

## How Do We Fix It?

| Technique | What It Does |
|-----------|--------------|
| **Residual Connections** | Add input to output — preserves original information |
| **Residual Scaling** | Scale down attention output to prevent dominating residual |
| **DropHead** | Randomly drop attention heads during training — prevents over-reliance |
| **Skip Connections** | Connect early layers directly to later layers |
| **Attention Temperature** | Control sharpness of attention distribution |

## The Residual Connection is Key

The `x + Attention(x)` residual connection is crucial. Without it, over-smoothing would happen much faster. The residual lets each layer make small modifications rather than complete replacements.

DeepNet (1000+ layers) scales residuals differently in different parts to maintain signal through extreme depth.

## Learn More

- [DeepNet](https://arxiv.org/abs/2203.00555) — Training transformers with 1000+ layers
- [Over-smoothing in GNNs](https://arxiv.org/abs/1801.07606) — Same problem in graphs

---

[← Back to Main](../README.md) | [← Previous](../10_training_instability/README.md) | [Next →](../12_no_recurrence/README.md)
