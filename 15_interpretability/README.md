# Problem 15: Poor Interpretability

## Problem

Attention weights are not true explanations. Hard to debug and understand.

While attention weights are often visualized as "explanations," research shows they don't reliably indicate which inputs influenced the output. This makes debugging and understanding model behavior difficult.

## Solutions

| Solution | Description |
|----------|-------------|
| **Attention Rollout** | Aggregate attention across layers for better interpretation |
| **Probing Models** | Train classifiers on intermediate representations |
| **Saliency Methods** | Gradient-based methods to identify important inputs |
| **Mechanistic Interpretability** | Reverse-engineer model circuits and computations |

## References

- [Attention is not Explanation](https://arxiv.org/abs/1902.10186) - Attention is not Explanation
- [BertViz](https://arxiv.org/abs/1904.02679) - A Tool for Visualizing Attention in BERT
- [Mechanistic Interpretability](https://transformer-circuits.pub/) - Transformer Circuits Thread

