# Problem 6: Data-Hungry Architecture

[← Back to Main](../README.md) | [← Previous](../05_local_bias/README.md) | [Next →](../07_memory_footprint/README.md)

---

![Problem 6](./problem.png)

## What's the Problem?

The original ViT paper said it clearly: train on ImageNet alone, and ViT loses to ResNet. Train on JFT-300M (300 million images), and ViT wins.

Transformers are incredibly flexible — they make very few assumptions about the data. That flexibility is powerful, but it means they need to learn everything from scratch. CNNs "know" about local patterns; transformers have to discover this themselves.

More parameters + fewer assumptions = need more data.

## Why Does This Happen?

It comes down to **inductive bias** (or lack thereof):

- **CNNs**: "I bet nearby pixels matter more" (baked in via convolutions)
- **RNNs**: "I bet sequence order matters" (baked in via recurrence)
- **Transformers**: "I make no assumptions, show me the data"

Fewer assumptions means more flexibility, but also means more examples needed to figure out what actually matters.

## The Numbers

| Model | Typical Data Requirement |
|-------|-------------------------|
| CNN | ~1M images works okay |
| Transformer | ~100M+ images to shine |
| GPT-3 | 300B+ tokens |
| LLaMA | 1-2T tokens |

## How Do We Fix It?

| Approach | The Idea |
|----------|----------|
| **Self-Supervised Pretraining** | BERT, GPT — learn from massive unlabeled data first |
| **Transfer Learning** | Train once on huge data, fine-tune for specific tasks |
| **Data Augmentation** | Artificially expand your dataset |
| **Knowledge Distillation** | Small model learns from big model's "knowledge" |

## The Pretraining Revolution

This is why the pretrain-then-finetune paradigm took over:
1. Someone (OpenAI, Google, Meta) trains on internet-scale data
2. You download the model and fine-tune on your small dataset
3. The pretrained knowledge transfers

You don't need billions of examples if someone else already did that work.

## Learn More

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — ViT paper
- [BERT](https://arxiv.org/abs/1810.04805) — Pretraining for NLP

---

[← Back to Main](../README.md) | [← Previous](../05_local_bias/README.md) | [Next →](../07_memory_footprint/README.md)
