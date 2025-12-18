# Problem 6: Data-Hungry Architecture

## Problem

Requires massive datasets to generalize well. Small data = poor performance.

Transformers lack strong inductive biases, making them extremely data-hungry. They typically require orders of magnitude more training data than CNNs or RNNs to achieve good performance.

## Solutions

| Solution | Description |
|----------|-------------|
| **Self-Supervised Pretraining** | Pretrain on large unlabeled corpora (BERT, GPT) |
| **Transfer Learning** | Fine-tune pretrained models on downstream tasks |
| **Fine-Tuning** | Adapt large pretrained models to specific domains |
| **Knowledge Distillation** | Transfer knowledge from large models to smaller ones |

## References

- [BERT](https://arxiv.org/abs/1810.04805) - Pre-training of Deep Bidirectional Transformers
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - Improving Language Understanding by Generative Pre-Training
- [DistilBERT](https://arxiv.org/abs/1910.01108) - A Distilled Version of BERT

