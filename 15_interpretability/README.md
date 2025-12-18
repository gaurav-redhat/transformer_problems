# Problem 15: Poor Interpretability

[← Back to Main](../README.md) | [← Previous](../14_noise_sensitivity/README.md) | [Next →](../16_dense_inputs/README.md)

---

![Problem 15](./problem.png)

## What's the Problem?

Your model says "The answer is 42." Why? You look at the attention weights. They highlight "meaning" and "life." Great, but... is that actually why it gave that answer? Probably not.

Transformers are black boxes. We can see what goes in and what comes out, but the reasoning in between is opaque. This matters for:
- Debugging wrong answers
- Building trust in high-stakes applications
- Understanding model failures
- Regulatory compliance

## The Attention Fallacy

People love visualizing attention weights as "explanations." But research shows:
- You can permute attention weights and get the same output
- High attention doesn't mean high influence on output
- Different heads learn different (often redundant) patterns

Attention tells you what the model looked at, not what mattered for the decision.

## Why This Is Hard

Modern LLMs have:
- Billions of parameters
- Hundreds of layers
- Thousands of attention heads
- Complex non-linear interactions

There's no simple "this input caused that output" story. The computation is distributed across millions of neurons.

## How Do We Try to Fix It?

| Method | What It Does |
|--------|--------------|
| **Attention Rollout** | Aggregate attention across layers for better approximation |
| **Probing Classifiers** | Train small models to extract info from hidden states |
| **Saliency Methods** | Use gradients to find important inputs |
| **Mechanistic Interpretability** | Reverse-engineer specific circuits in the model |
| **Chain-of-Thought** | Make the model show its reasoning steps |

## Mechanistic Interpretability

This is the hot new approach. Instead of treating the model as a black box, researchers:
1. Find individual neurons/circuits
2. Understand what they compute
3. Build up understanding piece by piece

Anthropic found "induction heads" that copy patterns. Others found "knowledge neurons" that store facts. Still early days, but promising.

## The Practical Reality

For most applications, we rely on:
- Extensive testing
- Prompt engineering to elicit reasoning
- Guardrails and filters
- Human oversight

True interpretability remains an open problem.

## Learn More

- [Attention is not Explanation](https://arxiv.org/abs/1902.10186)
- [Anthropic's Interpretability Work](https://transformer-circuits.pub/)

---

[← Back to Main](../README.md) | [← Previous](../14_noise_sensitivity/README.md) | [Next →](../16_dense_inputs/README.md)
