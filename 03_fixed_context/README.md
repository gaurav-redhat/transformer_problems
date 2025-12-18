# Problem 3: Fixed Context Window

[← Back to Main](../README.md) | [← Previous](../02_positional_awareness/README.md) | [Next →](../04_slow_decoding/README.md)

---

![Problem 3](./problem.png)

## What's the Problem?

Imagine reading a 500-page book, but you can only remember the last 10 pages at any time. That's basically what happens with transformers.

GPT-2 could only "see" 1024 tokens. GPT-4 pushed this to 128K, but there's still a hard limit. If your conversation or document exceeds it? The beginning just... disappears.

This is a real problem for:
- Long documents and books
- Extended conversations (chatbots forgetting earlier context)
- Code repositories with thousands of files

## Why Does This Happen?

Two reasons:

1. **Memory**: That N² attention matrix grows fast. 128K tokens means a 16 billion element attention matrix.

2. **Position Encodings**: Most PE methods are trained for specific lengths. Ask the model about position 200,000 when it only learned up to 100,000? Good luck.

## How Do We Fix It?

| Approach | The Idea |
|----------|----------|
| **Transformer-XL** | Process in chunks, but pass hidden states between chunks (recurrence is back!) |
| **RAG** | Don't remember everything — just retrieve relevant chunks when needed |
| **External Memory** | Store important stuff in a separate memory bank, query it as needed |
| **Mamba/SSMs** | Ditch attention entirely, use state space models with O(N) scaling |

## The Real-World Situation

Modern solutions often combine approaches:
- Use RAG to fetch relevant context
- Extend context windows with RoPE scaling
- Compress old context into summaries

No single solution is perfect yet. This is still an active research area.

## Learn More

- [Transformer-XL](https://arxiv.org/abs/1901.02860) — Segment-level recurrence
- [Mamba](https://arxiv.org/abs/2312.00752) — The SSM alternative

---

[← Back to Main](../README.md) | [← Previous](../02_positional_awareness/README.md) | [Next →](../04_slow_decoding/README.md)
