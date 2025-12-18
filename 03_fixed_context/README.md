# Problem 3: Fixed Context Window

## Problem

Cannot remember information beyond maximum sequence length. Past is forgotten.

Transformers have a fixed context window (e.g., 2048, 4096 tokens). Any information beyond this window is completely inaccessible, leading to loss of important context in long documents or conversations.

## Solutions

| Solution | Description |
|----------|-------------|
| **Transformer-XL** | Segment-level recurrence mechanism to extend context |
| **Retrieval-Augmented Generation (RAG)** | Retrieve relevant documents from external knowledge base |
| **External Memory** | Maintain external memory banks (like Neural Turing Machines) |
| **Mamba** | State Space Models with selective state spaces for infinite context |

## References

- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Attentive Language Models Beyond a Fixed-Length Context
- [RAG](https://arxiv.org/abs/2005.11401) - Retrieval-Augmented Generation
- [Mamba](https://arxiv.org/abs/2312.00752) - Linear-Time Sequence Modeling with Selective State Spaces

