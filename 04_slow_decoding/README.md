# Problem 4: Slow Autoregressive Decoding

[← Back to Main](../README.md) | [← Previous](../03_fixed_context/README.md) | [Next →](../05_local_bias/README.md)

---

![Problem 4](./problem.png)

## What's the Problem?

When GPT generates text, it works like this:
1. Generate "The"
2. Feed "The" back in, generate "cat"
3. Feed "The cat" back in, generate "sat"
4. And so on...

Want to generate 100 tokens? That's 100 separate forward passes through the model. You can't parallelize this because each token depends on all the previous ones.

This is why ChatGPT streams responses word-by-word — it literally can't give you the full answer instantly.

## Why Does This Happen?

Language models are trained to predict `P(next_token | previous_tokens)`. This autoregressive setup means:

- Token 5 needs tokens 1-4 to exist first
- Token 6 needs tokens 1-5
- And so on...

There's an inherent sequential dependency. Training can be parallelized (we know all the "right" answers), but generation cannot.

## How Do We Fix It?

| Approach | The Idea |
|----------|----------|
| **KV Cache** | Don't recompute attention for old tokens — cache their K and V values |
| **Speculative Decoding** | Use a tiny model to draft several tokens, big model just verifies (often faster!) |
| **Non-Autoregressive** | Generate all tokens at once, then refine (quality trade-off) |
| **Distillation** | Train smaller models that run faster |

## KV Cache: The Standard Fix

Without KV cache, generating token 100 means recomputing attention for tokens 1-99 from scratch. With caching:
- Token 1-99's K,V values are stored
- Token 100 only computes its own K,V and attends to the cache

This doesn't reduce the number of forward passes, but makes each one much cheaper.

## Learn More

- [Speculative Decoding](https://arxiv.org/abs/2211.17192) — Draft and verify approach
- [Medusa](https://arxiv.org/abs/2401.10774) — Multiple decoding heads

---

[← Back to Main](../README.md) | [← Previous](../03_fixed_context/README.md) | [Next →](../05_local_bias/README.md)
