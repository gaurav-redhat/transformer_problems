# Switch Transformer: Mixture of Experts

[← Back to Architectures](../README.md) | [← Previous: Longformer](../09_longformer/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb)

---

![Architecture](architecture.png)

## What is it?

**Switch Transformer** (Google, 2021) uses **Mixture of Experts (MoE)** to scale to **trillion parameters** while keeping compute constant. Each token is routed to one of many "expert" FFNs.

Key insight: **More parameters ≠ more compute per token**.

## The Problem

Scaling transformers traditionally means:
- More parameters → More FLOPs → More time/cost
- GPT-3 (175B) takes ~300K GPU hours to train

## The Solution: Sparse Activation

Instead of one large FFN, use N smaller "expert" FFNs:
- Each token is **routed to exactly 1 expert** (top-1 routing)
- N× more parameters, but **same FLOPs per token**

```
Standard:       All tokens → 1 large FFN
Switch:         Each token → 1 of N small FFNs (chosen by router)
```

## Architecture

```
Token Embeddings
       ↓
    Attention (standard)
       ↓
    ┌──────────────────────────────┐
    │          ROUTER              │
    │    (softmax over experts)    │
    └──────────────────────────────┘
           /    |    \    \
          ↓     ↓     ↓    ↓
       Expert Expert Expert Expert
         1     2      3     4
          \    |    /    /
           ↓   ↓   ↓   ↓
    Combined Output (weighted)
       ↓
    Next Layer
```

## The Router

A simple learned linear layer that decides which expert each token goes to:

```python
router_logits = W_router @ token  # (token_dim) → (n_experts)
expert_weights = softmax(router_logits)
chosen_expert = argmax(expert_weights)  # Top-1 routing
```

### Switch vs. Top-K Routing

| Method | Experts per Token | Complexity |
|--------|-------------------|------------|
| Top-1 (Switch) | 1 | Simplest, fastest |
| Top-2 | 2 | More capacity, 2× compute |
| Soft MoE | All (weighted) | Full capacity, N× compute |

Switch uses **top-1** for maximum efficiency.

## The Math

### Router

```
G(x) = softmax(W_r · x)           # Gate probabilities
i = argmax(G(x))                   # Chosen expert index
y = G(x)_i · E_i(x)               # Output (weighted by gate)
```

### Load Balancing Loss

Problem: All tokens might go to one expert ("expert collapse").

Solution: Auxiliary loss to encourage balanced routing:

```
L_aux = α · N · Σᵢ fᵢ · Pᵢ

where:
- fᵢ = fraction of tokens routed to expert i
- Pᵢ = average router probability for expert i
- α = auxiliary loss weight (typically 0.01)
- N = number of experts
```

This encourages: fᵢ ≈ 1/N for all experts.

## Capacity Factor

Experts have limited capacity per batch:

```
Capacity = (batch_size × seq_len / n_experts) × capacity_factor
```

- capacity_factor = 1.0: Balanced, no overflow
- capacity_factor = 1.25: 25% buffer for imbalance
- Overflow tokens: Skip expert (identity shortcut)

## Code Highlights

```python
class SwitchRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        logits = self.router(x)                    # (B, N, n_experts)
        probs = F.softmax(logits, dim=-1)
        
        # Top-1 routing
        expert_idx = probs.argmax(dim=-1)          # (B, N)
        expert_weight = probs.max(dim=-1).values   # (B, N)
        
        return expert_idx, expert_weight, probs

class SwitchFFN(nn.Module):
    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.router = SwitchRouter(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        B, N, D = x.shape
        expert_idx, expert_weight, probs = self.router(x)
        
        output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            # Get tokens for this expert
            mask = (expert_idx == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                output[mask] = expert_weight[mask].unsqueeze(-1) * expert_output
        
        # Compute auxiliary loss
        f = mask.float().mean(dim=(0, 1))  # Fraction per expert
        P = probs.mean(dim=(0, 1))          # Avg prob per expert
        aux_loss = self.n_experts * (f * P).sum()
        
        return output, aux_loss
```

## Scale

| Model | Parameters | Active Params | Experts | Speedup vs Dense |
|-------|------------|---------------|---------|------------------|
| Switch-Base | 7B | 335M | 128 | ~7× |
| Switch-Large | 26B | 783M | 128 | ~7× |
| Switch-C | 1.6T | 12.8B | 2048 | ~4× |

Switch-C has **1.6 trillion parameters** but only uses 12.8B per token!

## Training Tips

1. **Use bfloat16**: MoE is sensitive to precision
2. **Selective precision**: Keep router in float32
3. **Dropout in experts**: Regularize individual experts
4. **Start with fewer experts**: 8 or 16, scale up

## MoE vs Dense Scaling

| Aspect | Dense | MoE |
|--------|-------|-----|
| Parameters | N | N × E |
| FLOPs | N | N (same!) |
| Communication | Low | High (all-to-all) |
| Memory | N | N × E (more) |
| Training stability | Stable | Tricky |

## MoE Family (2021-2025)

| Model | Year | Experts | Parameters |
|-------|------|---------|------------|
| Switch Transformer | 2021 | Top-1 | 1.6T |
| GLaM | 2021 | Top-2 | 1.2T |
| Mixtral 8x7B | 2023 | Top-2 | 46B (12B active) |
| Mixtral 8x22B | 2024 | Top-2 | 141B (39B active) |
| DeepSeek-V2 | 2024 | Fine-grained | 236B |
| DeepSeek-V3 | 2024 | Fine-grained | 671B (37B active) |

## Key Papers

- [Switch Transformer](https://arxiv.org/abs/2101.03961) (2021) - Original
- [GLaM](https://arxiv.org/abs/2112.06905) (2021) - Google's language MoE
- [Mixtral](https://arxiv.org/abs/2401.04088) (2024) - Open MoE that works
- [DeepSeek-MoE](https://arxiv.org/abs/2401.06066) (2024) - Fine-grained experts

## Try It

Run the notebook to:
1. Build a Switch layer with router
2. Visualize expert assignment
3. Implement load balancing loss
4. Compare vs dense FFN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb)

