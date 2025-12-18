<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Switch_Transformer-E91E63?style=for-the-badge" alt="Switch"/>
  <img src="https://img.shields.io/badge/Method-Mixture_of_Experts-purple?style=for-the-badge" alt="MoE"/>
  <img src="https://img.shields.io/badge/Scale-1.6_Trillion-gold?style=for-the-badge" alt="Scale"/>
</p>

<h1 align="center">10. Switch Transformer</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../09_longformer/README.md">← Prev</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="architecture.png" alt="Architecture" width="90%"/>
</p>

---

## 💡 The Idea

> *What if you could have a trillion parameters but only use a fraction for each input?*

**Mixture of Experts (MoE):** Each token gets routed to one of many "expert" networks.

```
Standard:  Token → [Large FFN] → Output

Switch:    Token → [Router] → Expert 3 → Output
           Token → [Router] → Expert 7 → Output
```

**More capacity, same compute.**

---

## 📊 The Scale

| Model | Total Params | Active Params | Experts |
|-------|:------------:|:-------------:|:-------:|
| Switch-Base | 7B | 335M | 128 |
| Switch-Large | 26B | 783M | 128 |
| **Switch-C** | **1.6T** | **12.8B** | 2048 |

> 🤯 **1.6 trillion parameters** — but each token only uses 12.8B!

---

## 🔀 How Routing Works

```python
router_logits = W_router @ token    # → (n_experts,)
expert_weights = softmax(router_logits)
chosen_expert = argmax(expert_weights)  # Top-1 routing
```

Simple. Token goes in, expert index comes out.

---

## ⚖️ Load Balancing

**Problem:** All tokens might go to one expert ("expert collapse")

**Solution:** Auxiliary loss for balanced routing

```
L_aux = α × N × Σᵢ (fraction_i × avg_prob_i)
```

This pushes toward uniform distribution across experts.

---

## 🏗️ Architecture

```
Token Embeddings
       ↓
    Attention (standard)
       ↓
    ┌────────────────────┐
    │      ROUTER        │
    │  softmax → argmax  │
    └────────────────────┘
         /  |  |  \
        ↓   ↓  ↓   ↓
     E1  E2  E3  E4  ...
        \   |  |  /
         ↓  ↓  ↓ ↓
    Combined Output
       ↓
    Next Layer
```

---

## 💻 Code

```python
class SwitchRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        self.router = nn.Linear(d_model, n_experts)
    
    def forward(self, x):
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        expert_idx = probs.argmax(dim=-1)      # Top-1
        expert_weight = probs.max(dim=-1).values
        return expert_idx, expert_weight

class SwitchFFN(nn.Module):
    def __init__(self, d_model, d_ff, n_experts):
        self.router = SwitchRouter(d_model, n_experts)
        self.experts = nn.ModuleList([
            FFN(d_model, d_ff) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        expert_idx, expert_weight = self.router(x)
        output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                output[mask] = expert_weight[mask, None] * expert(x[mask])
        
        return output
```

---

## 🆚 Top-1 vs Top-K

| Method | Experts/Token | Compute |
|--------|:-------------:|:-------:|
| Top-1 (Switch) | 1 | Lowest |
| Top-2 (Mixtral) | 2 | 2× |
| Soft MoE | All | N× |

---

## 👨‍👩‍👧‍👦 The MoE Family

| Model | Year | Routing | Scale |
|-------|:----:|:-------:|:-----:|
| Switch | 2021 | Top-1 | 1.6T |
| GLaM | 2021 | Top-2 | 1.2T |
| **Mixtral** | 2023 | Top-2 | 46B (12B active) |
| **DeepSeek-V3** | 2024 | Fine-grained | 671B (37B active) |

> 💡 *MoE is now standard for frontier models — it's how you scale affordably.*

---

## 🔧 Training Tips

| Tip | Why |
|-----|-----|
| Use bfloat16 | MoE is numerically sensitive |
| Keep router in float32 | Precision matters for routing |
| Start with fewer experts | 8-16 before scaling to hundreds |
| Monitor utilization | Watch for collapse |

---

## 📚 Papers

| Paper | Year |
|-------|:----:|
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 2021 |
| [GLaM](https://arxiv.org/abs/2112.06905) | 2021 |
| [Mixtral](https://arxiv.org/abs/2401.04088) | 2024 |
| [DeepSeek-MoE](https://arxiv.org/abs/2401.06066) | 2024 |

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/10_switch_transformer/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Train_It_Yourself-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <sub>Build Switch layer • Visualize expert assignment • Implement load balancing</sub>
</p>
