# Vision Transformer (ViT)

[← Back to Architectures](../README.md) | [← Previous: GPT](../03_gpt/README.md) | [Next: Transformer-XL →](../05_transformer_xl/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb)

---

![Architecture](architecture.png)

## What is it?

**Vision Transformer (ViT)** applies the transformer architecture directly to images. Instead of words, it treats **image patches as tokens**. The 2020 paper showed that with enough data, pure transformers can match or beat CNNs on image tasks.

## The Key Insight

> "An image is worth 16x16 words"

Split the image into patches, flatten them, and treat them like a sequence of tokens.

```
224×224 image → 14×14 = 196 patches (each 16×16)
                     → 196 tokens for transformer
```

## Architecture

```
224×224×3 Image
      ↓
Split into 16×16 patches (196 patches)
      ↓
Flatten + Linear Projection → Patch Embeddings
      ↓
[CLS] + Patch Embeddings + Position Embeddings
      ↓
Transformer Encoder × L
      ↓
[CLS] output → MLP Head → Classification
```

### Patch Embedding

```python
# A Conv2d with kernel_size=patch_size, stride=patch_size acts as patch embedding
patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
# (B, 3, 224, 224) → (B, embed_dim, 14, 14) → (B, 196, embed_dim)
```

### [CLS] Token

A learnable token prepended to the sequence. Its output is used for classification (like BERT).

### Position Embeddings

Learned 1D position embeddings (not 2D!) - surprisingly works well.

## The Math

### Patch Embedding

```
x_p ∈ R^(N × (P² · C))

where:
- N = (H × W) / P² = number of patches
- P = patch size (typically 16)
- C = channels (3 for RGB)
```

### Full Forward Pass

```
z₀ = [x_class; x_p¹E; x_p²E; ...; x_pᴺE] + E_pos

zₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁        (attention)
zₗ = MLP(LN(zₗ)) + zₗ             (feed forward)

y = LN(z_L⁰)                       (CLS token output)
```

## Model Sizes

| Model | Layers | Hidden | Heads | Params | ImageNet Top-1 |
|-------|--------|--------|-------|--------|----------------|
| ViT-B/16 | 12 | 768 | 12 | 86M | 77.9% |
| ViT-L/16 | 24 | 1024 | 16 | 307M | 79.7% |
| ViT-H/14 | 32 | 1280 | 16 | 632M | 81.1% |

Note: `/16` means patch size 16, `/14` means patch size 14.

## Code Highlights

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2  # 196
        
        # Conv2d as patch embedding
        self.proj = nn.Conv2d(3, embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)
    
    def forward(self, x):
        # (B, 3, 224, 224) → (B, embed_dim, 14, 14)
        x = self.proj(x)
        # → (B, embed_dim, 196) → (B, 196, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViT(nn.Module):
    def __init__(self, ...):
        self.patch_embed = PatchEmbedding(...)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.transformer = TransformerEncoder(...)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])  # CLS token
```

## Key Findings from the Paper

1. **ViT needs lots of data** - Pre-trained on JFT-300M (300M images), then fine-tuned
2. **Beats CNNs at scale** - With enough data, pure attention > convolutions
3. **Position embeddings work** - 1D learned positions are fine
4. **Attention patterns are interpretable** - Can visualize what ViT "looks at"

## ViT vs CNN

| Aspect | ViT | CNN |
|--------|-----|-----|
| Inductive bias | Minimal (learns from data) | Strong (locality, translation equivariance) |
| Data efficiency | Poor (needs lots of data) | Good (works with less data) |
| Scalability | Excellent | Good |
| Global context | From layer 1 | Only in deep layers |

## ViT Family

| Model | Year | Key Innovation |
|-------|------|----------------|
| ViT | 2020 | Original |
| DeiT | 2021 | Data-efficient training, distillation |
| Swin | 2021 | Hierarchical, shifted windows |
| BEiT | 2021 | BERT-style pre-training for images |
| MAE | 2022 | Masked autoencoder pre-training |
| ViT-22B | 2023 | Largest vision model (22B params) |

## Key Papers

- [ViT](https://arxiv.org/abs/2010.11929) (2020) - Original
- [DeiT](https://arxiv.org/abs/2012.12877) (2021) - Training without JFT
- [Swin Transformer](https://arxiv.org/abs/2103.14030) (2021) - Hierarchical ViT
- [MAE](https://arxiv.org/abs/2111.06377) (2022) - Masked autoencoder

## Try It

Run the notebook to:
1. Build ViT from scratch
2. Visualize patch splitting
3. Train on CIFAR-10
4. See attention maps

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb)

