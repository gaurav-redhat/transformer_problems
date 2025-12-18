# Vision Transformer (ViT)

[← Back](../README.md) | [← Prev: GPT](../03_gpt/README.md) | [Next: Transformer-XL →](../05_transformer_xl/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb)

---

![Architecture](architecture.png)

For decades, CNNs were the only game in town for images. Then ViT showed up in 2020 and said: "What if we just treat image patches as tokens?" Turns out, with enough data, that works better.

The title says it all: *"An Image is Worth 16x16 Words"*

---

## The idea

Take an image, chop it into patches, flatten each patch, and feed them into a standard transformer:

```
224×224 image
    ↓
Split into 16×16 patches → 196 patches
    ↓
Flatten each patch → 196 tokens of dimension 768
    ↓
Add [CLS] token + position embeddings
    ↓
Transformer encoder
    ↓
[CLS] output → classification
```

No convolutions. No pooling. Just patches and attention.

---

## Why it works

CNNs have built-in assumptions about images:
- Local features matter (convolution)
- Translation invariance (same filter everywhere)
- Hierarchical structure (pool → bigger receptive field)

ViT throws all that away and learns everything from scratch. The catch: you need **way more data**. The paper pretrains on JFT-300M (300 million images). With less data, CNNs still win.

But at scale, pure attention beats the inductive biases.

---

## Patch embedding

The "tokenization" for images:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        # Conv2d with kernel=stride=patch_size acts as patch extraction
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
    
    def forward(self, x):
        # (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 196, 768)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

Each 16×16×3 patch becomes one 768-dim token. A 224×224 image becomes 196 tokens.

---

## Position embeddings

Transformers don't know order. For images, we add learned 1D position embeddings (surprisingly, 1D works fine even though images are 2D):

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))  # 196 patches + [CLS]
```

The model learns to interpret these positions spatially.

---

## The [CLS] token

Like BERT, ViT prepends a learnable [CLS] token. After the transformer, its output is used for classification. The intuition: it aggregates information from all patches through attention.

---

## Model sizes

| Model | Layers | Hidden | Heads | Params | ImageNet |
|-------|--------|--------|-------|--------|----------|
| ViT-B/16 | 12 | 768 | 12 | 86M | 77.9% |
| ViT-L/16 | 24 | 1024 | 16 | 307M | 79.7% |
| ViT-H/14 | 32 | 1280 | 16 | 632M | 81.1% |

The `/16` or `/14` is the patch size. Smaller patches = more tokens = more compute = often better.

---

## What we learned

1. **Attention can replace convolution** - Given enough data, inductive bias is unnecessary
2. **Scale matters** - ViT underperforms CNNs on ImageNet alone, but wins when pretrained on larger datasets
3. **Position embeddings are flexible** - 1D learned positions work for 2D images
4. **Attention patterns are interpretable** - You can visualize what the model "looks at"

---

## The ViT family

| Model | Year | Key idea |
|-------|------|----------|
| ViT | 2020 | Original |
| DeiT | 2021 | Train without JFT, distillation |
| Swin | 2021 | Hierarchical, shifted windows |
| BEiT | 2021 | BERT-style pretraining for images |
| MAE | 2022 | Masked autoencoder (mask 75% of patches) |

Swin is popular in practice because it's hierarchical (good for detection/segmentation). MAE showed you can pretrain ViT with 75% of patches masked - very data efficient.

---

## Code

The full ViT forward pass:

```python
class ViT(nn.Module):
    def __init__(self, ...):
        self.patch_embed = PatchEmbedding(...)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.transformer = TransformerEncoder(...)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)                              # (B, 196, 768)
        cls = self.cls_token.expand(x.size(0), -1, -1)       # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)                       # (B, 197, 768)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])                            # [CLS] → class
```

---

## Papers

- [ViT](https://arxiv.org/abs/2010.11929) (2020) - Original
- [DeiT](https://arxiv.org/abs/2012.12877) (2021) - Data-efficient training
- [Swin Transformer](https://arxiv.org/abs/2103.14030) (2021) - Hierarchical
- [MAE](https://arxiv.org/abs/2111.06377) (2022) - Masked pretraining

---

## Try it

The notebook builds ViT from scratch, visualizes patch splitting, trains on CIFAR-10, and shows attention maps.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/transformer_problems/blob/main/transformer_architectures/04_vision_transformer/demo.ipynb)
