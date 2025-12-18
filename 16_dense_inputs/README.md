# Problem 16: Inefficient for Dense Inputs

## Problem

Flattening images/videos creates huge token counts. Vision = many patches.

When applying Transformers to images or videos, the input must be flattened into sequences. A 224×224 image with 16×16 patches creates 196 tokens; videos create orders of magnitude more.

## Solutions

| Solution | Description |
|----------|-------------|
| **Patch Embedding** | Embed image patches as tokens with optimal patch sizes |
| **Hierarchical Vision Transformers** | Multi-scale processing with progressive downsampling |
| **Tubelet Embedding** | Embed 3D video cubes for efficient video processing |
| **Swin** | Shifted window attention for efficient hierarchical processing |

## References

- [ViT](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words
- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Hierarchical Vision Transformer
- [Video Swin](https://arxiv.org/abs/2106.13230) - Video Swin Transformer

