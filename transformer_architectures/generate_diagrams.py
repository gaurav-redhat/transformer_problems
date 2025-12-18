#!/usr/bin/env python3
"""
Generate detailed architecture diagrams for transformer variants.
Block diagram style with mathematical equations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import os

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edge='#1976D2', fontsize=9, bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, 
            weight=weight, wrap=True)

def draw_arrow(ax, start, end, color='#424242'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_math(ax, x, y, text, fontsize=10):
    """Draw mathematical text."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            style='italic', family='serif')


# ============================================================================
# 1. VANILLA TRANSFORMER
# ============================================================================
def create_vanilla_transformer():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Vanilla Transformer Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # ENCODER SIDE
    ax.text(3.5, 9.5, 'ENCODER', fontsize=14, fontweight='bold', ha='center', color='#1565C0')
    
    # Input Embedding
    draw_box(ax, 2.5, 8.5, 2, 0.6, 'Input\nEmbedding', '#BBDEFB', '#1976D2')
    draw_box(ax, 2.5, 7.7, 2, 0.6, 'Positional\nEncoding', '#C8E6C9', '#388E3C')
    
    # Encoder Block
    draw_box(ax, 1.5, 5.8, 4, 1.5, '', '#E3F2FD', '#1976D2')
    ax.text(3.5, 7.1, 'Encoder Block (Nx)', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 1.8, 6.5, 3.4, 0.5, 'Multi-Head Self-Attention', '#BBDEFB', '#1976D2', fontsize=8)
    draw_box(ax, 1.8, 5.9, 3.4, 0.5, 'Feed Forward Network', '#FFECB3', '#FFA000', fontsize=8)
    
    # Add & Norm labels
    ax.text(5.7, 6.7, 'Add & Norm', fontsize=7, ha='left')
    ax.text(5.7, 6.1, 'Add & Norm', fontsize=7, ha='left')
    
    # Encoder output
    draw_box(ax, 2.5, 4.8, 2, 0.6, 'Encoder\nOutput', '#C8E6C9', '#388E3C')
    
    # Arrows encoder
    draw_arrow(ax, (3.5, 8.5), (3.5, 8.35))
    draw_arrow(ax, (3.5, 7.7), (3.5, 7.35))
    draw_arrow(ax, (3.5, 5.8), (3.5, 5.45))
    
    # DECODER SIDE
    ax.text(10.5, 9.5, 'DECODER', fontsize=14, fontweight='bold', ha='center', color='#C62828')
    
    # Output Embedding
    draw_box(ax, 9.5, 8.5, 2, 0.6, 'Output\nEmbedding', '#FFCDD2', '#C62828')
    draw_box(ax, 9.5, 7.7, 2, 0.6, 'Positional\nEncoding', '#C8E6C9', '#388E3C')
    
    # Decoder Block
    draw_box(ax, 8.5, 4.8, 4, 2.5, '', '#FFEBEE', '#C62828')
    ax.text(10.5, 7.1, 'Decoder Block (Nx)', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 8.8, 6.5, 3.4, 0.5, 'Masked Multi-Head Attention', '#FFCDD2', '#C62828', fontsize=8)
    draw_box(ax, 8.8, 5.9, 3.4, 0.5, 'Cross-Attention (Enc-Dec)', '#E1BEE7', '#7B1FA2', fontsize=8)
    draw_box(ax, 8.8, 5.3, 3.4, 0.5, 'Feed Forward Network', '#FFECB3', '#FFA000', fontsize=8)
    
    # Output
    draw_box(ax, 9.5, 3.8, 2, 0.6, 'Linear', '#B3E5FC', '#0288D1')
    draw_box(ax, 9.5, 3.0, 2, 0.6, 'Softmax', '#B3E5FC', '#0288D1')
    draw_box(ax, 9.5, 2.2, 2, 0.6, 'Output\nProbabilities', '#C8E6C9', '#388E3C')
    
    # Arrows decoder
    draw_arrow(ax, (10.5, 8.5), (10.5, 8.35))
    draw_arrow(ax, (10.5, 7.7), (10.5, 7.35))
    draw_arrow(ax, (10.5, 4.8), (10.5, 4.45))
    draw_arrow(ax, (10.5, 3.8), (10.5, 3.65))
    draw_arrow(ax, (10.5, 3.0), (10.5, 2.85))
    
    # Cross connection
    ax.annotate('', xy=(8.8, 6.15), xytext=(5.5, 5.1),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2, 
                               connectionstyle='arc3,rad=0.2'))
    
    # MATH SECTION
    ax.text(7, 1.8, 'KEY EQUATIONS', fontsize=12, fontweight='bold', ha='center')
    
    # Attention equation
    ax.text(7, 1.3, r'Attention(Q,K,V) = softmax($\frac{QK^T}{\sqrt{d_k}}$)V', 
            fontsize=11, ha='center', family='serif')
    
    # Multi-head
    ax.text(7, 0.8, r'MultiHead = Concat(head$_1$,...,head$_h$)W$^O$', 
            fontsize=10, ha='center', family='serif')
    ax.text(7, 0.4, r'where head$_i$ = Attention(QW$_i^Q$, KW$_i^K$, VW$_i^V$)', 
            fontsize=9, ha='center', family='serif')
    
    # Complexity
    ax.text(13, 1.0, 'Complexity:', fontsize=10, fontweight='bold', ha='center')
    ax.text(13, 0.5, r'O(N$^2$ $\cdot$ d)', fontsize=12, ha='center', color='#C62828')
    
    plt.tight_layout()
    plt.savefig('01_vanilla_transformer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 01_vanilla_transformer/architecture.png")


# ============================================================================
# 2. BERT
# ============================================================================
def create_bert():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('BERT: Bidirectional Encoder Representations from Transformers', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Input
    ax.text(7, 9.5, 'INPUT: [CLS] Token1 Token2 [MASK] Token4 [SEP]', 
            fontsize=11, ha='center', family='monospace', 
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    
    # Token + Segment + Position embeddings
    draw_box(ax, 2, 8.2, 3, 0.6, 'Token\nEmbeddings', '#BBDEFB', '#1976D2')
    draw_box(ax, 5.5, 8.2, 3, 0.6, 'Segment\nEmbeddings', '#C8E6C9', '#388E3C')
    draw_box(ax, 9, 8.2, 3, 0.6, 'Position\nEmbeddings', '#FFECB3', '#FFA000')
    
    ax.text(7, 7.6, '+', fontsize=20, ha='center', fontweight='bold')
    
    # Stacked Encoders
    draw_box(ax, 4, 5.5, 6, 1.8, '', '#E3F2FD', '#1976D2')
    ax.text(7, 7.0, 'Transformer Encoder x 12 (BERT-base) / x 24 (BERT-large)', 
            fontsize=10, fontweight='bold', ha='center')
    
    draw_box(ax, 4.3, 6.3, 5.4, 0.5, 'Bidirectional Self-Attention (All tokens see all)', 
             '#BBDEFB', '#1976D2', fontsize=9)
    draw_box(ax, 4.3, 5.7, 5.4, 0.5, 'Feed Forward Network', '#FFECB3', '#FFA000', fontsize=9)
    
    # Outputs
    draw_box(ax, 1, 4.0, 2.5, 0.8, '[CLS]\nOutput', '#E1BEE7', '#7B1FA2')
    draw_box(ax, 4, 4.0, 2, 0.8, 'T1', '#FFCDD2', '#C62828')
    draw_box(ax, 6.5, 4.0, 2, 0.8, 'T2', '#FFCDD2', '#C62828')
    draw_box(ax, 9, 4.0, 2, 0.8, '[MASK]', '#C8E6C9', '#388E3C')
    draw_box(ax, 11.5, 4.0, 2, 0.8, '...', '#FFCDD2', '#C62828')
    
    # Pre-training tasks
    ax.text(7, 3.2, 'PRE-TRAINING OBJECTIVES', fontsize=12, fontweight='bold', ha='center')
    
    # MLM
    draw_box(ax, 1.5, 1.8, 5, 1.2, '', '#E8F5E9', '#4CAF50')
    ax.text(4, 2.7, 'Masked Language Model (MLM)', fontsize=10, fontweight='bold', ha='center')
    ax.text(4, 2.2, '15% tokens masked randomly', fontsize=9, ha='center')
    ax.text(4, 1.9, 'Predict: [MASK] -> "great"', fontsize=9, ha='center', family='monospace')
    
    # NSP
    draw_box(ax, 7.5, 1.8, 5, 1.2, '', '#FFF3E0', '#FF9800')
    ax.text(10, 2.7, 'Next Sentence Prediction (NSP)', fontsize=10, fontweight='bold', ha='center')
    ax.text(10, 2.2, '[CLS] used for classification', fontsize=9, ha='center')
    ax.text(10, 1.9, 'IsNext / NotNext', fontsize=9, ha='center', family='monospace')
    
    # Key insight
    ax.text(7, 1.0, 'KEY: Bidirectional = sees BOTH left and right context (unlike GPT)', 
            fontsize=11, ha='center', style='italic', 
            bbox=dict(boxstyle='round', facecolor='#FFEB3B', alpha=0.3))
    
    # Model sizes
    ax.text(13, 7, 'BERT-base:', fontsize=9, fontweight='bold', ha='left')
    ax.text(13, 6.6, '110M params', fontsize=9, ha='left')
    ax.text(13, 6.2, '12 layers', fontsize=9, ha='left')
    ax.text(13, 5.8, '768 hidden', fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig('02_bert/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 02_bert/architecture.png")


# ============================================================================
# 3. GPT
# ============================================================================
def create_gpt():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('GPT: Generative Pre-trained Transformer (Decoder-Only)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Autoregressive illustration
    ax.text(7, 9.5, 'AUTOREGRESSIVE: Generate one token at a time', 
            fontsize=11, ha='center', 
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', edgecolor='#C62828'))
    
    # Input sequence
    tokens = ['The', 'cat', 'sat', 'on', '???']
    colors = ['#BBDEFB', '#BBDEFB', '#BBDEFB', '#BBDEFB', '#FFECB3']
    for i, (tok, col) in enumerate(zip(tokens, colors)):
        draw_box(ax, 2 + i*2.2, 8.3, 1.8, 0.6, tok, col, '#1976D2')
    
    # Embedding
    draw_box(ax, 4, 7.2, 6, 0.6, 'Token Embedding + Positional Embedding', '#C8E6C9', '#388E3C')
    
    # Decoder stack
    draw_box(ax, 3, 4.5, 8, 2.4, '', '#FFEBEE', '#C62828')
    ax.text(7, 6.6, 'Decoder Block x N (Stacked)', fontsize=11, fontweight='bold', ha='center')
    
    draw_box(ax, 3.3, 5.9, 7.4, 0.5, 'Masked Self-Attention (Causal)', '#FFCDD2', '#C62828', fontsize=9)
    draw_box(ax, 3.3, 5.3, 7.4, 0.5, 'Feed Forward Network', '#FFECB3', '#FFA000', fontsize=9)
    draw_box(ax, 3.3, 4.7, 7.4, 0.5, 'Layer Norm + Residual', '#E1BEE7', '#7B1FA2', fontsize=9)
    
    # Output
    draw_box(ax, 5, 3.5, 4, 0.6, 'Linear + Softmax', '#B3E5FC', '#0288D1')
    draw_box(ax, 5, 2.7, 4, 0.6, 'Next Token Prediction', '#C8E6C9', '#388E3C')
    
    # Causal mask visualization
    ax.text(12, 8, 'CAUSAL MASK', fontsize=10, fontweight='bold', ha='center')
    mask_data = np.tril(np.ones((5, 5)))
    mask_ax = fig.add_axes([0.78, 0.62, 0.15, 0.15])
    mask_ax.imshow(mask_data, cmap='Greens', aspect='equal')
    mask_ax.set_xticks([])
    mask_ax.set_yticks([])
    mask_ax.set_title('Can only see\npast tokens', fontsize=8)
    
    # Math
    ax.text(7, 2.0, 'KEY EQUATIONS', fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 1.5, r'P(x$_t$ | x$_1$, ..., x$_{t-1}$) = softmax(W$_e$ h$_t$)', 
            fontsize=11, ha='center', family='serif')
    ax.text(7, 1.0, r'Loss = -$\sum_t$ log P(x$_t$ | x$_{<t}$)', 
            fontsize=11, ha='center', family='serif')
    
    # GPT versions
    ax.text(1, 3, 'Versions:', fontsize=10, fontweight='bold')
    ax.text(1, 2.6, 'GPT-1: 117M', fontsize=9)
    ax.text(1, 2.2, 'GPT-2: 1.5B', fontsize=9)
    ax.text(1, 1.8, 'GPT-3: 175B', fontsize=9)
    ax.text(1, 1.4, 'GPT-4: ~1.8T (MoE)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('03_gpt/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 03_gpt/architecture.png")


# ============================================================================
# 4. VISION TRANSFORMER
# ============================================================================
def create_vit():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Vision Transformer (ViT): An Image is Worth 16x16 Words', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Image and patches
    ax.text(2, 9.2, '224x224 Image', fontsize=10, ha='center', fontweight='bold')
    img_ax = fig.add_axes([0.08, 0.72, 0.12, 0.12])
    img_ax.imshow(np.random.rand(14, 14, 3) * 0.3 + 0.5)
    for i in range(0, 15, 2):
        img_ax.axhline(y=i-0.5, color='white', linewidth=1)
        img_ax.axvline(x=i-0.5, color='white', linewidth=1)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    img_ax.set_title('Split into\n16x16 patches', fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(4.5, 8.5), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
    
    # Patch embedding
    ax.text(4.8, 9.2, 'Flatten + Linear Projection', fontsize=9, ha='left')
    patches = ['P1', 'P2', 'P3', '...', 'P196']
    for i, p in enumerate(patches):
        draw_box(ax, 4.5 + i*1.5, 8.2, 1.2, 0.6, p, '#BBDEFB', '#1976D2', fontsize=9)
    
    # CLS token + Position
    ax.text(7, 7.5, '+ [CLS] token + Position Embeddings', fontsize=10, ha='center', fontweight='bold')
    
    draw_box(ax, 2, 6.5, 1.5, 0.6, '[CLS]', '#E1BEE7', '#7B1FA2', fontsize=9)
    draw_box(ax, 3.7, 6.5, 1.2, 0.6, 'P1', '#BBDEFB', '#1976D2', fontsize=9)
    draw_box(ax, 5.1, 6.5, 1.2, 0.6, 'P2', '#BBDEFB', '#1976D2', fontsize=9)
    ax.text(7, 6.8, '...', fontsize=14, ha='center')
    draw_box(ax, 8, 6.5, 1.5, 0.6, 'P196', '#BBDEFB', '#1976D2', fontsize=9)
    draw_box(ax, 9.7, 6.5, 2.5, 0.6, '+ Pos Emb', '#C8E6C9', '#388E3C', fontsize=9)
    
    # Transformer Encoder
    draw_box(ax, 3, 4.2, 8, 1.8, '', '#E3F2FD', '#1976D2')
    ax.text(7, 5.7, 'Transformer Encoder x L', fontsize=11, fontweight='bold', ha='center')
    draw_box(ax, 3.3, 5.0, 7.4, 0.5, 'Multi-Head Self-Attention', '#BBDEFB', '#1976D2', fontsize=9)
    draw_box(ax, 3.3, 4.4, 7.4, 0.5, 'MLP (GELU)', '#FFECB3', '#FFA000', fontsize=9)
    
    # Output
    draw_box(ax, 2, 3.0, 2, 0.6, '[CLS]\nOutput', '#E1BEE7', '#7B1FA2', fontsize=9)
    draw_arrow(ax, (3, 3.3), (5, 3.3))
    draw_box(ax, 5, 2.8, 3, 0.8, 'MLP Head\n(Classification)', '#C8E6C9', '#388E3C')
    draw_arrow(ax, (8, 3.2), (9, 3.2))
    draw_box(ax, 9, 2.8, 3, 0.8, 'Class:\nCat / Dog / ...', '#FFECB3', '#FFA000')
    
    # Math
    ax.text(7, 1.8, 'KEY EQUATIONS', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 1.3, r'x$_{patch}$ = [x$_p^1$E; x$_p^2$E; ...; x$_p^N$E] + E$_{pos}$', 
            fontsize=10, ha='center', family='serif')
    ax.text(7, 0.8, r'where E $\in$ R$^{(P^2 \cdot C) \times D}$, N = (H $\times$ W) / P$^2$', 
            fontsize=9, ha='center', family='serif')
    
    # Model sizes
    ax.text(12.5, 5.5, 'ViT-B/16:', fontsize=9, fontweight='bold')
    ax.text(12.5, 5.1, '86M params', fontsize=9)
    ax.text(12.5, 4.7, '12 layers', fontsize=9)
    ax.text(12.5, 4.3, '768 hidden', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('04_vision_transformer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 04_vision_transformer/architecture.png")


# ============================================================================
# 5. TRANSFORMER-XL
# ============================================================================
def create_transformer_xl():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Transformer-XL: Segment-Level Recurrence + Relative Position', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Problem illustration
    ax.text(7, 9.5, 'PROBLEM: Standard Transformer has fixed context window', 
            fontsize=11, ha='center', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2'))
    
    # Segment 1
    draw_box(ax, 1, 7.5, 3, 1.5, '', '#E3F2FD', '#1976D2')
    ax.text(2.5, 8.7, 'Segment t-1', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 1.2, 7.7, 2.6, 0.5, 'Hidden States', '#BBDEFB', '#1976D2', fontsize=9)
    draw_box(ax, 1.2, 8.1, 2.6, 0.3, 'h[t-1]', '#C8E6C9', '#388E3C', fontsize=8)
    
    # Memory arrow
    ax.annotate('', xy=(5, 8.2), xytext=(4.2, 8.2),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=3))
    ax.text(4.6, 8.6, 'MEMORY\n(cached)', fontsize=8, ha='center', color='#E65100', fontweight='bold')
    
    # Segment 2 (current)
    draw_box(ax, 5, 7.5, 4, 1.5, '', '#FFF3E0', '#E65100')
    ax.text(7, 8.7, 'Segment t (current)', fontsize=10, fontweight='bold', ha='center')
    draw_box(ax, 5.2, 7.7, 3.6, 0.5, 'Current Hidden States', '#FFECB3', '#FFA000', fontsize=9)
    ax.text(7, 8.1, 'Attends to memory + current', fontsize=8, ha='center', style='italic')
    
    # Segment 3
    draw_box(ax, 10, 7.5, 3, 1.5, '', '#E8EAF6', '#3F51B5')
    ax.text(11.5, 8.7, 'Segment t+1', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.5, 8.1, '(future)', fontsize=9, ha='center', style='italic')
    
    # Recurrence diagram
    ax.text(7, 6.5, 'SEGMENT-LEVEL RECURRENCE', fontsize=12, fontweight='bold', ha='center')
    
    # Math box
    draw_box(ax, 2, 4.5, 10, 1.5, '', '#F5F5F5', '#9E9E9E')
    ax.text(7, 5.7, 'Attention with Memory:', fontsize=10, fontweight='bold', ha='center')
    ax.text(7, 5.2, r'$\tilde{h}_\tau^{n-1}$ = [SG(m$_\tau^{n-1}$) $\circ$ h$_\tau^{n-1}$]', 
            fontsize=11, ha='center', family='serif')
    ax.text(7, 4.7, r'q, k, v = h$_\tau^{n-1}$W$_q$, $\tilde{h}_\tau^{n-1}$W$_k$, $\tilde{h}_\tau^{n-1}$W$_v$', 
            fontsize=10, ha='center', family='serif')
    
    # Relative Position
    ax.text(7, 3.8, 'RELATIVE POSITIONAL ENCODING', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(ax, 2, 2.5, 10, 1.0, '', '#E8F5E9', '#4CAF50')
    ax.text(7, 3.2, r'A$_{i,j}$ = q$_i^T$k$_j$ + q$_i^T$W$_{k,R}$R$_{i-j}$ + u$^T$k$_j$ + v$^T$R$_{i-j}$', 
            fontsize=10, ha='center', family='serif')
    ax.text(7, 2.7, 'R = sinusoidal encodings for relative position (i-j)', 
            fontsize=9, ha='center', style='italic')
    
    # Benefits
    ax.text(2, 1.8, 'BENEFITS:', fontsize=10, fontweight='bold')
    ax.text(2, 1.4, '1. Longer effective context', fontsize=9)
    ax.text(2, 1.0, '2. No context fragmentation', fontsize=9)
    ax.text(2, 0.6, '3. Faster evaluation (reuse cache)', fontsize=9)
    
    # Complexity
    ax.text(10, 1.5, 'Complexity:', fontsize=10, fontweight='bold')
    ax.text(10, 1.0, 'O(N) per segment', fontsize=11, color='#388E3C')
    ax.text(10, 0.5, '(memory cached)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('05_transformer_xl/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 05_transformer_xl/architecture.png")


# ============================================================================
# 6. SPARSE TRANSFORMER
# ============================================================================
def create_sparse_transformer():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Sparse Transformer: Efficient O(N sqrt(N)) Attention', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Full attention problem
    ax.text(7, 9.5, r'PROBLEM: Full attention is O(N$^2$) - quadratic scaling', 
            fontsize=11, ha='center', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2'))
    
    # Attention patterns
    ax.text(7, 8.5, 'SPARSE ATTENTION PATTERNS', fontsize=12, fontweight='bold', ha='center')
    
    # Pattern 1: Local (Strided)
    ax.text(2.5, 7.8, 'Local Window', fontsize=10, fontweight='bold', ha='center')
    pattern1 = fig.add_axes([0.08, 0.52, 0.15, 0.15])
    local = np.zeros((16, 16))
    for i in range(16):
        for j in range(max(0, i-3), min(16, i+4)):
            local[i, j] = 1
    pattern1.imshow(local, cmap='Blues')
    pattern1.set_xticks([])
    pattern1.set_yticks([])
    
    # Pattern 2: Strided
    ax.text(7, 7.8, 'Strided', fontsize=10, fontweight='bold', ha='center')
    pattern2 = fig.add_axes([0.40, 0.52, 0.15, 0.15])
    strided = np.zeros((16, 16))
    stride = 4
    for i in range(16):
        for j in range(0, i+1, stride):
            strided[i, j] = 1
    pattern2.imshow(strided, cmap='Greens')
    pattern2.set_xticks([])
    pattern2.set_yticks([])
    
    # Pattern 3: Combined
    ax.text(11.5, 7.8, 'Combined', fontsize=10, fontweight='bold', ha='center')
    pattern3 = fig.add_axes([0.72, 0.52, 0.15, 0.15])
    combined = np.clip(local + strided, 0, 1)
    pattern3.imshow(combined, cmap='Oranges')
    pattern3.set_xticks([])
    pattern3.set_yticks([])
    
    # Explanation
    ax.text(2.5, 5.0, 'Nearby tokens', fontsize=9, ha='center')
    ax.text(7, 5.0, 'Every k-th token', fontsize=9, ha='center')
    ax.text(11.5, 5.0, 'Local + Strided', fontsize=9, ha='center')
    
    # Architecture
    ax.text(7, 4.2, 'FACTORIZED ATTENTION', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(ax, 2, 2.8, 4.5, 1.2, '', '#E3F2FD', '#1976D2')
    ax.text(4.25, 3.7, 'Head A: Local', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.25, 3.2, 'Attends to window\nof size sqrt(N)', fontsize=9, ha='center')
    
    draw_box(ax, 7.5, 2.8, 4.5, 1.2, '', '#E8F5E9', '#4CAF50')
    ax.text(9.75, 3.7, 'Head B: Strided', fontsize=10, fontweight='bold', ha='center')
    ax.text(9.75, 3.2, 'Attends every\nsqrt(N) tokens', fontsize=9, ha='center')
    
    # Math
    ax.text(7, 1.8, 'COMPLEXITY ANALYSIS', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 1.3, r'Full: O(N$^2$)  vs  Sparse: O(N$\sqrt{N}$)', fontsize=12, ha='center')
    
    # Example
    ax.text(12, 1.3, 'N=4096:', fontsize=10, fontweight='bold')
    ax.text(12, 0.9, 'Full: 16M ops', fontsize=9, color='#C62828')
    ax.text(12, 0.5, 'Sparse: 262K ops', fontsize=9, color='#388E3C')
    
    plt.tight_layout()
    plt.savefig('06_sparse_transformer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 06_sparse_transformer/architecture.png")


# ============================================================================
# 7. PERFORMER
# ============================================================================
def create_performer():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Performer: Fast Attention via FAVOR+ (Linear O(N) Complexity)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Key insight
    ax.text(7, 9.3, 'KEY INSIGHT: Approximate softmax(QK^T) with random features', 
            fontsize=11, ha='center', 
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    
    # Standard attention
    ax.text(3.5, 8.2, 'STANDARD ATTENTION', fontsize=11, fontweight='bold', ha='center', color='#C62828')
    draw_box(ax, 1, 6.8, 5, 1.2, '', '#FFEBEE', '#C62828')
    ax.text(3.5, 7.7, r'Attn = softmax($\frac{QK^T}{\sqrt{d}}$)V', fontsize=11, ha='center', family='serif')
    ax.text(3.5, 7.2, r'Compute QK$^T$ first: O(N$^2$d)', fontsize=9, ha='center', color='#C62828')
    
    # FAVOR+
    ax.text(10.5, 8.2, 'FAVOR+ ATTENTION', fontsize=11, fontweight='bold', ha='center', color='#388E3C')
    draw_box(ax, 8, 6.8, 5, 1.2, '', '#E8F5E9', '#4CAF50')
    ax.text(10.5, 7.7, r"Attn $\approx$ $\phi$(Q)($\phi$(K)$^T$V)", fontsize=11, ha='center', family='serif')
    ax.text(10.5, 7.2, r"Compute (K$^T$V) first: O(Nd$^2$)", fontsize=9, ha='center', color='#388E3C')
    
    # Random features explanation
    ax.text(7, 5.8, 'RANDOM FEATURE MAP', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(ax, 2, 4.2, 10, 1.4, '', '#F5F5F5', '#9E9E9E')
    ax.text(7, 5.3, r'$\phi$(x) = $\frac{1}{\sqrt{m}}$[f(w$_1^T$x), ..., f(w$_m^T$x)]', 
            fontsize=11, ha='center', family='serif')
    ax.text(7, 4.8, 'where w_i are random orthogonal projections', fontsize=9, ha='center')
    ax.text(7, 4.4, r'Softmax kernel: K(x,y) $\approx$ $\phi$(x)$^T$$\phi$(y) = exp(x$^T$y)', 
            fontsize=10, ha='center', family='serif')
    
    # Complexity comparison
    ax.text(7, 3.5, 'COMPLEXITY COMPARISON', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(ax, 2, 2.3, 4, 1.0, '', '#FFEBEE', '#C62828')
    ax.text(4, 3.0, 'Standard', fontsize=10, fontweight='bold', ha='center')
    ax.text(4, 2.6, r'O(N$^2$ $\cdot$ d)', fontsize=11, ha='center', color='#C62828')
    
    draw_box(ax, 8, 2.3, 4, 1.0, '', '#E8F5E9', '#4CAF50')
    ax.text(10, 3.0, 'FAVOR+', fontsize=10, fontweight='bold', ha='center')
    ax.text(10, 2.6, r'O(N $\cdot$ d$^2$)', fontsize=11, ha='center', color='#388E3C')
    
    # When d << N, this is much faster!
    ax.text(7, 1.5, 'When N >> d (long sequences): FAVOR+ is much faster!', 
            fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='#FFEB3B', alpha=0.3))
    
    ax.text(7, 0.8, 'Example: N=16384, d=64 -> 256x fewer operations', 
            fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('07_performer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 07_performer/architecture.png")


# ============================================================================
# 8. REFORMER
# ============================================================================
def create_reformer():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Reformer: Efficient Transformers via LSH + Reversible Layers', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Two key innovations
    ax.text(4.5, 9.2, 'LSH ATTENTION', fontsize=12, fontweight='bold', ha='center', color='#1565C0')
    ax.text(10.5, 9.2, 'REVERSIBLE LAYERS', fontsize=12, fontweight='bold', ha='center', color='#388E3C')
    
    # LSH explanation
    draw_box(ax, 1.5, 6.5, 6, 2.5, '', '#E3F2FD', '#1976D2')
    ax.text(4.5, 8.7, 'Locality Sensitive Hashing', fontsize=10, fontweight='bold', ha='center')
    ax.text(4.5, 8.2, 'Similar queries -> same bucket', fontsize=9, ha='center')
    ax.text(4.5, 7.7, 'Only attend within buckets!', fontsize=9, ha='center', style='italic')
    
    # LSH diagram
    ax.text(2.5, 7.2, 'Bucket 1', fontsize=8, ha='center')
    ax.text(4.5, 7.2, 'Bucket 2', fontsize=8, ha='center')
    ax.text(6.5, 7.2, 'Bucket 3', fontsize=8, ha='center')
    draw_box(ax, 1.8, 6.7, 1.4, 0.4, 'q1,q4,q7', '#BBDEFB', '#1976D2', fontsize=7)
    draw_box(ax, 3.8, 6.7, 1.4, 0.4, 'q2,q5', '#BBDEFB', '#1976D2', fontsize=7)
    draw_box(ax, 5.8, 6.7, 1.4, 0.4, 'q3,q6', '#BBDEFB', '#1976D2', fontsize=7)
    
    # Reversible explanation
    draw_box(ax, 7.5, 6.5, 6, 2.5, '', '#E8F5E9', '#4CAF50')
    ax.text(10.5, 8.7, 'No Activation Storage!', fontsize=10, fontweight='bold', ha='center')
    ax.text(10.5, 8.2, 'Forward:', fontsize=9, ha='center')
    ax.text(10.5, 7.7, 'y1 = x1 + F(x2)', fontsize=9, ha='center', family='monospace')
    ax.text(10.5, 7.3, 'y2 = x2 + G(y1)', fontsize=9, ha='center', family='monospace')
    ax.text(10.5, 6.8, 'Backward: recompute x from y!', fontsize=9, ha='center', style='italic')
    
    # Math
    ax.text(7, 5.5, 'LSH ATTENTION COMPLEXITY', fontsize=11, fontweight='bold', ha='center')
    
    draw_box(ax, 2, 4.2, 10, 1.0, '', '#F5F5F5', '#9E9E9E')
    ax.text(7, 4.9, r'Standard: O(N$^2$)  ->  LSH: O(N log N)', fontsize=12, ha='center')
    ax.text(7, 4.4, 'n_rounds * (n_buckets * bucket_size^2)', fontsize=9, ha='center')
    
    # Memory savings
    ax.text(7, 3.5, 'MEMORY SAVINGS', fontsize=11, fontweight='bold', ha='center')
    
    draw_box(ax, 2, 2.2, 4.5, 1.1, '', '#FFEBEE', '#C62828')
    ax.text(4.25, 2.9, 'Standard Transformer', fontsize=9, fontweight='bold', ha='center')
    ax.text(4.25, 2.5, 'Store all activations', fontsize=9, ha='center')
    ax.text(4.25, 2.2, r'Memory: O(N $\cdot$ L)', fontsize=9, ha='center', color='#C62828')
    
    draw_box(ax, 7.5, 2.2, 4.5, 1.1, '', '#E8F5E9', '#4CAF50')
    ax.text(9.75, 2.9, 'Reformer', fontsize=9, fontweight='bold', ha='center')
    ax.text(9.75, 2.5, 'Recompute on backward', fontsize=9, ha='center')
    ax.text(9.75, 2.2, r'Memory: O(N)', fontsize=9, ha='center', color='#388E3C')
    
    # Summary
    ax.text(7, 1.2, 'RESULT: Process 64K tokens on single GPU!', 
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFEB3B', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('08_reformer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 08_reformer/architecture.png")


# ============================================================================
# 9. LONGFORMER
# ============================================================================
def create_longformer():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Longformer: Long Document Transformer (Local + Global Attention)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Main idea
    ax.text(7, 9.3, 'Sliding Window (Local) + Global Attention for [CLS] and special tokens', 
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    
    # Token sequence
    ax.text(7, 8.3, 'Input Sequence', fontsize=11, fontweight='bold', ha='center')
    tokens = ['[CLS]', 'The', 'cat', 'sat', 'on', 'the', 'mat', '.', '[SEP]']
    for i, tok in enumerate(tokens):
        color = '#E1BEE7' if tok in ['[CLS]', '[SEP]'] else '#BBDEFB'
        draw_box(ax, 1 + i*1.4, 7.5, 1.2, 0.6, tok, color, '#1976D2', fontsize=8)
    
    # Attention pattern
    ax.text(7, 6.5, 'ATTENTION PATTERNS', fontsize=12, fontweight='bold', ha='center')
    
    # Sliding window
    ax.text(3.5, 5.8, 'Sliding Window', fontsize=10, fontweight='bold', ha='center', color='#1565C0')
    pattern1 = fig.add_axes([0.12, 0.38, 0.18, 0.18])
    n = 9
    sliding = np.zeros((n, n))
    w = 2
    for i in range(n):
        for j in range(max(0, i-w), min(n, i+w+1)):
            sliding[i, j] = 1
    pattern1.imshow(sliding, cmap='Blues')
    pattern1.set_xticks([])
    pattern1.set_yticks([])
    pattern1.set_xlabel('Local context only', fontsize=8)
    
    # Global attention
    ax.text(10.5, 5.8, 'Global (for [CLS])', fontsize=10, fontweight='bold', ha='center', color='#7B1FA2')
    pattern2 = fig.add_axes([0.62, 0.38, 0.18, 0.18])
    global_attn = np.zeros((n, n))
    global_attn[0, :] = 1  # CLS attends to all
    global_attn[:, 0] = 1  # All attend to CLS
    global_attn[-1, :] = 1  # SEP
    global_attn[:, -1] = 1
    pattern2.imshow(global_attn, cmap='Purples')
    pattern2.set_xticks([])
    pattern2.set_yticks([])
    pattern2.set_xlabel('[CLS],[SEP] see everything', fontsize=8)
    
    # Combined
    ax.text(7, 3.5, 'COMBINED PATTERN', fontsize=11, fontweight='bold', ha='center')
    pattern3 = fig.add_axes([0.38, 0.18, 0.22, 0.22])
    combined = np.clip(sliding + global_attn, 0, 1)
    pattern3.imshow(combined, cmap='Oranges')
    pattern3.set_xticks(range(n))
    pattern3.set_xticklabels(tokens, fontsize=6, rotation=45)
    pattern3.set_yticks(range(n))
    pattern3.set_yticklabels(tokens, fontsize=6)
    
    # Complexity
    ax.text(12, 5.5, 'Complexity:', fontsize=10, fontweight='bold')
    ax.text(12, 5.0, r'O(N $\times$ w)', fontsize=11, color='#388E3C')
    ax.text(12, 4.5, 'w = window size', fontsize=9)
    ax.text(12, 4.0, '(typically 512)', fontsize=9)
    
    # Use cases
    ax.text(12, 3.0, 'Best for:', fontsize=10, fontweight='bold')
    ax.text(12, 2.5, '- Long documents', fontsize=9)
    ax.text(12, 2.0, '- Question answering', fontsize=9)
    ax.text(12, 1.5, '- Summarization', fontsize=9)
    
    # Comparison
    ax.text(2, 1.5, 'vs BERT (512) -> Longformer: 4096+ tokens', fontsize=10, ha='left',
            bbox=dict(boxstyle='round', facecolor='#FFEB3B', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('09_longformer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 09_longformer/architecture.png")


# ============================================================================
# 10. SWITCH TRANSFORMER
# ============================================================================
def create_switch_transformer():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Switch Transformer: Mixture of Experts (MoE) - Trillion Parameters!', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Key insight
    ax.text(7, 9.3, 'Key: More parameters WITHOUT more compute per token', 
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFEB3B', edgecolor='#FFA000'))
    
    # Standard FFN
    ax.text(3.5, 8.3, 'STANDARD FFN', fontsize=11, fontweight='bold', ha='center', color='#1565C0')
    draw_box(ax, 1.5, 6.8, 4, 1.3, '', '#E3F2FD', '#1976D2')
    draw_box(ax, 1.8, 7.5, 3.4, 0.5, 'Single FFN\n(all tokens)', '#BBDEFB', '#1976D2', fontsize=9)
    ax.text(3.5, 7.0, 'Every token uses same FFN', fontsize=8, ha='center')
    
    # Switch (MoE)
    ax.text(10.5, 8.3, 'SWITCH (MoE)', fontsize=11, fontweight='bold', ha='center', color='#388E3C')
    draw_box(ax, 7.5, 6.5, 6, 1.8, '', '#E8F5E9', '#4CAF50')
    
    # Router
    draw_box(ax, 9.8, 7.8, 1.4, 0.4, 'Router', '#FFECB3', '#FFA000', fontsize=8)
    
    # Experts
    experts = ['E1', 'E2', 'E3', 'E4']
    colors = ['#BBDEFB', '#C8E6C9', '#FFCDD2', '#E1BEE7']
    for i, (e, c) in enumerate(zip(experts, colors)):
        draw_box(ax, 7.8 + i*1.4, 6.7, 1.1, 0.6, e, c, '#424242', fontsize=9)
    
    ax.text(10.5, 6.4, 'Each token -> 1 expert (top-1 routing)', fontsize=8, ha='center')
    
    # Routing visualization
    ax.text(7, 5.3, 'TOKEN ROUTING', fontsize=11, fontweight='bold', ha='center')
    
    tokens_in = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    routing = [0, 2, 1, 0, 3, 2]  # Which expert each token goes to
    
    for i, (tok, r) in enumerate(zip(tokens_in, routing)):
        x = 2 + i * 1.8
        draw_box(ax, x, 4.6, 1.2, 0.5, tok, '#F5F5F5', '#9E9E9E', fontsize=9)
        # Arrow to expert
        exp_x = 3 + r * 2.2
        ax.annotate('', xy=(exp_x, 4.0), xytext=(x + 0.6, 4.6),
                   arrowprops=dict(arrowstyle='->', color=colors[r], lw=1.5))
    
    # Experts row
    for i, (e, c) in enumerate(zip(experts, colors)):
        draw_box(ax, 2.4 + i*2.2, 3.3, 1.6, 0.6, f'Expert {i+1}', c, '#424242', fontsize=8)
    
    # Math
    ax.text(7, 2.5, 'LOAD BALANCING LOSS', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 2.0, r'L$_{aux}$ = $\alpha$ $\cdot$ N $\cdot$ $\sum_i$ f$_i$ $\cdot$ P$_i$', 
            fontsize=11, ha='center', family='serif')
    ax.text(7, 1.5, 'Prevents all tokens going to one expert', fontsize=9, ha='center', style='italic')
    
    # Stats
    ax.text(12.5, 4.5, 'SCALE:', fontsize=10, fontweight='bold')
    ax.text(12.5, 4.0, 'Switch-C:', fontsize=9)
    ax.text(12.5, 3.6, '1.6T params', fontsize=9, color='#C62828')
    ax.text(12.5, 3.2, '2048 experts', fontsize=9)
    ax.text(12.5, 2.8, 'Same FLOPs as', fontsize=9)
    ax.text(12.5, 2.4, 'T5-XXL!', fontsize=9, color='#388E3C')
    
    plt.tight_layout()
    plt.savefig('10_switch_transformer/architecture.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: 10_switch_transformer/architecture.png")


# ============================================================================
# BANNER
# ============================================================================
def create_banner():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Background gradient effect
    for i in range(50):
        alpha = 0.02
        ax.axhspan(i/10, (i+1)/10, color='#1976D2', alpha=alpha*(50-i)/50)
    
    # Title
    ax.text(7, 3.8, 'TRANSFORMER ARCHITECTURES', fontsize=28, fontweight='bold', 
            ha='center', va='center', color='#1565C0')
    ax.text(7, 3.0, 'From Attention Is All You Need to GPT-4', fontsize=14, 
            ha='center', va='center', color='#424242')
    
    # Architecture names
    archs = ['Vanilla', 'BERT', 'GPT', 'ViT', 'Sparse', 'Performer', 'Longformer', 'MoE']
    for i, arch in enumerate(archs):
        x = 1.5 + i * 1.5
        draw_box(ax, x-0.5, 1.8, 1.3, 0.5, arch, '#E3F2FD', '#1976D2', fontsize=8)
    
    # Bottom text
    ax.text(7, 1.0, '10 Architectures | PyTorch Code | Train on Colab | 2017-2025', 
            fontsize=11, ha='center', va='center', color='#616161')
    
    plt.tight_layout()
    plt.savefig('banner.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: banner.png")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Generating architecture diagrams...")
    create_banner()
    create_vanilla_transformer()
    create_bert()
    create_gpt()
    create_vit()
    create_transformer_xl()
    create_sparse_transformer()
    create_performer()
    create_reformer()
    create_longformer()
    create_switch_transformer()
    
    print("\nAll diagrams generated successfully!")
