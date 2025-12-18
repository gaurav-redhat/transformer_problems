#!/usr/bin/env python3
"""
Generate clean infographic images for Transformer Problems.
Each image has: Title, Problem description with diagram/math, and Solutions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

# Colors
TITLE_COLOR = '#1565C0'
PROBLEM_BG = '#FFF3E0'
PROBLEM_BORDER = '#E65100'
SOLUTION_BG = '#E8F5E9'
SOLUTION_BORDER = '#2E7D32'
DIAGRAM_BG = '#FAFAFA'
TEXT_COLOR = '#212121'


def create_problem_01(fig, ax):
    """Quadratic Complexity O(N²)"""
    # Title
    ax.text(0.5, 0.95, 'Problem 1: Quadratic Complexity O(N²)', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.55), 0.9, 0.35,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    # Math equation
    ax.text(0.5, 0.78, r'Attention$(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Complexity explanation
    ax.text(0.5, 0.68, 'Sequence length N = 1000 tokens',
            fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.62, r'Attention matrix size: N × N = 1,000,000 elements!',
            fontsize=10, ha='center', color='#C62828', fontweight='bold', transform=ax.transAxes)
    
    # Draw mini attention matrix
    matrix_ax = fig.add_axes([0.35, 0.38, 0.3, 0.18])
    matrix = np.random.rand(6, 6)
    matrix_ax.imshow(matrix, cmap='Reds', aspect='auto')
    matrix_ax.set_xlabel('Keys (N)', fontsize=8)
    matrix_ax.set_ylabel('Queries (N)', fontsize=8)
    matrix_ax.set_xticks([])
    matrix_ax.set_yticks([])
    matrix_ax.set_title('N×N Attention Matrix', fontsize=9, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.28,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.27, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Sparse Attention', 'Longformer', 'Linformer', 
                 'Performer', 'FlashAttention', 'Transformer-XL']
    for i, sol in enumerate(solutions[:3]):
        ax.text(0.15, 0.20 - i*0.055, f'• {sol}', fontsize=10, transform=ax.transAxes)
    for i, sol in enumerate(solutions[3:]):
        ax.text(0.55, 0.20 - i*0.055, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_02(fig, ax):
    """No Positional Awareness"""
    ax.text(0.5, 0.95, 'Problem 2: No Positional Awareness', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Self-attention is permutation invariant!',
            fontsize=11, ha='center', transform=ax.transAxes, fontweight='bold')
    
    # Example
    ax.text(0.5, 0.72, 'Example:', fontsize=10, ha='center', transform=ax.transAxes, 
            fontstyle='italic', color='gray')
    
    # Draw boxes for tokens
    tokens1 = ['Dog', 'bites', 'man']
    tokens2 = ['man', 'bites', 'Dog']
    
    for i, token in enumerate(tokens1):
        rect = FancyBboxPatch((0.15 + i*0.22, 0.62), 0.18, 0.07,
                              boxstyle="round,pad=0.01", transform=ax.transAxes,
                              facecolor='#BBDEFB', edgecolor='#1976D2', lw=1.5)
        ax.add_patch(rect)
        ax.text(0.24 + i*0.22, 0.655, token, fontsize=10, ha='center', 
                va='center', transform=ax.transAxes, fontweight='bold')
    
    ax.text(0.5, 0.56, '= (Same to Transformer!)', fontsize=10, ha='center', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    for i, token in enumerate(tokens2):
        rect = FancyBboxPatch((0.15 + i*0.22, 0.47), 0.18, 0.07,
                              boxstyle="round,pad=0.01", transform=ax.transAxes,
                              facecolor='#FFCDD2', edgecolor='#C62828', lw=1.5)
        ax.add_patch(rect)
        ax.text(0.24 + i*0.22, 0.505, token, fontsize=10, ha='center', 
                va='center', transform=ax.transAxes, fontweight='bold')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = [('Sinusoidal PE', r'$PE_{pos,2i} = \sin(pos/10000^{2i/d})$'),
                 ('Learnable PE', 'Trained position embeddings'),
                 ('RoPE', 'Rotary Position Embedding'),
                 ('ALiBi', 'Attention with Linear Biases')]
    
    for i, (name, desc) in enumerate(solutions):
        ax.text(0.08, 0.30 - i*0.065, f'• {name}:', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.35, 0.30 - i*0.065, desc, fontsize=9, transform=ax.transAxes, color='gray')


def create_problem_03(fig, ax):
    """Fixed Context Window"""
    ax.text(0.5, 0.95, 'Problem 3: Fixed Context Window', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Transformer has a maximum sequence length limit',
            fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.74, 'Example: GPT-2 = 1024 tokens, GPT-4 = 128K tokens',
            fontsize=9, ha='center', transform=ax.transAxes, color='gray')
    
    # Draw token sequence
    ax.text(0.5, 0.67, 'Long Document:', fontsize=10, ha='center', 
            transform=ax.transAxes, fontstyle='italic')
    
    # Forgotten tokens (gray)
    for i in range(6):
        rect = Rectangle((0.1 + i*0.06, 0.58), 0.05, 0.06, transform=ax.transAxes,
                         facecolor='#E0E0E0', edgecolor='#9E9E9E', lw=1)
        ax.add_patch(rect)
    
    # Active window (green)
    for i in range(4):
        rect = Rectangle((0.52 + i*0.06, 0.58), 0.05, 0.06, transform=ax.transAxes,
                         facecolor='#C8E6C9', edgecolor='#388E3C', lw=1)
        ax.add_patch(rect)
    
    ax.text(0.25, 0.52, 'Forgotten!', fontsize=9, ha='center', 
            transform=ax.transAxes, color='#C62828')
    ax.text(0.66, 0.52, 'Context Window', fontsize=9, ha='center', 
            transform=ax.transAxes, color='#388E3C')
    
    ax.annotate('', xy=(0.46, 0.55), xytext=(0.10, 0.55),
                arrowprops=dict(arrowstyle='<->', color='#C62828', lw=1.5),
                transform=ax.transAxes)
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = [('Transformer-XL', 'Segment-level recurrence'),
                 ('RAG', 'Retrieval-Augmented Generation'),
                 ('External Memory', 'Memory banks like NTM'),
                 ('Mamba', 'State Space Models for infinite context')]
    
    for i, (name, desc) in enumerate(solutions):
        ax.text(0.08, 0.30 - i*0.065, f'• {name}:', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.38, 0.30 - i*0.065, desc, fontsize=9, transform=ax.transAxes, color='gray')


def create_problem_04(fig, ax):
    """Slow Autoregressive Decoding"""
    ax.text(0.5, 0.95, 'Problem 4: Slow Autoregressive Decoding', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Generation is sequential: one token at a time',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Draw sequential generation
    tokens = ['The', 'cat', 'sat', 'on', '...']
    for i, token in enumerate(tokens):
        color = '#C8E6C9' if i < 4 else '#FFF9C4'
        edge = '#388E3C' if i < 4 else '#F9A825'
        rect = FancyBboxPatch((0.1 + i*0.16, 0.62), 0.14, 0.08,
                              boxstyle="round,pad=0.01", transform=ax.transAxes,
                              facecolor=color, edgecolor=edge, lw=1.5)
        ax.add_patch(rect)
        ax.text(0.17 + i*0.16, 0.66, token, fontsize=9, ha='center', 
                va='center', transform=ax.transAxes, fontweight='bold')
        
        if i < 4:
            ax.annotate('', xy=(0.24 + i*0.16, 0.66), xytext=(0.26 + i*0.16, 0.66),
                        arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5),
                        transform=ax.transAxes)
    
    # Time labels
    for i in range(5):
        ax.text(0.17 + i*0.16, 0.57, f't={i+1}', fontsize=8, ha='center', 
                transform=ax.transAxes, color='gray')
    
    ax.text(0.5, 0.50, 'N tokens = N forward passes = SLOW!',
            fontsize=11, ha='center', transform=ax.transAxes, 
            color='#C62828', fontweight='bold')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = [('KV Cache', 'Cache key-value pairs'),
                 ('Speculative Decoding', 'Draft model proposes, main verifies'),
                 ('Non-Autoregressive', 'Generate all tokens in parallel'),
                 ('Distillation', 'Smaller, faster student models')]
    
    for i, (name, desc) in enumerate(solutions):
        ax.text(0.08, 0.30 - i*0.065, f'• {name}:', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.42, 0.30 - i*0.065, desc, fontsize=9, transform=ax.transAxes, color='gray')


def create_problem_05(fig, ax):
    """No Local Inductive Bias"""
    ax.text(0.5, 0.95, 'Problem 5: No Local Inductive Bias', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'No built-in preference for local patterns',
            fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.74, 'CNNs learn local-to-global; Transformers treat all equal',
            fontsize=9, ha='center', transform=ax.transAxes, color='gray')
    
    # Draw image grid
    ax.text(0.25, 0.68, 'Image Patches:', fontsize=9, ha='center', transform=ax.transAxes)
    for i in range(4):
        for j in range(4):
            if 1 <= i <= 2 and 1 <= j <= 2:
                color, edge = '#FFCDD2', '#C62828'  # Local region
            else:
                color, edge = '#E3F2FD', '#1976D2'
            rect = Rectangle((0.12 + j*0.07, 0.50 + (3-i)*0.04), 0.06, 0.035,
                            transform=ax.transAxes, facecolor=color, edgecolor=edge, lw=0.8)
            ax.add_patch(rect)
    
    ax.text(0.70, 0.68, 'CNN: Local first', fontsize=9, ha='center', 
            transform=ax.transAxes, color='#388E3C')
    ax.text(0.70, 0.62, 'Transformer: Global', fontsize=9, ha='center', 
            transform=ax.transAxes, color='#C62828')
    ax.text(0.70, 0.55, '(needs more data)', fontsize=8, ha='center', 
            transform=ax.transAxes, color='gray')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['CNN + Transformer hybrid', 'Swin Transformer (shifted windows)',
                 'Conformer (conv + attention)', 'Hierarchical Vision Transformers']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_06(fig, ax):
    """Data-Hungry Architecture"""
    ax.text(0.5, 0.95, 'Problem 6: Data-Hungry Architecture', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.42), 0.9, 0.48,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Transformers need massive datasets to generalize',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Bar chart
    bar_ax = fig.add_axes([0.2, 0.48, 0.6, 0.25])
    models = ['CNN', 'RNN', 'Transformer']
    data_needed = [100, 150, 500]
    colors = ['#81C784', '#81C784', '#EF5350']
    
    bars = bar_ax.bar(models, data_needed, color=colors, edgecolor='black', lw=1)
    bar_ax.set_ylabel('Relative Data Needed', fontsize=9)
    bar_ax.set_title('Data Requirements Comparison', fontsize=10)
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, data_needed):
        bar_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{val}%', ha='center', fontsize=9)
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.35,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.34, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Self-Supervised Pretraining (BERT, GPT)', 
                 'Transfer Learning from pretrained models',
                 'Fine-Tuning on downstream tasks',
                 'Knowledge Distillation']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.27 - i*0.06, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_07(fig, ax):
    """High Memory Footprint"""
    ax.text(0.5, 0.95, 'Problem 7: High Memory Footprint', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'GPU memory grows with sequence length',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.73, r'Memory: O(N² + N × d × layers)', fontsize=10, ha='center', 
            transform=ax.transAxes, color='#C62828')
    
    # Memory blocks
    components = [('Q', 0.12, '#BBDEFB'), ('K', 0.12, '#C8E6C9'), 
                  ('V', 0.12, '#FFF9C4'), ('KV Cache', 0.25, '#FFCDD2')]
    x = 0.12
    for name, width, color in components:
        rect = FancyBboxPatch((x, 0.55), width, 0.12,
                              boxstyle="round,pad=0.01", transform=ax.transAxes,
                              facecolor=color, edgecolor='gray', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + width/2, 0.61, name, fontsize=9, ha='center', 
                va='center', transform=ax.transAxes, fontweight='bold')
        x += width + 0.03
    
    ax.text(0.5, 0.50, 'OOM Error: Out of Memory!', fontsize=11, ha='center', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['FlashAttention (IO-aware attention)',
                 'KV Cache Optimization (PagedAttention)',
                 'Gradient Checkpointing',
                 'Memory-Efficient Attention']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_08(fig, ax):
    """High Compute & Power Cost"""
    ax.text(0.5, 0.95, 'Problem 8: High Compute & Power Cost', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.79, 'Training large transformers is extremely expensive',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Stats
    stats = [('GPT-3 Training:', '~$4.6 Million'),
             ('FLOPs:', '3.14 × 10²³'),
             ('CO2 Emissions:', '~552 tons')]
    
    for i, (label, value) in enumerate(stats):
        ax.text(0.25, 0.70 - i*0.06, label, fontsize=10, ha='right', 
                transform=ax.transAxes)
        ax.text(0.27, 0.70 - i*0.06, value, fontsize=10, ha='left', 
                transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    ax.text(0.5, 0.50, 'Not feasible for most organizations!',
            fontsize=10, ha='center', transform=ax.transAxes, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Quantization (INT8/INT4 precision)',
                 'Lightweight Transformers (MobileBERT)',
                 'Operator Fusion',
                 'Pruning (remove unimportant weights)']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_09(fig, ax):
    """Poor Length Generalization"""
    ax.text(0.5, 0.95, 'Problem 9: Poor Length Generalization', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Models fail on sequences longer than training length',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Training length
    ax.text(0.12, 0.72, 'Train (N=512):', fontsize=9, ha='left', 
            transform=ax.transAxes, fontweight='bold')
    for i in range(5):
        rect = Rectangle((0.35 + i*0.08, 0.70), 0.06, 0.05, transform=ax.transAxes,
                         facecolor='#C8E6C9', edgecolor='#388E3C', lw=1)
        ax.add_patch(rect)
    ax.text(0.80, 0.72, 'OK', fontsize=10, ha='left', 
            transform=ax.transAxes, color='#388E3C', fontweight='bold')
    
    # Test length (longer)
    ax.text(0.12, 0.62, 'Test (N=1024):', fontsize=9, ha='left', 
            transform=ax.transAxes, fontweight='bold')
    for i in range(8):
        color = '#C8E6C9' if i < 5 else '#FFCDD2'
        edge = '#388E3C' if i < 5 else '#C62828'
        rect = Rectangle((0.35 + i*0.06, 0.60), 0.05, 0.05, transform=ax.transAxes,
                         facecolor=color, edgecolor=edge, lw=1)
        ax.add_patch(rect)
    ax.text(0.86, 0.62, 'FAIL', fontsize=10, ha='left', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    ax.text(0.5, 0.52, 'Positional encodings break beyond training length!',
            fontsize=10, ha='center', transform=ax.transAxes, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Relative Position Encoding',
                 'RoPE Scaling (Position Interpolation)',
                 'ALiBi (linear bias extrapolates)',
                 'YaRN (efficient context extension)']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_10(fig, ax):
    """Training Instability"""
    ax.text(0.5, 0.95, 'Problem 10: Training Instability', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.42), 0.9, 0.48,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Deep transformers suffer from gradient issues',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Loss curve with spike
    loss_ax = fig.add_axes([0.2, 0.48, 0.6, 0.25])
    x = np.linspace(0, 10, 100)
    y = 3 * np.exp(-0.3 * x) + 0.5 + 0.2 * np.sin(x)
    y[40:45] = y[40:45] + 1.5  # Add spike
    
    loss_ax.plot(x, y, 'b-', lw=2)
    loss_ax.axvline(x=4.2, color='r', linestyle='--', lw=1)
    loss_ax.annotate('Loss Spike!', xy=(4.2, 2.8), xytext=(6, 3.2),
                    fontsize=9, color='#C62828',
                    arrowprops=dict(arrowstyle='->', color='#C62828'))
    loss_ax.set_xlabel('Training Steps', fontsize=9)
    loss_ax.set_ylabel('Loss', fontsize=9)
    loss_ax.set_title('Training Loss with Instability', fontsize=10)
    loss_ax.spines['top'].set_visible(False)
    loss_ax.spines['right'].set_visible(False)
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.35,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.34, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Pre-LayerNorm (normalize before attention)',
                 'Learning Rate Warmup',
                 'AdamW Optimizer',
                 'Gradient Clipping']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.27 - i*0.06, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_11(fig, ax):
    """Attention Over-Smoothing"""
    ax.text(0.5, 0.95, 'Problem 11: Attention Over-Smoothing', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Token representations converge in deep layers',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Show tokens becoming similar
    layers = ['Layer 1', 'Layer 12', 'Layer 24']
    layer_colors = [
        ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726'],  # Diverse
        ['#E57373', '#64B5F6', '#81C784', '#FFB74D'],  # Less diverse
        ['#BDBDBD', '#BDBDBD', '#BDBDBD', '#BDBDBD']   # All same (gray)
    ]
    
    for j, (layer, colors) in enumerate(zip(layers, layer_colors)):
        ax.text(0.18 + j*0.28, 0.73, layer, fontsize=9, ha='center', 
                transform=ax.transAxes, color='gray')
        for i, color in enumerate(colors):
            rect = Rectangle((0.10 + j*0.28 + i*0.04, 0.58), 0.03, 0.12,
                            transform=ax.transAxes, facecolor=color, edgecolor='gray', lw=0.5)
            ax.add_patch(rect)
    
    ax.annotate('', xy=(0.82, 0.64), xytext=(0.22, 0.64),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2),
                transform=ax.transAxes)
    
    ax.text(0.5, 0.52, 'All tokens become identical = Information loss!',
            fontsize=10, ha='center', transform=ax.transAxes, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Residual Scaling', 'DropHead (drop attention heads)',
                 'Attention Temperature Control', 'Skip Connections']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_12(fig, ax):
    """No Recurrence / Streaming"""
    ax.text(0.5, 0.95, 'Problem 12: No Recurrence / Streaming', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Must process entire sequence before outputting',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Streaming data
    ax.text(0.12, 0.72, 'Stream:', fontsize=10, ha='left', 
            transform=ax.transAxes, fontweight='bold')
    tokens = ['t1', 't2', 't3', '...', 'tN']
    for i, t in enumerate(tokens):
        rect = FancyBboxPatch((0.25 + i*0.12, 0.69), 0.10, 0.06,
                              boxstyle="round,pad=0.01", transform=ax.transAxes,
                              facecolor='#BBDEFB', edgecolor='#1976D2', lw=1)
        ax.add_patch(rect)
        ax.text(0.30 + i*0.12, 0.72, t, fontsize=8, ha='center', 
                va='center', transform=ax.transAxes)
    
    # Transformer waiting
    rect = FancyBboxPatch((0.25, 0.52), 0.5, 0.10,
                          boxstyle="round,pad=0.01", transform=ax.transAxes,
                          facecolor='#FFF9C4', edgecolor='#F9A825', lw=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.57, 'Transformer: Wait for ALL tokens!', fontsize=9, 
            ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    
    ax.text(0.5, 0.48, 'Bad for: Live speech, real-time translation, etc.',
            fontsize=9, ha='center', transform=ax.transAxes, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Transformer-XL (segment recurrence)',
                 'Chunk-based Attention',
                 'Streaming Transformers',
                 'Delta Attention (incremental updates)']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_13(fig, ax):
    """Large Model Size"""
    ax.text(0.5, 0.95, 'Problem 13: Large Model Size', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.42), 0.9, 0.48,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Modern LLMs have billions of parameters',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Model sizes bar chart
    bar_ax = fig.add_axes([0.15, 0.48, 0.7, 0.25])
    models = ['BERT\n(110M)', 'GPT-2\n(1.5B)', 'GPT-3\n(175B)', 'PaLM\n(540B)']
    sizes = [0.11, 1.5, 175, 540]
    colors = ['#81C784', '#FFF176', '#FFB74D', '#EF5350']
    
    bars = bar_ax.bar(models, sizes, color=colors, edgecolor='black', lw=1)
    bar_ax.set_ylabel('Parameters (Billions)', fontsize=9)
    bar_ax.set_yscale('log')
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.35,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.34, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['LoRA (Low-Rank Adaptation)', 'QLoRA (Quantized LoRA)',
                 'ALBERT (Parameter Sharing)', 'Pruning & Distillation']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.27 - i*0.06, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_14(fig, ax):
    """Sensitivity to Noise Tokens"""
    ax.text(0.5, 0.95, 'Problem 14: Sensitivity to Noise Tokens', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Global attention attends to irrelevant tokens',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Tokens with noise
    tokens = [('The', True), ('cat', True), ('[PAD]', False), 
              ('sat', True), ('[UNK]', False)]
    for i, (token, relevant) in enumerate(tokens):
        color = '#C8E6C9' if relevant else '#FFCDD2'
        edge = '#388E3C' if relevant else '#C62828'
        rect = FancyBboxPatch((0.12 + i*0.16, 0.65), 0.14, 0.08,
                              boxstyle="round,pad=0.01", transform=ax.transAxes,
                              facecolor=color, edgecolor=edge, lw=1.5)
        ax.add_patch(rect)
        ax.text(0.19 + i*0.16, 0.69, token, fontsize=8, ha='center', 
                va='center', transform=ax.transAxes, fontweight='bold')
    
    # Attention lines
    ax.text(0.5, 0.58, 'Query attends to ALL tokens equally!',
            fontsize=10, ha='center', transform=ax.transAxes, color='#C62828')
    ax.text(0.5, 0.52, 'Wastes model capacity on noise',
            fontsize=9, ha='center', transform=ax.transAxes, color='gray')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Sparse Attention (attend to subset)',
                 'Token Pruning (remove unimportant tokens)',
                 'Attention Masking',
                 'Gating Mechanisms']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_15(fig, ax):
    """Poor Interpretability"""
    ax.text(0.5, 0.95, 'Problem 15: Poor Interpretability', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Attention weights are NOT explanations',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Attention heatmap vs true importance
    ax.text(0.25, 0.72, 'Attention', fontsize=9, ha='center', 
            transform=ax.transAxes, color='gray')
    ax.text(0.75, 0.72, 'True Impact', fontsize=9, ha='center', 
            transform=ax.transAxes, color='gray')
    
    attention = [0.9, 0.3, 0.5, 0.1]
    importance = [0.2, 0.8, 0.3, 0.7]
    
    for i, (att, imp) in enumerate(zip(attention, importance)):
        # Attention (misleading)
        rect = Rectangle((0.12 + i*0.08, 0.60), 0.06, 0.08, transform=ax.transAxes,
                         facecolor=plt.cm.Blues(att), edgecolor='gray', lw=0.5)
        ax.add_patch(rect)
        # True importance (different)
        rect = Rectangle((0.62 + i*0.08, 0.60), 0.06, 0.08, transform=ax.transAxes,
                         facecolor=plt.cm.Greens(imp), edgecolor='gray', lw=0.5)
        ax.add_patch(rect)
    
    ax.text(0.5, 0.64, '!=', fontsize=14, ha='center', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    ax.text(0.5, 0.52, 'Hard to debug and understand model decisions',
            fontsize=10, ha='center', transform=ax.transAxes, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Attention Rollout', 'Probing Models',
                 'Saliency Methods', 'Mechanistic Interpretability']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_16(fig, ax):
    """Inefficient for Dense Inputs"""
    ax.text(0.5, 0.95, 'Problem 16: Inefficient for Dense Inputs', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Images/videos create huge token counts',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Image to patches
    ax.text(0.20, 0.73, '224x224', fontsize=9, ha='center', 
            transform=ax.transAxes, color='gray')
    rect = Rectangle((0.12, 0.55), 0.16, 0.16, transform=ax.transAxes,
                     facecolor='#E3F2FD', edgecolor='#1976D2', lw=2)
    ax.add_patch(rect)
    ax.text(0.20, 0.63, 'Image', fontsize=10, ha='center', 
            va='center', transform=ax.transAxes, fontweight='bold')
    
    ax.annotate('', xy=(0.38, 0.63), xytext=(0.32, 0.63),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2),
                transform=ax.transAxes)
    
    # Patches
    for i in range(4):
        for j in range(4):
            rect = Rectangle((0.42 + j*0.045, 0.55 + i*0.045), 0.04, 0.04,
                            transform=ax.transAxes, facecolor='#FFECB3', 
                            edgecolor='#F9A825', lw=0.5)
            ax.add_patch(rect)
    
    ax.text(0.60, 0.73, '16x16 patches', fontsize=9, ha='center', 
            transform=ax.transAxes, color='gray')
    ax.text(0.80, 0.63, '= 196\ntokens!', fontsize=10, ha='center', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    ax.text(0.5, 0.50, 'Video: 1000s of tokens per frame!',
            fontsize=10, ha='center', transform=ax.transAxes, color='#C62828')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Larger Patch Embedding', 'Hierarchical Vision Transformers',
                 'Tubelet Embedding (video)', 'Swin Transformer']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_17(fig, ax):
    """Hardware Inefficiency"""
    ax.text(0.5, 0.95, 'Problem 17: Hardware Inefficiency', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Naive attention is memory-bandwidth bound',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # GPU and Memory diagram
    rect1 = FancyBboxPatch((0.15, 0.58), 0.25, 0.15,
                           boxstyle="round,pad=0.02", transform=ax.transAxes,
                           facecolor='#424242', edgecolor='black', lw=2)
    ax.add_patch(rect1)
    ax.text(0.275, 0.655, 'GPU\nCompute', fontsize=10, ha='center', 
            va='center', transform=ax.transAxes, color='white', fontweight='bold')
    
    rect2 = FancyBboxPatch((0.60, 0.58), 0.25, 0.15,
                           boxstyle="round,pad=0.02", transform=ax.transAxes,
                           facecolor='#1976D2', edgecolor='black', lw=2)
    ax.add_patch(rect2)
    ax.text(0.725, 0.655, 'HBM\nMemory', fontsize=10, ha='center', 
            va='center', transform=ax.transAxes, color='white', fontweight='bold')
    
    ax.annotate('', xy=(0.60, 0.655), xytext=(0.40, 0.655),
                arrowprops=dict(arrowstyle='<->', color='#C62828', lw=3),
                transform=ax.transAxes)
    ax.text(0.50, 0.70, 'Bottleneck!', fontsize=10, ha='center', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    ax.text(0.5, 0.52, 'GPU utilization often < 50%',
            fontsize=10, ha='center', transform=ax.transAxes, color='gray')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['FlashAttention (IO-aware)', 'Fused MHA Kernels',
                 'xFormers library', 'Custom CUDA Kernels']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


def create_problem_18(fig, ax):
    """Real-Time Deployment"""
    ax.text(0.5, 0.95, 'Problem 18: Real-Time Deployment', 
            fontsize=18, fontweight='bold', color=TITLE_COLOR,
            ha='center', va='top', transform=ax.transAxes)
    
    # Problem section
    problem_box = FancyBboxPatch((0.05, 0.45), 0.9, 0.45,
                                  boxstyle="round,pad=0.02", transform=ax.transAxes,
                                  facecolor=PROBLEM_BG, edgecolor=PROBLEM_BORDER, lw=2)
    ax.add_patch(problem_box)
    
    ax.text(0.5, 0.87, 'PROBLEM', fontsize=12, fontweight='bold', color='#E65100',
            ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Too slow for real-time applications',
            fontsize=11, ha='center', transform=ax.transAxes)
    
    # Latency comparison
    ax.text(0.15, 0.70, 'Required:', fontsize=10, ha='left', 
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.35, 0.70, '< 100ms', fontsize=10, ha='left', 
            transform=ax.transAxes, color='#388E3C', fontweight='bold')
    
    ax.text(0.15, 0.63, 'LLM Actual:', fontsize=10, ha='left', 
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.35, 0.63, '500-2000ms', fontsize=10, ha='left', 
            transform=ax.transAxes, color='#C62828', fontweight='bold')
    
    # Use cases
    ax.text(0.70, 0.72, 'Use Cases:', fontsize=9, ha='center', 
            transform=ax.transAxes, fontstyle='italic')
    cases = ['Voice Assistants', 'Gaming AI', 'Autonomous Driving']
    for i, case in enumerate(cases):
        ax.text(0.70, 0.66 - i*0.05, f'- {case}', fontsize=9, ha='center', 
                transform=ax.transAxes, color='gray')
    
    # Solutions section
    solution_box = FancyBboxPatch((0.05, 0.02), 0.9, 0.38,
                                   boxstyle="round,pad=0.02", transform=ax.transAxes,
                                   facecolor=SOLUTION_BG, edgecolor=SOLUTION_BORDER, lw=2)
    ax.add_patch(solution_box)
    
    ax.text(0.5, 0.37, 'SOLUTIONS', fontsize=12, fontweight='bold', color='#2E7D32',
            ha='center', transform=ax.transAxes)
    
    solutions = ['Quantization (INT8/INT4)', 'Pruning',
                 'Distillation (smaller models)', 'ONNX Runtime / TensorRT']
    
    for i, sol in enumerate(solutions):
        ax.text(0.1, 0.30 - i*0.065, f'• {sol}', fontsize=10, transform=ax.transAxes)


# Mapping of problem numbers to functions
PROBLEM_FUNCS = {
    1: create_problem_01, 2: create_problem_02, 3: create_problem_03,
    4: create_problem_04, 5: create_problem_05, 6: create_problem_06,
    7: create_problem_07, 8: create_problem_08, 9: create_problem_09,
    10: create_problem_10, 11: create_problem_11, 12: create_problem_12,
    13: create_problem_13, 14: create_problem_14, 15: create_problem_15,
    16: create_problem_16, 17: create_problem_17, 18: create_problem_18,
}

FOLDERS = {
    1: "01_quadratic_complexity", 2: "02_positional_awareness", 
    3: "03_fixed_context", 4: "04_slow_decoding",
    5: "05_local_bias", 6: "06_data_hungry",
    7: "07_memory_footprint", 8: "08_compute_cost",
    9: "09_length_generalization", 10: "10_training_instability",
    11: "11_attention_smoothing", 12: "12_no_recurrence",
    13: "13_model_size", 14: "14_noise_sensitivity",
    15: "15_interpretability", 16: "16_dense_inputs",
    17: "17_hardware_inefficiency", 18: "18_realtime_deployment",
}


def generate_image(problem_num, output_path):
    """Generate a single problem image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    PROBLEM_FUNCS[problem_num](fig, ax)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()


def main():
    """Generate all 18 problem images."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating Transformer Problems Infographics...")
    print("=" * 50)
    
    for num in range(1, 19):
        folder = FOLDERS[num]
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        output_path = os.path.join(folder_path, 'problem.png')
        
        print(f"  [{num:02d}] Generating {folder}...")
        generate_image(num, output_path)
    
    print("=" * 50)
    print("Done! All 18 images generated.")


if __name__ == "__main__":
    main()
