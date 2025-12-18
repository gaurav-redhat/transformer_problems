import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

def create_vanilla_transformer_diagram():
    """Create Vanilla Transformer architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    encoder_color = '#3b82f6'
    decoder_color = '#22c55e'
    attention_color = '#f59e0b'
    ffn_color = '#8b5cf6'
    
    # Title
    ax.text(7, 9.5, 'Vanilla Transformer Architecture', fontsize=18, fontweight='bold', ha='center')
    ax.text(7, 9.0, '"Attention Is All You Need" (2017)', fontsize=12, ha='center', style='italic')
    
    # Encoder (left)
    encoder_box = FancyBboxPatch((1, 2), 4, 6, boxstyle="round,pad=0.1", 
                                  facecolor=encoder_color, alpha=0.2, edgecolor=encoder_color, linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(3, 7.7, 'ENCODER', fontsize=14, fontweight='bold', ha='center', color=encoder_color)
    
    # Encoder components
    ax.add_patch(FancyBboxPatch((1.5, 6.5), 3, 0.8, boxstyle="round", facecolor=attention_color, alpha=0.8))
    ax.text(3, 6.9, 'Multi-Head\nSelf-Attention', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((1.5, 5.2), 3, 0.8, boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax.text(3, 5.6, 'Add & Norm', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((1.5, 4.0), 3, 0.8, boxstyle="round", facecolor=ffn_color, alpha=0.8))
    ax.text(3, 4.4, 'Feed Forward', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((1.5, 2.8), 3, 0.8, boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax.text(3, 3.2, 'Add & Norm', fontsize=9, ha='center', va='center')
    
    ax.text(0.5, 4.5, 'N×', fontsize=16, fontweight='bold', color=encoder_color)
    
    # Decoder (right)
    decoder_box = FancyBboxPatch((9, 1), 4, 7.5, boxstyle="round,pad=0.1",
                                  facecolor=decoder_color, alpha=0.2, edgecolor=decoder_color, linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(11, 8.2, 'DECODER', fontsize=14, fontweight='bold', ha='center', color=decoder_color)
    
    # Decoder components
    ax.add_patch(FancyBboxPatch((9.5, 7.0), 3, 0.8, boxstyle="round", facecolor=attention_color, alpha=0.8))
    ax.text(11, 7.4, 'Masked\nSelf-Attention', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((9.5, 5.8), 3, 0.8, boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax.text(11, 6.2, 'Add & Norm', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((9.5, 4.6), 3, 0.8, boxstyle="round", facecolor='#ef4444', alpha=0.8))
    ax.text(11, 5.0, 'Cross-Attention', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((9.5, 3.4), 3, 0.8, boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax.text(11, 3.8, 'Add & Norm', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((9.5, 2.2), 3, 0.8, boxstyle="round", facecolor=ffn_color, alpha=0.8))
    ax.text(11, 2.6, 'Feed Forward', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((9.5, 1.2), 3, 0.8, boxstyle="round", facecolor='lightgray', alpha=0.8))
    ax.text(11, 1.6, 'Add & Norm', fontsize=9, ha='center', va='center')
    
    ax.text(8.5, 4.5, 'N×', fontsize=16, fontweight='bold', color=decoder_color)
    
    # Arrow from encoder to decoder
    ax.annotate('', xy=(9.5, 5.0), xytext=(5.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(7.5, 5.3, 'K, V', fontsize=10, ha='center')
    
    # Input/Output
    ax.add_patch(FancyBboxPatch((1.5, 0.5), 3, 0.8, boxstyle="round", facecolor='white', edgecolor='black'))
    ax.text(3, 0.9, 'Input Embedding\n+ Positional Enc', fontsize=9, ha='center', va='center')
    
    ax.add_patch(FancyBboxPatch((9.5, 0.0), 3, 0.8, boxstyle="round", facecolor='white', edgecolor='black'))
    ax.text(11, 0.4, 'Output Embedding\n+ Positional Enc', fontsize=9, ha='center', va='center')
    
    # Legend
    legend_items = [
        (attention_color, 'Self-Attention'),
        ('#ef4444', 'Cross-Attention'),
        (ffn_color, 'Feed Forward'),
        ('lightgray', 'Add & LayerNorm'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.5, 9.2-i*0.4), 0.3, 0.25, facecolor=color, alpha=0.8))
        ax.text(1.0, 9.3-i*0.4, label, fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig('01_vanilla_transformer/architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('Created: 01_vanilla_transformer/architecture.png')

def create_bert_diagram():
    """Create BERT architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'BERT: Bidirectional Encoder', fontsize=18, fontweight='bold', ha='center')
    ax.text(6, 7.0, 'Encoder-only, Bidirectional Context', fontsize=12, ha='center', style='italic')
    
    # Input tokens
    tokens = ['[CLS]', 'The', 'cat', '[MASK]', 'on', 'mat', '[SEP]']
    for i, token in enumerate(tokens):
        x = 1.5 + i * 1.3
        ax.add_patch(FancyBboxPatch((x-0.5, 0.5), 1, 0.6, boxstyle="round", facecolor='#e0e7ff', edgecolor='#3b82f6'))
        ax.text(x, 0.8, token, fontsize=10, ha='center', va='center')
    
    # Embedding layer
    ax.add_patch(FancyBboxPatch((1, 1.5), 9.5, 0.8, boxstyle="round", facecolor='#fef3c7', edgecolor='#f59e0b'))
    ax.text(5.75, 1.9, 'Token + Segment + Position Embeddings', fontsize=11, ha='center', va='center')
    
    # Transformer layers
    for layer in range(3):
        y = 2.8 + layer * 1.2
        color = '#dbeafe' if layer % 2 == 0 else '#dcfce7'
        ax.add_patch(FancyBboxPatch((1, y), 9.5, 1.0, boxstyle="round", facecolor=color, edgecolor='#3b82f6'))
        ax.text(5.75, y+0.5, f'Transformer Block {layer+1} (Self-Attention + FFN)', fontsize=10, ha='center', va='center')
    
    # Arrows showing bidirectional attention
    ax.annotate('', xy=(3, 3.3), xytext=(7, 3.3), arrowprops=dict(arrowstyle='<->', color='#ef4444', lw=2))
    ax.text(5, 3.6, 'Bidirectional Attention', fontsize=9, ha='center', color='#ef4444')
    
    # Output
    ax.add_patch(FancyBboxPatch((1, 6.2), 9.5, 0.6, boxstyle="round", facecolor='#f3e8ff', edgecolor='#8b5cf6'))
    ax.text(5.75, 6.5, 'Output: [CLS] for classification, token embeddings for NER/QA', fontsize=10, ha='center', va='center')
    
    # Key features box
    ax.add_patch(FancyBboxPatch((0.3, 0), 3.5, 0.4, boxstyle="round", facecolor='#fef3c7'))
    ax.text(2.05, 0.2, 'MLM + NSP Pretraining', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('02_bert/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 02_bert/architecture.png')

def create_gpt_diagram():
    """Create GPT architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'GPT: Autoregressive Decoder', fontsize=18, fontweight='bold', ha='center')
    ax.text(6, 7.0, 'Decoder-only, Causal (Left-to-Right) Attention', fontsize=12, ha='center', style='italic')
    
    # Input tokens
    tokens = ['The', 'quick', 'brown', 'fox', '???']
    for i, token in enumerate(tokens):
        x = 2 + i * 1.8
        ax.add_patch(FancyBboxPatch((x-0.7, 0.5), 1.4, 0.6, boxstyle="round", facecolor='#dcfce7', edgecolor='#22c55e'))
        ax.text(x, 0.8, token, fontsize=11, ha='center', va='center')
    
    # Causal mask visualization
    ax.text(6, 1.4, 'Causal Mask (Can only see past tokens)', fontsize=10, ha='center', color='#ef4444')
    
    # Decoder blocks
    for layer in range(3):
        y = 2.0 + layer * 1.4
        ax.add_patch(FancyBboxPatch((1.5, y), 9, 1.2, boxstyle="round", facecolor='#fef3c7', edgecolor='#f59e0b'))
        ax.text(6, y+0.6, f'Decoder Block {layer+1}: Masked Self-Attention → FFN', fontsize=10, ha='center', va='center')
    
    # Arrows showing causal flow
    for i in range(4):
        x = 2 + i * 1.8
        ax.annotate('', xy=(x+0.9, 2.2), xytext=(x+0.9, 1.1), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    # Output prediction
    ax.add_patch(FancyBboxPatch((1.5, 6.2), 9, 0.6, boxstyle="round", facecolor='#fee2e2', edgecolor='#ef4444'))
    ax.text(6, 6.5, 'Next Token Prediction: P(next | previous tokens)', fontsize=10, ha='center', va='center')
    
    # Causal mask illustration (small)
    mask_x, mask_y = 10, 1
    mask_size = 0.3
    for i in range(4):
        for j in range(4):
            if j <= i:  # Lower triangle (can attend)
                color = '#22c55e'
            else:  # Upper triangle (masked)
                color = '#ef4444'
            ax.add_patch(plt.Rectangle((mask_x + j*mask_size, mask_y + (3-i)*mask_size), 
                                       mask_size, mask_size, facecolor=color, alpha=0.7))
    ax.text(10.6, 0.6, 'Causal\nMask', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig('03_gpt/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 03_gpt/architecture.png')

def create_vit_diagram():
    """Create Vision Transformer diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Vision Transformer (ViT)', fontsize=18, fontweight='bold', ha='center')
    ax.text(7, 7.0, 'Image Patches as Tokens', fontsize=12, ha='center', style='italic')
    
    # Input image
    ax.add_patch(plt.Rectangle((0.5, 3), 2, 2, facecolor='#dbeafe', edgecolor='#3b82f6', linewidth=2))
    # Grid lines for patches
    for i in range(1, 4):
        ax.plot([0.5 + i*0.5, 0.5 + i*0.5], [3, 5], color='#3b82f6', linewidth=1)
        ax.plot([0.5, 2.5], [3 + i*0.5, 3 + i*0.5], color='#3b82f6', linewidth=1)
    ax.text(1.5, 2.5, 'Image\n(224×224)', fontsize=10, ha='center')
    
    # Arrow
    ax.annotate('', xy=(3.5, 4), xytext=(2.7, 4), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(3.1, 4.4, 'Split into\n16×16 patches', fontsize=9, ha='center')
    
    # Patch embeddings
    patches = ['[CLS]', 'P1', 'P2', 'P3', '...', 'P196']
    for i, p in enumerate(patches):
        x = 4 + i * 1.2
        color = '#f59e0b' if p == '[CLS]' else '#22c55e'
        ax.add_patch(FancyBboxPatch((x-0.4, 3.5), 0.8, 1, boxstyle="round", facecolor=color, alpha=0.7))
        ax.text(x, 4, p, fontsize=9, ha='center', va='center')
    
    # Position embeddings
    ax.add_patch(FancyBboxPatch((4, 2.3), 7, 0.8, boxstyle="round", facecolor='#fef3c7', edgecolor='#f59e0b'))
    ax.text(7.5, 2.7, 'Patch Embedding + Position Embedding', fontsize=10, ha='center', va='center')
    
    # Transformer encoder
    ax.add_patch(FancyBboxPatch((4, 5), 7, 1.5, boxstyle="round", facecolor='#dbeafe', edgecolor='#3b82f6', linewidth=2))
    ax.text(7.5, 5.75, 'Transformer Encoder (L layers)\nMulti-Head Attention + MLP', fontsize=11, ha='center', va='center')
    
    # Output
    ax.add_patch(FancyBboxPatch((6, 0.5), 3, 1, boxstyle="round", facecolor='#f3e8ff', edgecolor='#8b5cf6'))
    ax.text(7.5, 1, '[CLS] → MLP Head\nImage Classification', fontsize=10, ha='center', va='center')
    
    # Arrow from encoder to output
    ax.annotate('', xy=(7.5, 1.5), xytext=(7.5, 4.9), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    plt.savefig('04_vision_transformer/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 04_vision_transformer/architecture.png')

def create_sparse_transformer_diagram():
    """Create Sparse Transformer attention pattern diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    fig.suptitle('Sparse Transformer: Attention Patterns', fontsize=16, fontweight='bold')
    
    n = 16
    
    # Full attention
    full = np.ones((n, n))
    axes[0].imshow(full, cmap='Blues', aspect='equal')
    axes[0].set_title(f'Full Attention\nO(N²) = {n*n}', fontsize=12)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # Strided attention
    strided = np.zeros((n, n))
    stride = 4
    for i in range(n):
        strided[i, ::stride] = 1
        for j in range(max(0, i-2), min(n, i+3)):
            strided[i, j] = 1
    axes[1].imshow(strided, cmap='Greens', aspect='equal')
    axes[1].set_title(f'Strided Pattern\nO(N√N)', fontsize=12)
    axes[1].set_xlabel('Key Position')
    
    # Fixed attention
    fixed = np.zeros((n, n))
    block_size = 4
    for i in range(n):
        block_start = (i // block_size) * block_size
        fixed[i, block_start:block_start+block_size] = 1
        fixed[i, :block_size] = 1  # Global tokens
    axes[2].imshow(fixed, cmap='Oranges', aspect='equal')
    axes[2].set_title(f'Fixed Pattern\n(Block + Global)', fontsize=12)
    axes[2].set_xlabel('Key Position')
    
    plt.tight_layout()
    plt.savefig('06_sparse_transformer/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 06_sparse_transformer/architecture.png')

def create_performer_diagram():
    """Create Performer (FAVOR+) diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.5, 'Performer: Linear Attention via FAVOR+', fontsize=16, fontweight='bold', ha='center')
    
    # Standard attention
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 5, 2.5, boxstyle="round", facecolor='#fee2e2', edgecolor='#ef4444', linewidth=2))
    ax.text(3, 4.5, 'Standard Attention', fontsize=12, fontweight='bold', ha='center', color='#ef4444')
    ax.text(3, 3.8, 'softmax(QKᵀ/√d) × V', fontsize=11, ha='center', family='monospace')
    ax.text(3, 3.2, 'O(N²) time & memory', fontsize=10, ha='center')
    ax.text(3, 2.7, 'N×N attention matrix', fontsize=10, ha='center')
    
    # Arrow
    ax.annotate('', xy=(6.5, 3.75), xytext=(5.7, 3.75), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(6.1, 4.1, 'FAVOR+', fontsize=10, ha='center', fontweight='bold')
    
    # Linear attention
    ax.add_patch(FancyBboxPatch((6.5, 2.5), 5, 2.5, boxstyle="round", facecolor='#dcfce7', edgecolor='#22c55e', linewidth=2))
    ax.text(9, 4.5, 'Linear Attention', fontsize=12, fontweight='bold', ha='center', color='#22c55e')
    ax.text(9, 3.8, 'φ(Q) × (φ(K)ᵀV)', fontsize=11, ha='center', family='monospace')
    ax.text(9, 3.2, 'O(N) time & memory', fontsize=10, ha='center')
    ax.text(9, 2.7, 'Random feature maps φ', fontsize=10, ha='center')
    
    # Complexity comparison
    ax.text(6, 1.5, 'Key Insight: softmax ≈ kernel → decompose as φ(Q)φ(K)ᵀ', fontsize=11, ha='center')
    ax.text(6, 0.8, 'φ uses random orthogonal features for unbiased estimation', fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('07_performer/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 07_performer/architecture.png')

def create_longformer_diagram():
    """Create Longformer attention pattern diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Longformer: Local + Global Attention', fontsize=16, fontweight='bold', ha='center')
    
    # Attention matrix visualization
    n = 20
    attention = np.zeros((n, n))
    
    # Local attention (sliding window)
    window = 3
    for i in range(n):
        for j in range(max(0, i-window), min(n, i+window+1)):
            attention[i, j] = 0.5
    
    # Global tokens (first 2 tokens attend to all)
    attention[:2, :] = 1
    attention[:, :2] = 1
    
    # Plot
    im = ax.imshow(attention, cmap='YlOrRd', extent=[1, 9, 1, 6], aspect='auto')
    ax.text(5, 6.3, 'Attention Pattern', fontsize=12, ha='center')
    
    # Labels
    ax.text(5, 0.5, 'Key Position →', fontsize=10, ha='center')
    ax.text(0.5, 3.5, 'Query\nPosition\n↓', fontsize=10, ha='center', va='center')
    
    # Legend
    ax.add_patch(plt.Rectangle((7.5, 6.8), 0.3, 0.3, facecolor='#fef3c7'))
    ax.text(8, 6.95, 'Local (sliding window)', fontsize=9, va='center')
    ax.add_patch(plt.Rectangle((7.5, 6.3), 0.3, 0.3, facecolor='#ef4444'))
    ax.text(8, 6.45, 'Global ([CLS] tokens)', fontsize=9, va='center')
    
    # Complexity
    ax.text(5, 0, 'Complexity: O(N × window) + O(G × N) = O(N) for fixed window', fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('09_longformer/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 09_longformer/architecture.png')

def create_switch_transformer_diagram():
    """Create Switch Transformer (MoE) diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Switch Transformer: Mixture of Experts', fontsize=16, fontweight='bold', ha='center')
    ax.text(6, 7.0, 'Scale to Trillion Parameters with Sparse Activation', fontsize=11, ha='center', style='italic')
    
    # Input
    ax.add_patch(FancyBboxPatch((4.5, 0.5), 3, 0.8, boxstyle="round", facecolor='#dbeafe', edgecolor='#3b82f6'))
    ax.text(6, 0.9, 'Input Token', fontsize=10, ha='center', va='center')
    
    # Router
    ax.add_patch(FancyBboxPatch((4.5, 1.8), 3, 0.8, boxstyle="round", facecolor='#fef3c7', edgecolor='#f59e0b'))
    ax.text(6, 2.2, 'Router (Softmax)', fontsize=10, ha='center', va='center')
    ax.annotate('', xy=(6, 1.8), xytext=(6, 1.3), arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Experts
    experts = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
    colors = ['#dcfce7', '#fee2e2', '#f3e8ff', '#fef3c7']
    x_positions = [1.5, 4, 8, 10.5]
    
    for i, (expert, color, x) in enumerate(zip(experts, colors, x_positions)):
        ax.add_patch(FancyBboxPatch((x-1, 3.5), 2, 1.5, boxstyle="round", facecolor=color, edgecolor='gray'))
        ax.text(x, 4.25, expert, fontsize=10, ha='center', va='center')
        ax.text(x, 3.8, 'FFN', fontsize=9, ha='center', va='center', style='italic')
        
        # Arrow from router
        if i == 1:  # Highlight selected expert
            ax.annotate('', xy=(x, 3.5), xytext=(6, 2.6), 
                       arrowprops=dict(arrowstyle='->', lw=2, color='#22c55e'))
        else:
            ax.annotate('', xy=(x, 3.5), xytext=(6, 2.6), 
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.3))
    
    # Selected indicator
    ax.add_patch(plt.Rectangle((3, 3.5), 2, 1.5, fill=False, edgecolor='#22c55e', linewidth=3))
    ax.text(4, 5.2, 'Selected!', fontsize=10, ha='center', color='#22c55e', fontweight='bold')
    
    # Output
    ax.add_patch(FancyBboxPatch((4.5, 5.8), 3, 0.8, boxstyle="round", facecolor='#dbeafe', edgecolor='#3b82f6'))
    ax.text(6, 6.2, 'Output Token', fontsize=10, ha='center', va='center')
    ax.annotate('', xy=(6, 5.8), xytext=(4, 5), arrowprops=dict(arrowstyle='->', lw=2, color='#22c55e'))
    
    # Key insight
    ax.text(6, -0.2, 'Key: Each token routed to ONE expert → Sparse activation → Efficient scaling', 
            fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('10_switch_transformer/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 10_switch_transformer/architecture.png')

def create_transformer_xl_diagram():
    """Create Transformer-XL diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(6, 6.5, 'Transformer-XL: Segment-Level Recurrence', fontsize=16, fontweight='bold', ha='center')
    
    # Segment 1
    ax.add_patch(FancyBboxPatch((0.5, 2), 5, 3.5, boxstyle="round", facecolor='#dbeafe', alpha=0.5, edgecolor='#3b82f6', linewidth=2))
    ax.text(3, 5.2, 'Segment t-1', fontsize=12, fontweight='bold', ha='center', color='#3b82f6')
    
    ax.add_patch(FancyBboxPatch((1, 4), 4, 0.8, boxstyle="round", facecolor='#3b82f6', alpha=0.7))
    ax.text(3, 4.4, 'Hidden States', fontsize=10, ha='center', color='white')
    
    ax.add_patch(FancyBboxPatch((1, 2.5), 4, 0.8, boxstyle="round", facecolor='lightgray'))
    ax.text(3, 2.9, 'Tokens [t-L : t-1]', fontsize=10, ha='center')
    
    # Arrow for memory
    ax.annotate('', xy=(6.5, 4.4), xytext=(5.5, 4.4), 
               arrowprops=dict(arrowstyle='->', lw=2, color='#ef4444'))
    ax.text(6, 4.8, 'Cache as\nMemory', fontsize=9, ha='center', color='#ef4444')
    
    # Segment 2
    ax.add_patch(FancyBboxPatch((6.5, 2), 5, 3.5, boxstyle="round", facecolor='#dcfce7', alpha=0.5, edgecolor='#22c55e', linewidth=2))
    ax.text(9, 5.2, 'Segment t', fontsize=12, fontweight='bold', ha='center', color='#22c55e')
    
    ax.add_patch(FancyBboxPatch((7, 4), 4, 0.8, boxstyle="round", facecolor='#22c55e', alpha=0.7))
    ax.text(9, 4.4, 'Hidden States', fontsize=10, ha='center', color='white')
    
    ax.add_patch(FancyBboxPatch((7, 2.5), 4, 0.8, boxstyle="round", facecolor='lightgray'))
    ax.text(9, 2.9, 'Tokens [t : t+L-1]', fontsize=10, ha='center')
    
    # Key insight
    ax.text(6, 1, 'Key: Reuse hidden states from previous segment as extended context', fontsize=11, ha='center')
    ax.text(6, 0.4, 'Enables modeling very long sequences without recomputation', fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('05_transformer_xl/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 05_transformer_xl/architecture.png')

def create_reformer_diagram():
    """Create Reformer diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(6, 6.5, 'Reformer: LSH Attention + Reversible Layers', fontsize=16, fontweight='bold', ha='center')
    
    # LSH Attention
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 5, 3, boxstyle="round", facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=2))
    ax.text(3, 5.2, 'LSH Attention', fontsize=12, fontweight='bold', ha='center', color='#f59e0b')
    ax.text(3, 4.5, '1. Hash Q, K into buckets', fontsize=10, ha='center')
    ax.text(3, 4.0, '2. Sort by bucket', fontsize=10, ha='center')
    ax.text(3, 3.5, '3. Attend within buckets', fontsize=10, ha='center')
    ax.text(3, 2.9, 'O(N log N) complexity', fontsize=10, ha='center', style='italic')
    
    # Reversible Layers
    ax.add_patch(FancyBboxPatch((6.5, 2.5), 5, 3, boxstyle="round", facecolor='#f3e8ff', edgecolor='#8b5cf6', linewidth=2))
    ax.text(9, 5.2, 'Reversible Layers', fontsize=12, fontweight='bold', ha='center', color='#8b5cf6')
    ax.text(9, 4.5, 'Y₁ = X₁ + Attention(X₂)', fontsize=10, ha='center', family='monospace')
    ax.text(9, 4.0, 'Y₂ = X₂ + FFN(Y₁)', fontsize=10, ha='center', family='monospace')
    ax.text(9, 3.3, 'Can recover activations', fontsize=10, ha='center')
    ax.text(9, 2.9, 'No need to store!', fontsize=10, ha='center', style='italic')
    
    # Benefits
    ax.text(6, 1.5, 'Combined Benefits:', fontsize=11, ha='center', fontweight='bold')
    ax.text(6, 0.9, '• LSH: O(N log N) attention instead of O(N²)', fontsize=10, ha='center')
    ax.text(6, 0.4, '• Reversible: O(1) memory for activations instead of O(L)', fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('08_reformer/architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: 08_reformer/architecture.png')

def create_banner():
    """Create main banner image."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Background gradient effect
    for i in range(100):
        ax.axhspan(i/25, (i+1)/25, color=plt.cm.Blues(0.1 + i/150), alpha=0.5)
    
    # Title
    ax.text(7, 2.5, '🤖 Transformer Architectures', fontsize=28, fontweight='bold', 
            ha='center', color='#1e3a5f')
    ax.text(7, 1.5, 'From Vanilla to Vision, BERT to Switch', fontsize=16, 
            ha='center', color='#3b82f6')
    ax.text(7, 0.8, '10 Implementations • Diagrams • Training Code • Google Colab', 
            fontsize=12, ha='center', color='#64748b')
    
    plt.tight_layout()
    plt.savefig('banner.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: banner.png')

if __name__ == '__main__':
    create_banner()
    create_vanilla_transformer_diagram()
    create_bert_diagram()
    create_gpt_diagram()
    create_vit_diagram()
    create_transformer_xl_diagram()
    create_sparse_transformer_diagram()
    create_performer_diagram()
    create_reformer_diagram()
    create_longformer_diagram()
    create_switch_transformer_diagram()
    print('\nAll diagrams created!')
