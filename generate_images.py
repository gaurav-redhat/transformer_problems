#!/usr/bin/env python3
"""
Generate infographic images for all 18 Transformer Problems.
Each image shows the problem and its solutions in a clean card format.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# Define all 18 transformer problems with their content
PROBLEMS = [
    {
        "num": 1,
        "folder": "01_quadratic_complexity",
        "title": "Quadratic Complexity O(N²)",
        "problem": "Self-attention computes N×N matrix, causing\nmemory and compute explosion for long sequences.",
        "solutions": ["Sparse Attention", "Longformer", "Linformer", "Performer", "FlashAttention", "Transformer-XL"]
    },
    {
        "num": 2,
        "folder": "02_positional_awareness",
        "title": "No Positional Awareness",
        "problem": "Transformer has no inherent notion of token\norder. 'Dog bites man' = 'Man bites dog'",
        "solutions": ["Sinusoidal Positional Encoding", "Learnable PE", "Relative PE", "RoPE", "ALiBi"]
    },
    {
        "num": 3,
        "folder": "03_fixed_context",
        "title": "Fixed Context Window",
        "problem": "Cannot remember information beyond maximum\nsequence length. Past is forgotten.",
        "solutions": ["Transformer-XL", "Retrieval-Augmented Generation (RAG)", "External Memory", "Mamba"]
    },
    {
        "num": 4,
        "folder": "04_slow_decoding",
        "title": "Slow Autoregressive Decoding",
        "problem": "Decoder generates one token at a time causing\nhigh latency. Sequential bottleneck.",
        "solutions": ["KV Cache", "Speculative Decoding", "Non-Autoregressive Models", "Distillation"]
    },
    {
        "num": 5,
        "folder": "05_local_bias",
        "title": "No Local Inductive Bias",
        "problem": "No bias toward local patterns. Hurts vision and\naudio tasks where locality matters.",
        "solutions": ["CNN + Transformer", "Swin Transformer", "Conformer", "Hierarchical ViT"]
    },
    {
        "num": 6,
        "folder": "06_data_hungry",
        "title": "Data-Hungry Architecture",
        "problem": "Requires massive datasets to generalize well.\nSmall data = poor performance.",
        "solutions": ["Self-Supervised Pretraining", "Transfer Learning", "Fine-Tuning", "Knowledge Distillation"]
    },
    {
        "num": 7,
        "folder": "07_memory_footprint",
        "title": "High Memory Footprint",
        "problem": "Q, K, V tensors and KV cache consume large\nmemory. GPU OOM errors common.",
        "solutions": ["FlashAttention", "KV Cache Optimization", "Gradient Checkpointing", "Memory-Efficient Attention"]
    },
    {
        "num": 8,
        "folder": "08_compute_cost",
        "title": "High Compute & Power Cost",
        "problem": "Expensive FLOPs. Not edge-friendly. High\nelectricity bills for training.",
        "solutions": ["Quantization (INT8/INT4)", "Lightweight Transformers", "Operator Fusion", "Pruning"]
    },
    {
        "num": 9,
        "folder": "09_length_generalization",
        "title": "Poor Length Generalization",
        "problem": "Fails when sequence length exceeds training\nlength. Positional encodings break.",
        "solutions": ["Relative Position Encoding", "RoPE Scaling", "ALiBi", "Length Extrapolation"]
    },
    {
        "num": 10,
        "folder": "10_training_instability",
        "title": "Training Instability",
        "problem": "Gradient explosion/vanishing in deep\nTransformers. Training diverges or stalls.",
        "solutions": ["Pre-LayerNorm", "Learning Rate Warmup", "AdamW Optimizer", "Gradient Clipping"]
    },
    {
        "num": 11,
        "folder": "11_attention_smoothing",
        "title": "Attention Over-Smoothing",
        "problem": "Token representations become too similar in deep\nlayers. Loss of information.",
        "solutions": ["Residual Scaling", "DropHead", "Attention Temperature Control", "Skip Connections"]
    },
    {
        "num": 12,
        "folder": "12_no_recurrence",
        "title": "No Recurrence / Streaming",
        "problem": "Hard to use for streaming or online inference.\nMust process full sequence.",
        "solutions": ["Transformer-XL", "Chunk-based Attention", "Streaming Transformers", "Delta Attention"]
    },
    {
        "num": 13,
        "folder": "13_model_size",
        "title": "Large Model Size",
        "problem": "Too many parameters. Deployment difficulty on\nedge devices.",
        "solutions": ["LoRA", "QLoRA", "Parameter Sharing (ALBERT)", "Pruning", "Distillation"]
    },
    {
        "num": 14,
        "folder": "14_noise_sensitivity",
        "title": "Sensitivity to Noise Tokens",
        "problem": "Global attention attends to irrelevant tokens.\nWastes capacity on noise.",
        "solutions": ["Sparse Attention", "Token Pruning", "Attention Masking", "Gating Mechanisms"]
    },
    {
        "num": 15,
        "folder": "15_interpretability",
        "title": "Poor Interpretability",
        "problem": "Attention weights are not true explanations. Hard\nto debug and understand.",
        "solutions": ["Attention Rollout", "Probing Models", "Saliency Methods", "Mechanistic Interpretability"]
    },
    {
        "num": 16,
        "folder": "16_dense_inputs",
        "title": "Inefficient for Dense Inputs",
        "problem": "Flattening images/videos creates huge token\ncounts. Vision = many patches.",
        "solutions": ["Patch Embedding", "Hierarchical Vision Transformers", "Tubelet Embedding", "Swin"]
    },
    {
        "num": 17,
        "folder": "17_hardware_inefficiency",
        "title": "Hardware Inefficiency",
        "problem": "Naive attention is memory-bandwidth bound. Poor\nGPU utilization.",
        "solutions": ["FlashAttention", "Fused MHA Kernels", "xFormers", "Custom CUDA Kernels"]
    },
    {
        "num": 18,
        "folder": "18_realtime_deployment",
        "title": "Real-Time Deployment",
        "problem": "Latency and memory constraints in production. Too\nslow for real-time apps.",
        "solutions": ["Quantization", "Pruning", "Distillation", "Edge-Optimized Transformers", "ONNX"]
    },
]


def create_infographic(problem_data, output_path):
    """Create a single infographic image for a transformer problem."""
    
    # Figure setup
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Colors
    title_color = '#1565C0'  # Blue
    problem_header_color = '#C62828'  # Red
    problem_bg = '#FFEBEE'  # Light red
    problem_border = '#C62828'  # Red border
    solution_header_color = '#2E7D32'  # Green
    solution_bg = '#E8F5E9'  # Light green
    solution_border = '#2E7D32'  # Green border
    text_color = '#212121'  # Dark gray
    
    # Title
    title_text = f"Problem {problem_data['num']}: {problem_data['title']}"
    ax.text(5, 9.3, title_text, fontsize=22, fontweight='bold', 
            color=title_color, ha='center', va='center',
            fontfamily='DejaVu Sans')
    
    # Problem Box
    problem_box = FancyBboxPatch(
        (0.5, 5.5), 9, 3.2,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=problem_bg,
        edgecolor=problem_border,
        linewidth=2.5
    )
    ax.add_patch(problem_box)
    
    # Problem Header
    ax.text(5, 8.2, "Problem", fontsize=18, fontweight='bold',
            color=problem_header_color, ha='center', va='center',
            fontfamily='DejaVu Sans')
    
    # Problem Description
    ax.text(5, 6.9, problem_data['problem'], fontsize=14,
            color=text_color, ha='center', va='center',
            fontfamily='DejaVu Sans', linespacing=1.4)
    
    # Calculate solution box height based on number of solutions
    num_solutions = len(problem_data['solutions'])
    solution_box_height = max(3.5, 1.5 + num_solutions * 0.55)
    solution_box_y = 0.5
    
    # Solution Box
    solution_box = FancyBboxPatch(
        (0.5, solution_box_y), 9, solution_box_height,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=solution_bg,
        edgecolor=solution_border,
        linewidth=2.5
    )
    ax.add_patch(solution_box)
    
    # Solutions Header
    header_y = solution_box_y + solution_box_height - 0.6
    ax.text(5, header_y, "Solutions", fontsize=18, fontweight='bold',
            color=solution_header_color, ha='center', va='center',
            fontfamily='DejaVu Sans')
    
    # Solution Items
    start_y = header_y - 0.7
    for i, solution in enumerate(problem_data['solutions']):
        y_pos = start_y - (i * 0.55)
        ax.text(1.2, y_pos, f"• {solution}", fontsize=14,
                color=text_color, ha='left', va='center',
                fontfamily='DejaVu Sans')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                pad_inches=0.3)
    plt.close()


def main():
    """Generate all infographic images."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🎨 Generating Transformer Problems Infographics...")
    print("=" * 50)
    
    for problem in PROBLEMS:
        folder_path = os.path.join(base_dir, problem['folder'])
        
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        output_path = os.path.join(folder_path, 'problem.png')
        
        print(f"  [{problem['num']:02d}] {problem['title']}")
        create_infographic(problem, output_path)
    
    print("=" * 50)
    print("✅ All 18 infographics generated successfully!")
    print(f"📁 Output location: {base_dir}")


if __name__ == "__main__":
    main()

