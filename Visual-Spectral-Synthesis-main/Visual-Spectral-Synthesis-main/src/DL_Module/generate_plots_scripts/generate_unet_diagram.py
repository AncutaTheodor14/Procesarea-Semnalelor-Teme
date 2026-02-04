import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("experimental_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def draw_conv_block(ax, x, y, width, height, in_ch, out_ch, size, color, label):
    rect = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.01", 
                          edgecolor='black', 
                          facecolor=color, 
                          linewidth=2)
    ax.add_patch(rect)
    
    # Text
    ax.text(x + width/2, y + height/2 + 0.3, label, 
           ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x + width/2, y + height/2 - 0.1, f'{in_ch}→{out_ch}', 
           ha='center', va='center', fontsize=8)
    ax.text(x + width/2, y + height/2 - 0.4, f'{size}×{size}', 
           ha='center', va='center', fontsize=7, style='italic')

def draw_arrow(ax, x1, y1, x2, y2, style='simple', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', 
                          color=color,
                          linewidth=2,
                          mutation_scale=20)
    ax.add_patch(arrow)

def draw_skip_connection(ax, x1, y1, x2, y2):
    mid_x = (x1 + x2) / 2
    control_y = y1 + 1.5
    
    t = np.linspace(0, 1, 50)
    x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
    y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
    
    ax.plot(x, y, '--', color='#e74c3c', linewidth=2, alpha=0.7)
    ax.plot(x2, y2, 'o', color='#e74c3c', markersize=8)

def generate_unet_diagram():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    encoder_color = '#3498db'  
    decoder_color = '#2ecc71' 
    bottleneck_color = '#9b59b6' 
    final_color = '#f39c12' 
    
    ax.text(8, 11.5, 'Arhitectura PhaseUNet', 
           ha='center', fontsize=16, fontweight='bold')
    
    ax.text(2, 10.5, 'ENCODER', ha='center', fontsize=12, fontweight='bold', color=encoder_color)
    
    ax.text(2, 9.8, 'Input: 128×128×1', ha='center', fontsize=9, style='italic')
    draw_arrow(ax, 2, 9.5, 2, 9.0, color='gray')
    
    e1_x, e1_y = 0.8, 8.0
    draw_conv_block(ax, e1_x, e1_y, 2.4, 1.2, 1, 64, '64×64', encoder_color, 'e1: Conv2D')
    draw_arrow(ax, 2, e1_y - 0.2, 2, e1_y - 0.8, color='black')
    
    e2_x, e2_y = 0.8, 6.2
    draw_conv_block(ax, e2_x, e2_y, 2.4, 1.2, 64, 128, '32×32', encoder_color, 'e2: Conv+BN+LReLU')
    draw_arrow(ax, 2, e2_y - 0.2, 2, e2_y - 0.8, color='black')
    
    e3_x, e3_y = 0.8, 4.4
    draw_conv_block(ax, e3_x, e3_y, 2.4, 1.2, 128, 256, '16×16', encoder_color, 'e3: Conv+BN+LReLU')
    draw_arrow(ax, 2, e3_y - 0.2, 2, e3_y - 0.8, color='black')
    
    e4_x, e4_y = 0.8, 2.6
    draw_conv_block(ax, e4_x, e4_y, 2.4, 1.2, 256, 512, '8×8', bottleneck_color, 'e4: Bottleneck')
    
    ax.text(14, 10.5, 'DECODER', ha='center', fontsize=12, fontweight='bold', color=decoder_color)
    
    d1_x, d1_y = 13.2, 4.4
    draw_arrow(ax, 3.2, 3.2, 13.2, 5.0, color='black')
    draw_conv_block(ax, d1_x, d1_y, 2.4, 1.2, 512, 256, '16×16', decoder_color, 'd1: ConvT+BN+ReLU')
    
    draw_skip_connection(ax, 3.2, 5.0, 13.2, 5.0)
    ax.text(8, 6.8, 'Skip (cat)', ha='center', fontsize=8, color='#e74c3c')
    ax.text(12.5, 5.0, '512 ch', ha='right', fontsize=7, style='italic')
    
    d2_x, d2_y = 13.2, 6.2
    draw_arrow(ax, 14.4, d1_y + 1.2, 14.4, d2_y, color='black')
    draw_conv_block(ax, d2_x, d2_y, 2.4, 1.2, 512, 128, '32×32', decoder_color, 'd2: ConvT+BN+ReLU')
    
    draw_skip_connection(ax, 3.2, 6.8, 13.2, 6.8)
    ax.text(8, 8.5, 'Skip (cat)', ha='center', fontsize=8, color='#e74c3c')
    ax.text(12.5, 6.8, '256 ch', ha='right', fontsize=7, style='italic')
    
    d3_x, d3_y = 13.2, 8.0
    draw_arrow(ax, 14.4, d2_y + 1.2, 14.4, d3_y, color='black')
    draw_conv_block(ax, d3_x, d3_y, 2.4, 1.2, 256, 64, '64×64', decoder_color, 'd3: ConvT+BN+ReLU')
    
    draw_skip_connection(ax, 3.2, 8.6, 13.2, 8.6)
    ax.text(8, 10.0, 'Skip (cat)', ha='center', fontsize=8, color='#e74c3c')
    ax.text(12.5, 8.6, '128 ch', ha='right', fontsize=7, style='italic')
    
    final_x, final_y = 13.2, 0.5
    draw_arrow(ax, 14.4, d3_y - 0.2, 14.4, final_y + 1.2, color='black')
    draw_conv_block(ax, final_x, final_y, 2.4, 1.2, 128, 2, '128×128', final_color, 'Final: ConvT+Tanh')
    
    ax.text(14.4, 0.2, 'Output: [cos(φ), sin(φ)]', ha='center', fontsize=9, 
        style='italic', fontweight='bold')
    ax.text(14.4, -0.2, '128×128×2', ha='center', fontsize=8, style='italic')
    
    legend_y = 1.2
    legend_items = [
        ('Encoder (Down)', encoder_color),
        ('Bottleneck', bottleneck_color),
        ('Decoder (Up)', decoder_color),
        ('Final Layer', final_color)
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x = 5 + i * 2.5
        rect = patches.Rectangle((x, legend_y), 0.3, 0.3, 
                                facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y + 0.15, label, va='center', fontsize=9)
    
    notes = [
        'Skip Connections: Concatenare pe dimensiunea canalelor',
        'Kernel size: 4×4, Stride: 2 (pentru toate conv/deconv)',
        'BatchNorm + Activation după fiecare layer (exceptând final)',
        'L2 Normalization aplicată pe output'
    ]
    
    note_y = 0.3
    for i, note in enumerate(notes):
        ax.text(0.5, note_y - i*0.3, f'• {note}', fontsize=7, style='italic')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "unet_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    print("\n" + "="*60)
    print(" UNET ARCHITECTURE DIAGRAM GENERATION ".center(60))
    print("="*60)
    
    generate_unet_diagram()
    
    print("\n" + "="*60)
    print(" ✓ ARCHITECTURE DIAGRAM COMPLETE ".center(60))
    print("="*60)

if __name__ == "__main__":
    main()