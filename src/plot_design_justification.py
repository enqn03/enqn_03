import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.font_manager as fm
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('outputs', exist_ok=True)
k_values = [1, 2, 4, 8]
f1_scores = [0.15, 0.28, 0.435, 0.35]
vram_usage = [3, 5, 8.5, 17]
fig, ax1 = plt.subplots(figsize=(8, 5))
color_bar = '#4c72b0'
ax1.bar(k_values, f1_scores, color=color_bar, alpha=0.7, width=1.5, label='F1-Score')
ax1.set_xlabel('Temporal History (K Frames)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1-Score', color=color_bar, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_bar)
ax1.set_xticks(k_values)
ax1.set_ylim(0, 0.5)
ax2 = ax1.twinx()
color_line = '#c44e52'
ax2.plot(k_values, vram_usage, color=color_line, marker='o', linewidth=2.5, markersize=8, label='VRAM Usage (GB)')
ax2.set_ylabel('VRAM Usage (GB)', color=color_line, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_line)
ax2.set_ylim(0, 20)
ax2.axhline(y=16, color='red', linestyle='--', alpha=0.6)
ax2.text(1, 16.5, '16GB VRAM Limit (OOM Risk)', color='red', fontsize=10)
plt.title('Performance vs. Memory Trade-off by Temporal History (K)', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('outputs/k_history_tradeoff.png', dpi=300)
plt.close()
models = ['Simple Concat', 'Gated CBAM Fusion']
fp_counts = [523, 111]
precisions = [0.08, 0.44]
fig, ax1 = plt.subplots(figsize=(8, 5))
color_bar2 = '#dd8452'
bars = ax1.bar(models, fp_counts, color=color_bar2, alpha=0.8, width=0.5, label='False Positives')
ax1.set_ylabel('False Positives (Count)', color=color_bar2, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_bar2)
ax1.set_ylim(0, 600)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{yval}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color_bar2)
ax2 = ax1.twinx()
color_line2 = '#55a868'
ax2.plot(models, precisions, color=color_line2, marker='D', linewidth=3, markersize=10, label='Precision')
ax2.set_ylabel('Precision', color=color_line2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_line2)
ax2.set_ylim(0, 0.5)
for i, v in enumerate(precisions):
    ax2.text(i + 0.1, v + 0.02, f'{v*100:.1f}%', fontweight='bold', color=color_line2)
plt.title('Impact of CBAM Gate on Noise Reduction and Precision', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('outputs/cbam_fusion_comparison.png', dpi=300)
plt.close()
print("Saved outputs/k_history_tradeoff.png")
print("Saved outputs/cbam_fusion_comparison.png")
