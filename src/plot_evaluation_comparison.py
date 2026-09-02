import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def main():
    # Data
    labels = ['Pixel-level (0mm)', 'Blob-level (2mm)']
    
    recall = [4.5, 28.3]
    precision = [2.1, 10.2]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use standard colors #1f77b4 (blue) and #ff7f0e (orange) to match other plots roughly
    rects1 = ax.bar(x - width/2, recall, width, label='Recall (%)', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, precision, width, label='Precision (%)', color='#ff7f0e', edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Change by Evaluation Tolerance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 35)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "pixel_vs_blob_comparison.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시스템] 비교 차트가 생성되었습니다: {out_path}")

if __name__ == '__main__':
    main()
