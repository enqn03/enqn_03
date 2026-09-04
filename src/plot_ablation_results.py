utf-8
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
def main():
    labels = ['A-only (Powder)', 'B-only (Melt Pool)', 'A+B Fusion (Proposed)']
    precision = [0.00, 7.84, 17.92]
    recall = [0.00, 63.04, 43.48]
    f1_score = [0.00, 13.94, 25.38]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision (%)', color='#ff7f0e', edgecolor='black')
    rects2 = ax.bar(x, recall, width, label='Recall (%)', color='#1f77b4', edgecolor='black')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score (%)', color='#2ca02c', edgecolor='black')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Blob-level Performance at Threshold 0.85', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),                            
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    out_path = Path("outputs/ablation_bar_chart.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to {out_path}")
if __name__ == "__main__":
    main()
