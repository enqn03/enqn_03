import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 한글 폰트 설정 (Mac OS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    labels = ['Recall (결함 발견율)', 'Precision (정밀도)']
    
    # 평가 방식별 성능 지표 (%)
    pixel_metrics = [4.5, 2.1]
    blob_metrics = [28.3, 10.2]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rects1 = ax.bar(x - width/2, pixel_metrics, width, label='픽셀 단위 (0mm 허용)', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x + width/2, blob_metrics, width, label='객체 단위 (2mm 허용)', color='#66b3ff', edgecolor='black')
    
    ax.set_ylabel('성능 (%)', fontsize=12)
    ax.set_title('평가 방식(Tolerance) 차이에 따른 성능 변화', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 35)
    
    # 값 표시
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
