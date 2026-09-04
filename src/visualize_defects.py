utf-8
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
def main():
    csv_path = Path("outputs/live_inference_results_v6.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV file is empty. No defects to visualize.")
        return
    df['layer_z'] = df['layer_z'].astype(str).str.extract(r'(\d+)').astype(int)
    print(f"Loaded {len(df)} defect points.")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df['machine_x_mm'], 
        df['machine_y_mm'], 
        c=df['score_percent'], 
        cmap='YlOrRd', 
        s=50, 
        alpha=0.8, 
        edgecolors='k',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Confidence Score (%)')
    plt.title('AMMT Defect Distribution (Top-Down View)')
    plt.xlabel('Machine X (mm)')
    plt.ylabel('Machine Y (mm)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    out_2d = Path("outputs/defect_distribution_2d.png")
    plt.savefig(out_2d, dpi=300, bbox_inches='tight')
    print(f"Saved 2D visualization to {out_2d}")
    plt.close()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter3d = ax.scatter(
        df['machine_x_mm'], 
        df['machine_y_mm'], 
        df['layer_z'], 
        c=df['score_percent'], 
        cmap='YlOrRd', 
        s=50, 
        alpha=0.8,
        edgecolors='k',
        linewidth=0.5
    )
    fig.colorbar(scatter3d, label='Confidence Score (%)', shrink=0.7, pad=0.1)
    ax.set_title('AMMT Defect Distribution (3D View)')
    ax.set_xlabel('Machine X (mm)')
    ax.set_ylabel('Machine Y (mm)')
    ax.set_zlabel('Layer (Z)')
    out_3d = Path("outputs/defect_distribution_3d.png")
    plt.savefig(out_3d, dpi=300, bbox_inches='tight')
    print(f"Saved 3D visualization to {out_3d}")
    plt.close()
if __name__ == "__main__":
    main()
