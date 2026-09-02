import json
import matplotlib.pyplot as plt
import os
import numpy as np

def load_history(path):
    if not os.path.exists(path):
        return [], []
    with open(path, 'r') as f:
        data = json.load(f)
    train_loss = []
    val_loss = []
    
    for item in data.get('history', []):
        if item.get('split') == 'train':
            train_loss.append(item.get('mean_loss'))
        elif item.get('split') == 'validation':
            val_loss.append(item.get('mean_loss'))
            
    return train_loss, val_loss

def main():
    v6_train, v6_val = load_history('outputs/a_b_cbam_fusion_bce_c16_v6/training_history.json')
    v7_train, v7_val = load_history('outputs/a_b_cbam_fusion_bce_c16_v7_tuning/training_history.json')
    
    plt.figure(figsize=(10, 6))
    
    # Plot v6 (Un-tuned)
    if v6_val:
        epochs_v6 = range(1, len(v6_val) + 1)
        plt.plot(epochs_v6, v6_val, label='Validation Loss (v6: LR 3e-4, WD 0.01)', linestyle='--', color='red', alpha=0.7)
    
    # Plot v7 (Tuned)
    if v7_val:
        epochs_v7 = range(1, len(v7_val) + 1)
        plt.plot(epochs_v7, v7_val, label='Validation Loss (v7: LR 1e-4, WD 0.05)', linestyle='-', color='blue', linewidth=2)
        
    plt.title('Hyperparameter Tuning: Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation BCE Loss (Masked)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # Save the figure
    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('outputs/hyperparameter_tuning_loss.png', dpi=300)
    print("Saved outputs/hyperparameter_tuning_loss.png")

if __name__ == "__main__":
    main()
