import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time

# Path to results
results_path = Path("../../runs/detect/train_pipeline/results.csv")

# Enable interactive mode for live updates
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Live Training Performance Tracker')

last_mtime = 0

while True:
    if results_path.exists():
        current_mtime = results_path.stat().st_mtime
        if current_mtime > last_mtime:
            df = pd.read_csv(results_path)
            last_mtime = current_mtime

            # Clear previous plots
            for ax in axs.flat:
                ax.clear()

            # Plot loss curves
            axs[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            axs[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            axs[0, 0].set_title('Box Loss')
            axs[0, 0].legend()

            axs[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
            axs[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
            axs[0, 1].set_title('Cls Loss')
            axs[0, 1].legend()

            axs[1, 0].plot(df['epoch'], df['train/dfl_loss'], label='Train Dfl Loss')
            axs[1, 0].plot(df['epoch'], df['val/dfl_loss'], label='Val Dfl Loss')
            axs[1, 0].set_title('Dfl Loss')
            axs[1, 0].legend()

            axs[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            axs[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
            axs[1, 1].set_title('mAP')
            axs[1, 1].legend()

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

            print(f"Updated plot at epoch {df['epoch'].max()}")
        else:
            print("Waiting for updates...")
    else:
        print("Results.csv not found. Run training first.")
        break

    time.sleep(5)  # Check every 5 seconds

# Save final plot
plt.savefig('training_progress.png')
plt.ioff()
plt.show()
print("Training progress plot saved as training_progress.png")

# For confusion matrix, if val is done, it should be in runs/val/
val_dir = Path("../../Desktop/main/runs/val")
if val_dir.exists():
    confusion_path = val_dir / "confusion_matrix.png"
    if confusion_path.exists():
        print(f"Confusion matrix available at {confusion_path}")
    else:
        print("Confusion matrix not found.")
else:
    print("Val directory not found. Run validation first.")
