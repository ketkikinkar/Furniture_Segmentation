from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- SPECIFIED WEIGHTS PATH ---
weights_path = 'furniture_training/furniture_seg_20250513_134438/weights/best.pt'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found at {weights_path}")

val_dir = 'dataset/images/val'
output_dir = 'val_predictions_visualized_updated'
metrics_dir = 'validation_metrics'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Class names and colors (should match your training config)
CLASSES = ['Background', 'Shelf', 'Door', 'Drawer']
CLASS_COLORS = {
    "Background": (0, 0, 255),    # Red
    "Shelf": (0, 255, 0),         # Green
    "Door": (255, 255, 0),        # Yellow
    "Drawer": (255, 0, 0)         # Blue
}

def plot_metrics(metrics, save_dir):
    """Plot various metrics and save them as images"""
    # Create figure for loss curves
    plt.figure(figsize=(15, 10))
    
    # Plot box loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch'], metrics['train/box_loss'], label='Train Box Loss')
    plt.plot(metrics['epoch'], metrics['val/box_loss'], label='Val Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.title('Box Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot segmentation loss
    plt.subplot(2, 2, 2)
    plt.plot(metrics['epoch'], metrics['train/seg_loss'], label='Train Seg Loss')
    plt.plot(metrics['epoch'], metrics['val/seg_loss'], label='Val Seg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Segmentation Loss')
    plt.title('Segmentation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot mAP curves
    plt.subplot(2, 2, 3)
    plt.plot(metrics['epoch'], metrics['metrics/mAP50(B)'], label='mAP50')
    plt.plot(metrics['epoch'], metrics['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot precision and recall
    plt.subplot(2, 2, 4)
    plt.plot(metrics['epoch'], metrics['metrics/precision(B)'], label='Precision')
    plt.plot(metrics['epoch'], metrics['metrics/recall(B)'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Precision and Recall')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_and_metrics_curves.png'))
    plt.close()
    
    # Create per-class metrics plot
    plt.figure(figsize=(15, 5))
    for i, class_name in enumerate(CLASSES[1:], 1):  # Skip background
        plt.subplot(1, 3, i)
        plt.plot(metrics['epoch'], metrics[f'metrics/precision_{i}(B)'], label='Precision')
        plt.plot(metrics['epoch'], metrics[f'metrics/recall_{i}(B)'], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f'{class_name} Metrics')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'))
    plt.close()

# --- LOAD MODEL ---
model = YOLO(str(weights_path))

# --- VISUALIZATION FUNCTION ---
def visualize_prediction(image_path, results):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    overlay = image.copy()
    if results.masks is not None:
        for mask, box in zip(results.masks, results.boxes):
            mask_coords = mask.xy[0].astype(int)
            cls = int(box.cls[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            class_name = CLASSES[cls]
            color = CLASS_COLORS.get(class_name, (128, 128, 128))
            cv2.fillPoly(overlay, [mask_coords], color)
            cv2.polylines(overlay, [mask_coords], True, (255, 255, 255), 2)
            label = f"{class_name} {conf:.2f}"
            x, y = mask_coords[0]
            cv2.putText(overlay, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    alpha = 0.5
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result

# --- RUN INFERENCE AND SAVE VISUALIZATIONS ---
val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img_name in val_images:
    img_path = os.path.join(val_dir, img_name)
    results = model(img_path)[0]
    vis_img = visualize_prediction(img_path, results)
    if vis_img is not None:
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, vis_img)
        print(f"Saved: {out_path}")

print(f"Visualization complete! Used weights: {weights_path}")
print("Check the", output_dir, "folder.")

# Path to your data.yaml
data_yaml = 'dataset/data.yaml'

# Load model and run validation
model = YOLO(str(weights_path))
results = model.val(data=data_yaml, split='val')

# Get training history
results_file = os.path.join(os.path.dirname(weights_path), 'results.csv')
if os.path.exists(results_file):
    import pandas as pd
    history = pd.read_csv(results_file)
    plot_metrics(history, metrics_dir)
    print(f"Saved metric plots to {metrics_dir}")

# Print summary metrics
print("\nValidation metrics:")
print("Box Detection Metrics:")
print(f"mAP50: {results.box.map50:.3f}")
print(f"mAP50-95: {results.box.map:.3f}")

print("\nSegmentation Metrics:")
print(f"mAP50: {results.seg.map50:.3f}")
print(f"mAP50-95: {results.seg.map:.3f}")

print("\nSpeed metrics:")
for k, v in results.speed.items():
    print(f"{k}: {v:.1f}ms")

# Save detailed metrics
with open(os.path.join(metrics_dir, 'val_metrics.txt'), 'w') as f:
    f.write("=== Final Validation Metrics ===\n\n")
    
    f.write("Box Detection Metrics:\n")
    f.write(f"mAP50: {results.box.map50:.3f}\n")
    f.write(f"mAP50-95: {results.box.map:.3f}\n\n")
    
    f.write("Segmentation Metrics:\n")
    f.write(f"mAP50: {results.seg.map50:.3f}\n")
    f.write(f"mAP50-95: {results.seg.map:.3f}\n\n")
    
    f.write("Speed Metrics:\n")
    for k, v in results.speed.items():
        f.write(f"{k}: {v:.1f}ms\n")
    
    if os.path.exists(results_file):
        f.write("\n=== Training History ===\n\n")
        f.write(history.to_string())
print(f"Metrics saved to {metrics_dir}/val_metrics.txt") 