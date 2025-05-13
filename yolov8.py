from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import pandas as pd
from collections import defaultdict

# Define class colors for visualization
CLASS_COLORS = {
    "Background": (0, 0, 255),    # Red
    "Shelf": (0, 255, 0),         # Green
    "Door": (255, 255, 0),        # Yellow
    "Drawer": (255, 0, 0)         # Blue
}

# Define class mapping for YOLO format
CLASS_MAPPING = {
    "Background": 0,
    "Shelf": 1,
    "Door": 2,
    "Drawer": 3
}

# Define classes list
CLASSES = ['Background', 'Shelf', 'Door', 'Drawer']

class ValidationVisualizer:
    def __init__(self, save_dir, val_path, num_samples=5):
        self.save_dir = save_dir
        self.val_path = val_path
        self.num_samples = num_samples
        self.vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Get validation image paths
        self.val_images = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def visualize_epoch(self, epoch, model):
        """Visualize random validation samples for current epoch"""
        # Create epoch directory
        epoch_dir = os.path.join(self.vis_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Select random images
        selected_images = random.sample(self.val_images, min(self.num_samples, len(self.val_images)))
        
        for img_name in selected_images:
            img_path = os.path.join(self.val_path, img_name)
            img = cv2.imread(img_path)
            
            # Get predictions
            results = model(img)
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Ground truth
            gt_img = self.visualize_ground_truth(img_path, img_path.replace('images', 'labels').replace('.jpg', '.json'))
            if gt_img is not None:
                ax2.imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
            ax2.set_title('Ground Truth')
            ax2.axis('off')
            
            # Predictions
            pred_img = self.visualize_prediction(img_path, results)
            if pred_img is not None:
                ax3.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
            ax3.set_title('Predictions')
            ax3.axis('off')
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f'{img_name}_visualization.png'))
            plt.close()

    def visualize_ground_truth(self, image_path, annotation_path):
        """Visualize ground truth annotation on the image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Read annotation
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except:
            return image
        
        # Create overlay
        overlay = image.copy()
        
        # Draw annotations
        if 'parts' in data:
            for part in data['parts']:
                if 'label' in part and 'points' in part:
                    label = part['label']
                    points = np.array(part['points'], dtype=np.int32)
                    
                    # Get color for this class
                    color = CLASS_COLORS.get(label, (128, 128, 128))
                    
                    # Draw filled polygon with transparency
                    cv2.fillPoly(overlay, [points], color)
                    
                    # Draw polygon outline
                    cv2.polylines(overlay, [points], True, (255, 255, 255), 2)
                    
                    # Add label
                    x, y = points[0]
                    cv2.putText(overlay, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend overlay with original image
        alpha = 0.5
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return result

    def visualize_prediction(self, image_path, results):
        """Visualize model predictions on the image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Create overlay
        overlay = image.copy()
        
        # Draw predictions
        if results.masks is not None:
            for mask, box in zip(results.masks, results.boxes):
                # Get mask coordinates
                mask_coords = mask.xy[0].astype(np.int32)
                
                # Get class and confidence
                cls = int(box.cls[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
                # Get class name and color
                class_name = CLASSES[cls]
                color = CLASS_COLORS.get(class_name, (128, 128, 128))
                
                # Draw filled polygon with transparency
                cv2.fillPoly(overlay, [mask_coords], color)
                
                # Draw polygon outline
                cv2.polylines(overlay, [mask_coords], True, (255, 255, 255), 2)
                
                # Add label with confidence
                label = f"{class_name} {conf:.2f}"
                x, y = mask_coords[0]
                cv2.putText(overlay, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend overlay with original image
        alpha = 0.5
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return result

class MetricsTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics = defaultdict(list)
        self.best_map = 0
        self.best_epoch = 0
        
    def update(self, epoch, metrics_dict):
        """Update metrics for current epoch"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            
        # Save best model metrics
        if 'metrics/mAP50(B)' in metrics_dict and metrics_dict['metrics/mAP50(B)'] > self.best_map:
            self.best_map = metrics_dict['metrics/mAP50(B)']
            self.best_epoch = epoch
            
    def save_metrics(self, epoch):
        """Save essential metrics for current epoch"""
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(self.save_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save current epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'mAP50': self.metrics['metrics/mAP50(B)'][-1] if 'metrics/mAP50(B)' in self.metrics else None,
            'precision': self.metrics['metrics/precision(B)'][-1] if 'metrics/precision(B)' in self.metrics else None,
            'recall': self.metrics['metrics/recall(B)'][-1] if 'metrics/recall(B)' in self.metrics else None,
            'box_loss': self.metrics['train/box_loss'][-1] if 'train/box_loss' in self.metrics else None,
            'cls_loss': self.metrics['train/cls_loss'][-1] if 'train/cls_loss' in self.metrics else None,
            'dfl_loss': self.metrics['train/dfl_loss'][-1] if 'train/dfl_loss' in self.metrics else None,
            'lr': self.metrics['lr/pg0'][-1] if 'lr/pg0' in self.metrics else None
        }
        
        # Save as JSON
        with open(os.path.join(metrics_dir, f'epoch_{epoch:03d}_metrics.json'), 'w') as f:
            json.dump(epoch_metrics, f, indent=2)
            
    def save_summary(self):
        """Save training summary"""
        summary = {
            'best_epoch': self.best_epoch,
            'best_map': self.best_map,
            'final_metrics': {
                'mAP50': self.metrics['metrics/mAP50(B)'][-1] if 'metrics/mAP50(B)' in self.metrics else None,
                'precision': self.metrics['metrics/precision(B)'][-1] if 'metrics/precision(B)' in self.metrics else None,
                'recall': self.metrics['metrics/recall(B)'][-1] if 'metrics/recall(B)' in self.metrics else None
            }
        }
        
        with open(os.path.join(self.save_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save metrics history as CSV
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(os.path.join(self.save_dir, 'metrics_history.csv'), index=False)

def train_yolo_model(data_yaml_path, epochs=300, batch_size=32, img_size=640):
    """Train YOLO model on the prepared dataset"""
    # Initialize YOLO model
    model = YOLO('yolov8m-seg.pt')  # Use medium model for better performance
    
    # Ensure model is in training mode and parameters require gradients
    model.model.train()
    for param in model.model.parameters():
        param.requires_grad = True
    
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('furniture_detection', f'train_{timestamp}')
    
    # Initialize metrics tracker and visualizer
    metrics_tracker = MetricsTracker(save_dir)
    visualizer = ValidationVisualizer(save_dir, 'dataset/images/val')
    
    # Train the model with optimized configuration
    print("Starting YOLO training...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=50,  # Early stopping patience
        save=True,    # Save best model
        project='furniture_detection',
        name=f'train_{timestamp}',
        device='0' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        pretrained=True,  # Use pretrained weights
        optimizer='AdamW',  # Use AdamW optimizer
        lr0=0.0005,  # Lower initial learning rate
        lrf=0.01,   # Final learning rate
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=5,  # Increased warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,  # Warmup bias learning rate
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution Focal Loss gain
        pose=12.0,  # Pose loss gain
        kobj=2.0,  # Keypoint obj loss gain
        nbs=64,  # Nominal batch size
        overlap_mask=True,  # Masks should overlap during training
        mask_ratio=4,  # Mask downsample ratio
        dropout=0.1,  # Add dropout regularization
        val=True,  # Validate during training
        amp=True,  # Enable AMP for faster training
        deterministic=False,  # Disable deterministic mode
        workers=4,  # Increase workers for faster data loading
        close_mosaic=0,  # Enable mosaic augmentation
        cos_lr=True,  # Use cosine learning rate scheduler
        multi_scale=True,  # Enable multi-scale training
        single_cls=False,  # Multi-class training
        rect=False,  # Rectangular training
        cache=True,  # Cache images in memory
        exist_ok=False,  # Don't overwrite existing experiment
        resume=False,  # Don't resume from previous checkpoint
        fraction=1.0,  # Use full dataset
        seed=42,  # Fixed random seed for reproducibility
        verbose=True  # Print verbose output
    )
    
    # Save final training summary
    metrics_tracker.save_summary()
    
    return results

if __name__ == "__main__":
    # Path to data.yaml file
    data_yaml_path = "dataset/data.yaml"
    
    # Train the model
    results = train_yolo_model(
        data_yaml_path=data_yaml_path,
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    print("Training complete!")
    print(f"Results saved in: {results.save_dir}") 
