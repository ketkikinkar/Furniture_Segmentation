import os
import json
import shutil
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import datetime

# Configuration
RGB_DIR = 'RGB'
DEPTH_DIR = 'depth'
ANNOTATIONS_DIR = 'annotations'
OUTPUT_DIR = 'dataset'
IMG_SIZE = (640, 640)  # YOLO default size
CLASSES = ['Background', 'Shelf', 'Door', 'Drawer']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}
# Colors for visualization (B,G,R format)
COLOR_MAP = {
    "Background": (255, 0, 0),    # Blue
    "Shelf": (0, 255, 0),         # Green
    "Door": (0, 255, 255),        # Yellow
    "Drawer": (0, 0, 255)         # Red
}

def create_directory_structure():
    """Create YOLO dataset directory structure"""
    # Create main directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'visualization'), exist_ok=True)

def load_and_resize_image(image_path, size=IMG_SIZE):
    """Load and resize image with error handling"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.resize(img, size)

def validate_polygon(points, img_size):
    """Validate polygon points are within image boundaries and form a valid polygon"""
    if len(points) < 3:  # Need at least 3 points for a polygon
        return False
    
    width, height = img_size
    # Check if points are within image boundaries
    for x, y in points:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
    
    return True

def convert_annotation_to_yolo(annotation_path, orig_size=(640, 480), img_size=(640, 640)):
    """Convert JSON annotation to YOLO segmentation format with improved validation"""
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading annotation file {annotation_path}: {str(e)}")
        return []

    yolo_annotations = []
    new_w, new_h = img_size
    orig_w, orig_h = orig_size

    # Extract region of interest if available to improve scaling
    roi = data.get('regionOfInterest', None)
    
    for part in data.get('parts', []):
        if 'points' not in part or 'label' not in part:
            continue

        label = part['label']
        class_id = CLASS_MAP.get(label, None)
        if class_id is None:
            print(f"Warning: Unknown label '{label}' in {annotation_path}")
            continue
            
        points = []
        for x, y in part['points']:
            # Scale to new image size with better boundary handling
            x_scaled = max(0, min(new_w - 1, x * new_w / orig_w))
            y_scaled = max(0, min(new_h - 1, y * new_h / orig_h))
            # Normalize to [0,1]
            x_norm = x_scaled / new_w
            y_norm = y_scaled / new_h
            points.extend([x_norm, y_norm])
        
        # Verify the polygon has enough points and is valid
        if len(points) >= 6:  # At least 3 points (x,y pairs)
            # Check if the polygon is valid
            if validate_polygon([(points[i], points[i+1]) for i in range(0, len(points), 2)], (1, 1)):
                yolo_annotations.append([class_id] + points)
            else:
                print(f"Warning: Invalid polygon for {label} in {annotation_path}")
    
    return yolo_annotations

def visualize_annotations(img_id, rgb_image, yolo_annotations, output_dir, img_size=IMG_SIZE):
    """Visualize YOLO annotations on the image for verification"""
    vis_img = rgb_image.copy()
    height, width = vis_img.shape[:2]
    
    for ann in yolo_annotations:
        class_id = int(ann[0])
        class_name = CLASSES[class_id]
        color = COLOR_MAP.get(class_name, (128, 128, 128))
        
        # Convert normalized coordinates back to image coordinates
        points = []
        for i in range(1, len(ann), 2):
            x, y = ann[i] * width, ann[i+1] * height
            points.append((int(x), int(y)))
        
        # Draw polygon
        if len(points) >= 3:
            pts = np.array([points], dtype=np.int32)
            cv2.polylines(vis_img, pts, isClosed=True, color=color, thickness=2)
            cv2.fillPoly(vis_img, pts, color=(color[0]//2, color[1]//2, color[2]//2))  # Fill with semi-transparent color
            
            # Add label text
            centroid_x = sum(p[0] for p in points) // len(points)
            centroid_y = sum(p[1] for p in points) // len(points)
            cv2.putText(vis_img, class_name, (centroid_x, centroid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'visualization', f"{img_id}_annotated.jpg")
    cv2.imwrite(vis_path, vis_img)
    return vis_img

def process_dataset(validate=True, visualize=True):
    """Process the entire dataset and create YOLO format with validation and visualization"""
    print("Creating YOLO dataset structure...")
    create_directory_structure()

    # Get all image IDs
    image_ids = [f.split('.')[0] for f in os.listdir(RGB_DIR) if f.endswith('.JPG') or f.endswith('.jpg')]
    
    if not image_ids:
        print("Error: No images found in RGB directory")
        return
    
    print(f"Found {len(image_ids)} images")
    
    # Split into train and validation sets
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    print(f"Processing {len(train_ids)} training images and {len(val_ids)} validation images")

    # Track statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'class_counts': {c: 0 for c in CLASSES}
    }

    # Process training set
    print("Processing training images...")
    for img_id in tqdm(train_ids):
        try:
            # Check for both uppercase and lowercase extensions
            rgb_path = None
            for ext in ['.JPG', '.jpg']:
                path = os.path.join(RGB_DIR, f"{img_id}{ext}")
                if os.path.exists(path):
                    rgb_path = path
                    break
            
            if rgb_path is None:
                print(f"Warning: RGB image for {img_id} not found, skipping")
                stats['skipped'] += 1
                continue
                
            # Load and resize RGB image
            rgb_img = load_and_resize_image(rgb_path)
            
            # Check if annotation exists
            annotation_path = os.path.join(ANNOTATIONS_DIR, f"{img_id}_annotation.json")
            if not os.path.exists(annotation_path):
                print(f"Warning: Annotation for {img_id} not found, skipping")
                stats['skipped'] += 1
                continue
            
            # Save RGB image
            output_img_path = os.path.join(OUTPUT_DIR, 'images', 'train', f"{img_id}.jpg")
            cv2.imwrite(output_img_path, rgb_img)
            
            # Process depth image if available
            depth_path = os.path.join(DEPTH_DIR, f"{img_id}.png")
            if os.path.exists(depth_path):
                depth_img = load_and_resize_image(depth_path)
                # We could process depth image here if needed
            
            # Convert and save annotation
            yolo_annotations = convert_annotation_to_yolo(annotation_path)
            
            if yolo_annotations:
                # Update class counts
                for ann in yolo_annotations:
                    class_id = int(ann[0])
                    if 0 <= class_id < len(CLASSES):
                        stats['class_counts'][CLASSES[class_id]] += 1
                
                # Save annotation
                output_label_path = os.path.join(OUTPUT_DIR, 'labels', 'train', f"{img_id}.txt")
                with open(output_label_path, 'w') as f:
                    for ann in yolo_annotations:
                        f.write(' '.join(map(str, ann)) + '\n')
                
                # Create visualization
                if visualize:
                    visualize_annotations(img_id, rgb_img, yolo_annotations, OUTPUT_DIR)
                
                stats['processed'] += 1
            else:
                print(f"Warning: No valid annotations found for {img_id}")
                stats['skipped'] += 1
            
        except Exception as e:
            print(f"Error processing training image {img_id}: {str(e)}")
            stats['errors'] += 1
            continue

    # Process validation set (similar to training set)
    print("Processing validation images...")
    for img_id in tqdm(val_ids):
        try:
            # Similar process as training set
            rgb_path = None
            for ext in ['.JPG', '.jpg']:
                path = os.path.join(RGB_DIR, f"{img_id}{ext}")
                if os.path.exists(path):
                    rgb_path = path
                    break
            
            if rgb_path is None:
                print(f"Warning: RGB image for {img_id} not found, skipping")
                stats['skipped'] += 1
                continue
                
            rgb_img = load_and_resize_image(rgb_path)
            
            annotation_path = os.path.join(ANNOTATIONS_DIR, f"{img_id}_annotation.json")
            if not os.path.exists(annotation_path):
                print(f"Warning: Annotation for {img_id} not found, skipping")
                stats['skipped'] += 1
                continue
            
            output_img_path = os.path.join(OUTPUT_DIR, 'images', 'val', f"{img_id}.jpg")
            cv2.imwrite(output_img_path, rgb_img)
            
            yolo_annotations = convert_annotation_to_yolo(annotation_path)
            
            if yolo_annotations:
                # Update class counts
                for ann in yolo_annotations:
                    class_id = int(ann[0])
                    if 0 <= class_id < len(CLASSES):
                        stats['class_counts'][CLASSES[class_id]] += 1
                
                output_label_path = os.path.join(OUTPUT_DIR, 'labels', 'val', f"{img_id}.txt")
                with open(output_label_path, 'w') as f:
                    for ann in yolo_annotations:
                        f.write(' '.join(map(str, ann)) + '\n')
                
                if visualize:
                    visualize_annotations(img_id, rgb_img, yolo_annotations, OUTPUT_DIR)
                
                stats['processed'] += 1
            else:
                print(f"Warning: No valid annotations found for {img_id}")
                stats['skipped'] += 1
            
        except Exception as e:
            print(f"Error processing validation image {img_id}: {str(e)}")
            stats['errors'] += 1
            continue

    # Create dataset.yaml file with improved metadata
    yaml_content = f"""# YOLO dataset configuration
path: {os.path.abspath(OUTPUT_DIR)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: {len(CLASSES)}  # number of classes
names:
{chr(10).join(f'  {idx}: {name}' for idx, name in enumerate(CLASSES))}

# Dataset information
dataset_info:
  description: Furniture segmentation dataset
  date_created: {datetime.datetime.now().strftime("%Y-%m-%d")}
  contributor: Furniture Dataset Creator
"""

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    # Generate dataset statistics visualization
    if visualize:
        generate_dataset_statistics(stats, train_ids, val_ids)

    print("\nDataset conversion completed!")
    print(f"Dataset saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nDataset structure:")
    print(f"- {len(train_ids)} training images")
    print(f"- {len(val_ids)} validation images")
    print(f"- {stats['processed']} successfully processed images")
    print(f"- {stats['skipped']} skipped images")
    print(f"- {stats['errors']} errors")
    print("\nClass distribution:")
    for cls, count in stats['class_counts'].items():
        print(f"- {cls}: {count} instances")
    print("\nYou can now use this dataset with YOLOv8 for segmentation training.")

def generate_dataset_statistics(stats, train_ids, val_ids):
    """Generate visualizations of dataset statistics"""
    # Create output directory
    stats_dir = os.path.join(OUTPUT_DIR, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    classes = list(stats['class_counts'].keys())
    counts = list(stats['class_counts'].values())
    plt.bar(classes, counts, color='skyblue')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'class_distribution.png'))
    
    # Plot dataset splits
    plt.figure(figsize=(8, 6))
    plt.pie([len(train_ids), len(val_ids)], 
            labels=['Training', 'Validation'],
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'])
    plt.title('Dataset Split')
    plt.savefig(os.path.join(stats_dir, 'dataset_split.png'))
    
    # Plot processing results
    plt.figure(figsize=(8, 6))
    plt.pie([stats['processed'], stats['skipped'], stats['errors']], 
            labels=['Processed', 'Skipped', 'Errors'],
            autopct='%1.1f%%',
            colors=['#99ff99','#ffcc99','#ff9999'])
    plt.title('Processing Results')
    plt.savefig(os.path.join(stats_dir, 'processing_results.png'))

def analyze_annotation_quality(sample_image_ids=5):
    """Analyze annotation quality on sample images"""
    print(f"Analyzing annotation quality on {sample_image_ids} sample images...")
    
    # Get all image IDs
    image_ids = [f.split('.')[0] for f in os.listdir(RGB_DIR) if f.endswith('.JPG') or f.endswith('.jpg')]
    
    if len(image_ids) == 0:
        print("No images found. Please check the RGB directory.")
        return
    
    # Select sample images
    sample_ids = image_ids[:sample_image_ids] if len(image_ids) > sample_image_ids else image_ids
    
    for img_id in sample_ids:
        try:
            # Find RGB image
            rgb_path = None
            for ext in ['.JPG', '.jpg']:
                path = os.path.join(RGB_DIR, f"{img_id}{ext}")
                if os.path.exists(path):
                    rgb_path = path
                    break
            
            if rgb_path is None:
                print(f"RGB image for {img_id} not found, skipping")
                continue
                
            # Load RGB image
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None:
                print(f"Could not read image: {rgb_path}")
                continue
            
            orig_h, orig_w = rgb_img.shape[:2]
            rgb_img_resized = cv2.resize(rgb_img, IMG_SIZE)
            
            # Check for annotation
            annotation_path = os.path.join(ANNOTATIONS_DIR, f"{img_id}_annotation.json")
            if not os.path.exists(annotation_path):
                print(f"Annotation for {img_id} not found, skipping")
                continue
            
            # Load and visualize the original annotation
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            
            # Create two images side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image with annotations
            original_vis = rgb_img_resized.copy()
            for part in data.get('parts', []):
                label = part.get('label', 'Unknown')
                color = COLOR_MAP.get(label, (128, 128, 128))
                points = part.get('points', [])
                
                # Convert points to image coordinates
                scaled_points = []
                for x, y in points:
                    x_scaled = int(x * IMG_SIZE[0] / orig_w)
                    y_scaled = int(y * IMG_SIZE[1] / orig_h)
                    scaled_points.append((x_scaled, y_scaled))
                
                if len(scaled_points) >= 3:
                    pts = np.array([scaled_points], dtype=np.int32)
                    cv2.polylines(original_vis, pts, isClosed=True, color=color, thickness=2)
                    cv2.fillPoly(original_vis, pts, color=(color[0]//2, color[1]//2, color[2]//2))
                    
                    # Add label
                    if scaled_points:
                        centroid_x = sum(p[0] for p in scaled_points) // len(scaled_points)
                        centroid_y = sum(p[1] for p in scaled_points) // len(scaled_points)
                        cv2.putText(original_vis, label, (centroid_x, centroid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert to RGB for matplotlib
            original_vis = cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB)
            ax1.imshow(original_vis)
            ax1.set_title(f"Original Annotation: {img_id}")
            ax1.axis('off')
            
            # YOLO converted annotations
            yolo_annotations = convert_annotation_to_yolo(annotation_path, (orig_w, orig_h), IMG_SIZE)
            yolo_vis = rgb_img_resized.copy()
            
            for ann in yolo_annotations:
                class_id = int(ann[0])
                class_name = CLASSES[class_id]
                color = COLOR_MAP.get(class_name, (128, 128, 128))
                
                # Convert normalized coordinates back to image coordinates
                points = []
                for i in range(1, len(ann), 2):
                    x, y = ann[i] * IMG_SIZE[0], ann[i+1] * IMG_SIZE[1]
                    points.append((int(x), int(y)))
                
                if len(points) >= 3:
                    pts = np.array([points], dtype=np.int32)
                    cv2.polylines(yolo_vis, pts, isClosed=True, color=color, thickness=2)
                    cv2.fillPoly(yolo_vis, pts, color=(color[0]//2, color[1]//2, color[2]//2))
                    
                    # Add label
                    centroid_x = sum(p[0] for p in points) // len(points)
                    centroid_y = sum(p[1] for p in points) // len(points)
                    cv2.putText(yolo_vis, class_name, (centroid_x, centroid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert to RGB for matplotlib
            yolo_vis = cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB)
            ax2.imshow(yolo_vis)
            ax2.set_title(f"YOLO Converted: {img_id}")
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'visualization', f"{img_id}_comparison.png"))
            plt.close()
            
            print(f"Generated comparison visualization for {img_id}")
            
        except Exception as e:
            print(f"Error analyzing annotation for {img_id}: {str(e)}")

if __name__ == '__main__':
    # First analyze annotation quality on sample images
    print("Step 1: Analyzing annotation quality on sample images...")
    analyze_annotation_quality()
    
    # Then process the full dataset
    print("\nStep 2: Processing the full dataset...")
    process_dataset(validate=True, visualize=True)