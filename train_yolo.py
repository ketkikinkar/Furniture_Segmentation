import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import glob
from tqdm import tqdm

# Define class colors for visualization
CLASS_COLORS = {
    "Background": (0, 0, 255),    # Red
    "Shelf": (0, 255, 0),         # Green
    "Door": (255, 255, 0),        # Yellow
    "Drawer": (255, 0, 0)         # Blue
}

# Define image extensions
IMAGE_EXTS = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')

def has_label(img_name, label_dir):
    """Check if a label file exists for the given image"""
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    label_path = os.path.join(label_dir, f"{base_name}.txt")
    return os.path.exists(label_path)

def check_dataset_structure(dataset_path):
    """Check if dataset structure is correct for YOLO format"""
    print("\nChecking dataset structure...")
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' does not exist!")
        return False
    
    # Define paths
    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')
    train_img_dir = os.path.join(image_dir, 'train')
    val_img_dir = os.path.join(image_dir, 'val')
    train_label_dir = os.path.join(label_dir, 'train')
    val_label_dir = os.path.join(label_dir, 'val')
    
    # Check directories exist
    for dir_path, dir_name in [
        (image_dir, 'Images'),
        (label_dir, 'Labels'),
        (train_img_dir, 'Training images'),
        (val_img_dir, 'Validation images'),
        (train_label_dir, 'Training labels'),
        (val_label_dir, 'Validation labels')
    ]:
        if not os.path.exists(dir_path):
            print(f"Error: {dir_name} directory '{dir_path}' does not exist!")
            return False
    
    # Check for YAML file
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Warning: data.yaml not found at '{yaml_path}'")
    
    # Get all images with extensions
    train_images = []
    val_images = []
    
    for ext in IMAGE_EXTS:
        train_images.extend(glob.glob(os.path.join(train_img_dir, f"*{ext}")))
        val_images.extend(glob.glob(os.path.join(val_img_dir, f"*{ext}")))
    
    # Check for matching labels
    train_with_labels = [img for img in train_images if has_label(img, train_label_dir)]
    val_with_labels = [img for img in val_images if has_label(img, val_label_dir)]
    
    if not train_with_labels and not val_with_labels:
        print(f"Error: No images with matching labels found!")
        return False
    
    print(f"Found {len(train_with_labels)} training and {len(val_with_labels)} validation images with labels")
    
    # Sample check for segmentation format (not bounding box)
    if train_with_labels:
        sample_label = os.path.join(train_label_dir, 
                                   f"{os.path.splitext(os.path.basename(train_with_labels[0]))[0]}.txt")
        with open(sample_label, 'r') as f:
            first_line = f.readline().strip().split()
            if len(first_line) <= 5:  # x, y, w, h format (bounding box) has 5 values including class id
                print("Warning: Labels appear to be in bounding box format, not segmentation format")
            else:
                print("Labels appear to be in segmentation format")
    
    return True

def visualize_dataset(dataset_path, num_samples=5, save_dir='dataset_visualization'):
    """Visualize dataset images with segmentation masks and overlays"""
    # Check dataset structure first
    if not check_dataset_structure(dataset_path):
        return
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving visualizations to: {save_dir}")
    
    # Define paths
    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')
    train_img_dir = os.path.join(image_dir, 'train')
    val_img_dir = os.path.join(image_dir, 'val')
    train_label_dir = os.path.join(label_dir, 'train')
    val_label_dir = os.path.join(label_dir, 'val')
    
    # Get all images with extensions
    train_images = []
    val_images = []
    
    for ext in IMAGE_EXTS:
        train_images.extend(glob.glob(os.path.join(train_img_dir, f"*{ext}")))
        val_images.extend(glob.glob(os.path.join(val_img_dir, f"*{ext}")))
    
    # Filter for images with labels
    train_with_labels = [img for img in train_images if has_label(img, train_label_dir)]
    val_with_labels = [img for img in val_images if has_label(img, val_label_dir)]
    
    # Randomly select samples from both sets
    num_train = min(num_samples // 2, len(train_with_labels))
    num_val = min(num_samples - num_train, len(val_with_labels))
    
    selected_train = random.sample(train_with_labels, num_train) if train_with_labels else []
    selected_val = random.sample(val_with_labels, num_val) if val_with_labels else []
    
    print(f"\nSelected {len(selected_train)} training images and {len(selected_val)} validation images")
    
    # Process training images
    for img_path in selected_train:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(train_label_dir, f"{base_name}.txt")
        process_image(img_path, label_path, base_name, "Train", save_dir)
    
    # Process validation images
    for img_path in selected_val:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_label_dir, f"{base_name}.txt")
        process_image(img_path, label_path, base_name, "Val", save_dir)
    
    # Create class distribution visualization
    analyze_class_distribution(dataset_path, save_dir)

def process_image(img_path, label_path, base_name, split_type, save_dir):
    """Process a single image and its label"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image: {img_path}")
        return
    
    # Check label file
    if not os.path.exists(label_path):
        print(f"Error: Label file not found: {label_path}")
        return
    
    print(f"\nProcessing {split_type} image: {base_name}...")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Segmentation masks (individual classes)
    mask_img = create_segmentation_masks(img, label_path)
    ax2.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Segmentation Masks')
    ax2.axis('off')
    
    # Segmentation overlay
    overlay_img = create_segmentation_overlay(img, label_path)
    ax3.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    ax3.set_title('Segmentation Overlay')
    ax3.axis('off')
    
    # Add legend
    classes_in_image = get_classes_in_image(label_path)
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=tuple(c/255 for c in CLASS_COLORS[cls]), label=cls)
                     for cls in classes_in_image]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0),
              ncol=len(classes_in_image), frameon=False)
    
    # Add information text
    plt.figtext(0.5, -0.05, f"Image ID: {base_name} | Split: {split_type}", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save visualization
    output_path = os.path.join(save_dir, f'{base_name}_visualization.png')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")

def get_classes_in_image(label_path):
    """Get list of classes present in the image"""
    # Create a list of category names from class IDs found in the label file
    classes = []
    class_names = list(CLASS_COLORS.keys())
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                        if class_name not in classes:
                            classes.append(class_name)
    except Exception as e:
        print(f"Error reading label file {label_path}: {str(e)}")
    
    return classes

def create_segmentation_masks(image, label_path):
    """Create visualization of segmentation masks with different colors per class"""
    # Read the label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading label file {label_path}: {str(e)}")
        return image
    
    # Create mask overlay
    overlay = image.copy()
    mask = np.zeros_like(image)
    height, width = image.shape[:2]
    
    # Draw each segmentation mask with its class color
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:  # Need at least class and one point
            continue
            
        class_id = int(parts[0])
        class_names = list(CLASS_COLORS.keys())
        
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
            color = CLASS_COLORS[class_name]
            
            # Parse polygon points
            points = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) * width
                    y = float(parts[i + 1]) * height
                    points.append((int(x), int(y)))
            
            # Draw polygon if there are enough points
            if len(points) >= 3:
                pts = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, pts, color)
                
                # Add label at centroid of polygon
                M = cv2.moments(pts[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(mask, class_name, (cx, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return mask

def create_segmentation_overlay(image, label_path):
    """Create overlay of segmentation masks on the image"""
    # Create the mask
    mask = create_segmentation_masks(image, label_path)
    
    # Blend with original image
    alpha = 0.5
    result = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)
    
    return result

def analyze_class_distribution(dataset_path, save_dir):
    """Analyze and visualize class distribution in the dataset"""
    # Define paths
    label_dir = os.path.join(dataset_path, 'labels')
    train_label_dir = os.path.join(label_dir, 'train')
    val_label_dir = os.path.join(label_dir, 'val')
    
    print("\nAnalyzing class distribution...")
    
    # Initialize counters
    train_counts = {cls: 0 for cls in CLASS_COLORS.keys()}
    val_counts = {cls: 0 for cls in CLASS_COLORS.keys()}
    
    # Count classes in training set
    train_labels = glob.glob(os.path.join(train_label_dir, "*.txt"))
    for label_path in tqdm(train_labels, desc="Processing training labels"):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        class_names = list(CLASS_COLORS.keys())
                        if 0 <= class_id < len(class_names):
                            train_counts[class_names[class_id]] += 1
        except Exception as e:
            print(f"Error reading {label_path}: {str(e)}")
    
    # Count classes in validation set
    val_labels = glob.glob(os.path.join(val_label_dir, "*.txt"))
    for label_path in tqdm(val_labels, desc="Processing validation labels"):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        class_names = list(CLASS_COLORS.keys())
                        if 0 <= class_id < len(class_names):
                            val_counts[class_names[class_id]] += 1
        except Exception as e:
            print(f"Error reading {label_path}: {str(e)}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    classes = list(CLASS_COLORS.keys())
    train_values = [train_counts[cls] for cls in classes]
    val_values = [val_counts[cls] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, train_values, width, label='Training')
    bars2 = ax.bar(x + width/2, val_values, width, label='Validation')
    
    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    # Add count labels above bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()
    
    print(f"Saved class distribution visualization to: {os.path.join(save_dir, 'class_distribution.png')}")
    
    # Create pie charts for each set
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Training set
    train_total = sum(train_values)
    if train_total > 0:
        train_labels = [f'{cls}: {count} ({count/train_total*100:.1f}%)' 
                        for cls, count in zip(classes, train_values) if count > 0]
        train_values_filtered = [count for count in train_values if count > 0]
        ax1.pie(train_values_filtered, labels=train_labels, autopct='', 
                colors=[tuple(c/255 for c in CLASS_COLORS[cls]) for cls in classes if train_counts[cls] > 0],
                startangle=90)
        ax1.set_title(f'Training Set Distribution ({train_total} instances)')
    else:
        ax1.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
    
    # Validation set
    val_total = sum(val_values)
    if val_total > 0:
        val_labels = [f'{cls}: {count} ({count/val_total*100:.1f}%)' 
                     for cls, count in zip(classes, val_values) if count > 0]
        val_values_filtered = [count for count in val_values if count > 0]
        ax2.pie(val_values_filtered, labels=val_labels, autopct='',
               colors=[tuple(c/255 for c in CLASS_COLORS[cls]) for cls in classes if val_counts[cls] > 0],
               startangle=90)
        ax2.set_title(f'Validation Set Distribution ({val_total} instances)')
    else:
        ax2.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution_pie.png'))
    plt.close()
    
    print(f"Saved class distribution pie charts to: {os.path.join(save_dir, 'class_distribution_pie.png')}")
    
    # Create a summary text file
    with open(os.path.join(save_dir, 'dataset_summary.txt'), 'w') as f:
        f.write("Dataset Summary\n")
        f.write("==============\n\n")
        f.write(f"Total training instances: {train_total}\n")
        f.write(f"Total validation instances: {val_total}\n\n")
        f.write("Training set class distribution:\n")
        for cls in classes:
            if train_total > 0:
                percent = train_counts[cls]/train_total*100
            else:
                percent = 0
            f.write(f"  - {cls}: {train_counts[cls]} ({percent:.1f}%)\n")
        f.write("\nValidation set class distribution:\n")
        for cls in classes:
            if val_total > 0:
                percent = val_counts[cls]/val_total*100
            else:
                percent = 0
            f.write(f"  - {cls}: {val_counts[cls]} ({percent:.1f}%)\n")
    
    print(f"Saved dataset summary to: {os.path.join(save_dir, 'dataset_summary.txt')}")

def create_dataset_overview(dataset_path, save_dir, grid_size=(3, 3)):
    """Create an overview grid of images with their segmentation masks"""
    # Define paths
    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')
    train_img_dir = os.path.join(image_dir, 'train')
    val_img_dir = os.path.join(image_dir, 'val')
    train_label_dir = os.path.join(label_dir, 'train')
    val_label_dir = os.path.join(label_dir, 'val')
    
    # Get all images with labels
    train_images = []
    for ext in IMAGE_EXTS:
        train_images.extend([img for img in glob.glob(os.path.join(train_img_dir, f"*{ext}")) 
                            if has_label(img, train_label_dir)])
    
    # Select random images
    num_images = grid_size[0] * grid_size[1]
    if len(train_images) < num_images:
        selected_images = train_images
    else:
        selected_images = random.sample(train_images, num_images)
    
    # Create grid
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i, img_path in enumerate(selected_images):
        if i >= len(axes):
            break
            
        # Get image and label
        img = cv2.imread(img_path)
        if img is None:
            axes[i].text(0.5, 0.5, 'Image Error', ha='center', va='center')
            continue
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(train_label_dir, f"{base_name}.txt")
        
        # Create overlay
        overlay = create_segmentation_overlay(img, label_path)
        
        # Display
        axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{base_name}")
        axes[i].axis('off')
    
    # Handle any unused axes
    for i in range(len(selected_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_overview_grid.png'))
    plt.close()
    
    print(f"Saved dataset overview grid to: {os.path.join(save_dir, 'dataset_overview_grid.png')}")

if __name__ == "__main__":
    # Path to dataset directory
    dataset_path = "dataset"
    
    # Number of samples to visualize in detail
    num_samples = 6
    
    # Output directory
    save_dir = "dataset_visualization"
    
    print("\n=== YOLO Segmentation Dataset Visualization ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples to visualize: {num_samples}")
    print(f"Output directory: {save_dir}")
    
    # Create visualizations
    visualize_dataset(dataset_path, num_samples, save_dir)
    
    # Create dataset overview grid
    create_dataset_overview(dataset_path, save_dir)
    
    print(f"\nVisualization complete!")
    print(f"Check the '{save_dir}' directory for results")
    print("\nVerify the following in the visualizations:")
    print("1. Segmentation masks accurately cover furniture parts")
    print("2. Class labels are correctly assigned")
    print("3. Class distribution is balanced")
    print("4. Annotation quality is consistent across the dataset")
