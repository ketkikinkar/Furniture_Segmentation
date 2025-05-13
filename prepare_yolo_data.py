import os
import json
import shutil
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

class YOLODataPreparator:
    def __init__(self, rgb_dir="RGB", annotations_dir="annotations", output_dir="dataset"):
        self.rgb_dir = rgb_dir
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.classes = {
            "Shelf": 0,
            "Drawer": 1,
            "Background": 2
        }
        
        # Create output directory structure
        self.train_dir = os.path.join(output_dir, "images", "train")
        self.val_dir = os.path.join(output_dir, "images", "val")
        self.train_labels_dir = os.path.join(output_dir, "labels", "train")
        self.val_labels_dir = os.path.join(output_dir, "labels", "val")
        
        for dir_path in [self.train_dir, self.val_dir, self.train_labels_dir, self.val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def normalize_points(self, points, img_width, img_height):
        """Normalize polygon points to [0,1] range"""
        points = np.array(points)
        points[:, 0] = points[:, 0] / img_width
        points[:, 1] = points[:, 1] / img_height
        return points.flatten().tolist()

    def process_annotation(self, annotation_file, img_file):
        """Process a single annotation file and convert to YOLO format"""
        try:
            # Read image to get dimensions
            img = cv2.imread(img_file)
            if img is None:
                print(f"Could not read image: {img_file}")
                return None
            
            img_height, img_width = img.shape[:2]
            
            # Read annotation
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            yolo_annotations = []
            
            # Process each part in the annotation
            if 'parts' in data:
                for part in data['parts']:
                    if 'label' in part and 'points' in part:
                        label = part['label']
                        points = part['points']
                        
                        if label in self.classes:
                            class_id = self.classes[label]
                            
                            # Normalize points for segmentation
                            normalized_points = self.normalize_points(points, img_width, img_height)
                            
                            # Create YOLO format annotation
                            # Format: class_id x1 y1 x2 y2 ... xn yn
                            yolo_annotation = [class_id] + normalized_points
                            yolo_annotations.append(yolo_annotation)
            
            return yolo_annotations
        except Exception as e:
            print(f"Error processing annotation file {annotation_file}: {str(e)}")
            return None

    def prepare_dataset(self, train_ratio=0.8):
        """Prepare the dataset for YOLO training"""
        try:
            # Get all image files
            image_files = [f for f in os.listdir(self.rgb_dir) if f.endswith('.JPG')]
            np.random.shuffle(image_files)
            
            # Split into train and validation sets
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            # Process training set
            print("Processing training set...")
            for img_file in tqdm(train_files):
                img_id = img_file.split('.')[0]
                annotation_file = os.path.join(self.annotations_dir, f"{img_id}_annotation.json")
                
                if os.path.exists(annotation_file):
                    # Copy image
                    shutil.copy2(
                        os.path.join(self.rgb_dir, img_file),
                        os.path.join(self.train_dir, img_file)
                    )
                    
                    # Process and save annotation
                    yolo_annotations = self.process_annotation(
                        annotation_file, 
                        os.path.join(self.rgb_dir, img_file)
                    )
                    
                    if yolo_annotations:
                        # Save YOLO format annotations
                        with open(os.path.join(self.train_labels_dir, f"{img_id}.txt"), 'w') as f:
                            for ann in yolo_annotations:
                                f.write(' '.join(map(str, ann)) + '\n')
            
            # Process validation set
            print("Processing validation set...")
            for img_file in tqdm(val_files):
                img_id = img_file.split('.')[0]
                annotation_file = os.path.join(self.annotations_dir, f"{img_id}_annotation.json")
                
                if os.path.exists(annotation_file):
                    # Copy image
                    shutil.copy2(
                        os.path.join(self.rgb_dir, img_file),
                        os.path.join(self.val_dir, img_file)
                    )
                    
                    # Process and save annotation
                    yolo_annotations = self.process_annotation(
                        annotation_file, 
                        os.path.join(self.rgb_dir, img_file)
                    )
                    
                    if yolo_annotations:
                        # Save YOLO format annotations
                        with open(os.path.join(self.val_labels_dir, f"{img_id}.txt"), 'w') as f:
                            for ann in yolo_annotations:
                                f.write(' '.join(map(str, ann)) + '\n')
            
            # Create data.yaml file
            yaml_content = {
                'path': os.path.abspath(self.output_dir),
                'train': 'images/train',
                'val': 'images/val',
                'names': {v: k for k, v in self.classes.items()},
                'nc': len(self.classes),  # Number of classes
                'task': 'segment'  # Specify that this is a segmentation task
            }
            
            with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
                import yaml
                yaml.dump(yaml_content, f)
            
            print(f"Dataset preparation complete. Output directory: {self.output_dir}")
            
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")

if __name__ == "__main__":
    preparator = YOLODataPreparator()
    preparator.prepare_dataset() 