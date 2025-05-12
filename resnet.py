import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RGB_DIR = 'RGB'
DEPTH_DIR = 'depth'
ANNOTATIONS_DIR = 'annotations'
IMG_SIZE = (256, 256)  # Resize all images/masks to this size
CLASSES = ['Background', 'Shelf', 'Door', 'Drawer']  # Updated classes based on annotations
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

# --- CUSTOM METRICS ---
class IoU(keras.metrics.Metric):
    def __init__(self, num_classes, name='iou', **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.intersection = self.add_weight(name='intersection', shape=(num_classes,), initializer='zeros')
        self.union = self.add_weight(name='union', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true)  # Remove extra dimension if present
        
        for i in range(self.num_classes):
            true_i = tf.cast(tf.equal(y_true, i), tf.float32)
            pred_i = tf.cast(tf.equal(y_pred, i), tf.float32)
            
            intersection = tf.reduce_sum(true_i * pred_i)
            union = tf.reduce_sum(true_i) + tf.reduce_sum(pred_i) - intersection
            
            self.intersection[i].assign(self.intersection[i] + intersection)
            self.union[i].assign(self.union[i] + union)

    def result(self):
        return tf.reduce_mean(self.intersection / (self.union + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.intersection.assign(tf.zeros_like(self.intersection))
        self.union.assign(tf.zeros_like(self.union))

class DiceCoefficient(keras.metrics.Metric):
    def __init__(self, num_classes, name='dice', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.intersection = self.add_weight(name='intersection', shape=(num_classes,), initializer='zeros')
        self.sum = self.add_weight(name='sum', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true)  # Remove extra dimension if present
        
        for i in range(self.num_classes):
            true_i = tf.cast(tf.equal(y_true, i), tf.float32)
            pred_i = tf.cast(tf.equal(y_pred, i), tf.float32)
            
            intersection = tf.reduce_sum(true_i * pred_i)
            sum_total = tf.reduce_sum(true_i) + tf.reduce_sum(pred_i)
            
            self.intersection[i].assign(self.intersection[i] + intersection)
            self.sum[i].assign(self.sum[i] + sum_total)

    def result(self):
        return tf.reduce_mean((2. * self.intersection) / (self.sum + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.intersection.assign(tf.zeros_like(self.intersection))
        self.sum.assign(tf.zeros_like(self.sum))

class PerClassAccuracy(keras.metrics.Metric):
    def __init__(self, num_classes, name='per_class_accuracy', **kwargs):
        super(PerClassAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.total = self.add_weight(name='total', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true)  # Remove extra dimension if present
        
        for i in range(self.num_classes):
            true_i = tf.cast(tf.equal(y_true, i), tf.float32)
            pred_i = tf.cast(tf.equal(y_pred, i), tf.float32)
            
            tp = tf.reduce_sum(true_i * pred_i)
            total = tf.reduce_sum(true_i)
            
            self.true_positives[i].assign(self.true_positives[i] + tp)
            self.total[i].assign(self.total[i] + total)

    def result(self):
        return tf.reduce_mean(self.true_positives / (self.total + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.total.assign(tf.zeros_like(self.total))

# --- DATA LOADING & PREPROCESSING ---
def load_image(path, size=IMG_SIZE, mode='RGB'):
    img = Image.open(path).convert(mode)
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)

def load_depth(path, size=IMG_SIZE):
    img = Image.open(path).convert('L')  # Grayscale
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)[..., np.newaxis]  # Add channel axis

def load_mask(annotation_path, size=IMG_SIZE):
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Create a blank mask
        mask = Image.new('L', size, 0)  # Start with background (0)
        draw = ImageDraw.Draw(mask)
        
        # Draw each polygon
        for part in data.get('parts', []):
            if 'points' not in part or 'label' not in part:
                continue
                
            points = [(x * size[0] / 640, y * size[1] / 480) for x, y in part['points']]
            label = part['label']
            class_idx = CLASS_MAP.get(label, 0)  # Default to background if label not found
            draw.polygon(points, outline=class_idx, fill=class_idx)
            
        return np.array(mask)
    except Exception as e:
        print(f"Error loading mask from {annotation_path}: {str(e)}")
        # Return a blank mask in case of error
        return np.zeros(size, dtype=np.uint8)

def get_file_ids():
    rgb_files = [f for f in os.listdir(RGB_DIR) if f.endswith('.JPG')]
    ids = [os.path.splitext(f)[0] for f in rgb_files]
    return ids

def train_val_split(ids, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    ids = np.array(ids)
    np.random.shuffle(ids)
    split = int(len(ids) * (1 - val_ratio))
    return ids[:split], ids[split:]

def overlay_mask(image, mask, alpha=0.5):
    # image: HxWx3, mask: HxW
    color_mask = np.zeros_like(image)
    # Assign a color for each class
    colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255)]  # Background, Shelf, Door, drawer
    for idx, color in enumerate(colors):
        color_mask[mask==idx] = color
    overlay = image.copy()
    overlay = np.array(overlay * (1-alpha) + color_mask * alpha, dtype=np.uint8)
    return overlay

def visualize_samples(ids, num_samples=10):
    # Randomly select samples
    selected_ids = np.random.choice(ids, min(num_samples, len(ids)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, id_ in enumerate(selected_ids):
        try:
            # Load image and mask
            rgb = load_image(os.path.join(RGB_DIR, f'{id_}.JPG'))
            mask = load_mask(os.path.join(ANNOTATIONS_DIR, f'{id_}_annotation.json'))
            
            # Create overlay
            overlay = overlay_mask(rgb, mask)
            
            # Display
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'Image {id_}')
            axes[idx].axis('off')
        except Exception as e:
            print(f"Error visualizing {id_}: {str(e)}")
            continue
    
    plt.tight_layout()
    # Save the plot instead of showing it
    plt.savefig('training_samples_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved training samples visualization to 'training_samples_overlay.png'")

# --- DATA GENERATOR (with class weights) ---
def data_generator(ids, batch_size=4, class_weights=None):
    # Pre-load all images and masks into memory
    print("Pre-loading data...")
    all_images = {}
    all_depths = {}
    all_masks = {}
    
    for id_ in ids:
        try:
            rgb = load_image(os.path.join(RGB_DIR, f'{id_}.JPG'))
            depth = load_depth(os.path.join(DEPTH_DIR, f'{id_}_depth.png'))
            mask = load_mask(os.path.join(ANNOTATIONS_DIR, f'{id_}_annotation.json'))
            
            all_images[id_] = rgb
            all_depths[id_] = depth
            all_masks[id_] = mask
        except Exception as e:
            print(f"Error loading {id_}: {str(e)}")
            continue
    
    print("Data pre-loading complete!")
    
    while True:
        # Shuffle IDs for each epoch
        np.random.shuffle(ids)
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            imgs, depths, masks = [], [], []
            
            for id_ in batch_ids:
                if id_ in all_images:
                    imgs.append(all_images[id_])
                    depths.append(all_depths[id_])
                    masks.append(all_masks[id_])
            
            if not imgs:
                continue
                
            X = np.concatenate([np.array(imgs), np.array(depths)], axis=-1)
            y = np.array(masks)
            X = X.astype('float32') / 255.0
            y = y.astype('int32')
            
            # Apply class weights to the loss
            if class_weights is not None:
                sample_weights = np.ones_like(y, dtype=np.float32)
                for class_idx, weight in class_weights.items():
                    sample_weights[y == class_idx] = weight
                yield X, y, sample_weights
            else:
                yield X, y

# --- MODEL DEFINITION (ResNet50 U-Net) ---
def resnet_unet_model(input_shape, num_classes):
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Split input into RGB and depth
    rgb_input = inputs[..., :3]  # RGB channels
    depth_input = inputs[..., 3:]  # Depth channel
    
    # Preprocess RGB for ResNet50
    x_rgb = keras.applications.resnet50.preprocess_input(rgb_input)
    
    # Load pre-trained ResNet50 without top layers
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape[0], input_shape[1], 3)  # RGB input
    )
    
    # Freeze first 15 layers
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    
    # Process RGB through ResNet50
    x_rgb = base_model(x_rgb)  # This will be (None, 8, 8, 2048)
    
    # Process depth through separate conv layers
    x_depth = layers.Conv2D(64, 3, padding='same', activation='relu')(depth_input)
    x_depth = layers.MaxPooling2D(pool_size=(2, 2))(x_depth)  # 128x128
    x_depth = layers.Conv2D(128, 3, padding='same', activation='relu')(x_depth)
    x_depth = layers.MaxPooling2D(pool_size=(2, 2))(x_depth)  # 64x64
    x_depth = layers.Conv2D(256, 3, padding='same', activation='relu')(x_depth)
    x_depth = layers.MaxPooling2D(pool_size=(2, 2))(x_depth)  # 32x32
    x_depth = layers.Conv2D(512, 3, padding='same', activation='relu')(x_depth)
    x_depth = layers.MaxPooling2D(pool_size=(2, 2))(x_depth)  # 16x16
    x_depth = layers.Conv2D(1024, 3, padding='same', activation='relu')(x_depth)
    x_depth = layers.MaxPooling2D(pool_size=(2, 2))(x_depth)  # 8x8
    
    # Adjust depth features to match RGB features
    x_depth = layers.Conv2D(2048, 1, padding='same', activation='relu')(x_depth)
    
    # Combine RGB and depth features
    x = layers.concatenate([x_rgb, x_depth])
    
    # Decoder
    # Level 5 -> 4
    up5 = layers.UpSampling2D(size=(2, 2))(x)
    conv5_up = layers.Conv2D(512, 3, activation='relu', padding='same')(up5)
    conv5_up = layers.Conv2D(512, 3, activation='relu', padding='same')(conv5_up)
    
    # Level 4 -> 3
    up4 = layers.UpSampling2D(size=(2, 2))(conv5_up)
    conv4_up = layers.Conv2D(256, 3, activation='relu', padding='same')(up4)
    conv4_up = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4_up)
    
    # Level 3 -> 2
    up3 = layers.UpSampling2D(size=(2, 2))(conv4_up)
    conv3_up = layers.Conv2D(128, 3, activation='relu', padding='same')(up3)
    conv3_up = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3_up)
    
    # Level 2 -> 1
    up2 = layers.UpSampling2D(size=(2, 2))(conv3_up)
    conv2_up = layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
    conv2_up = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2_up)
    
    # Final upsampling to original size
    up1 = layers.UpSampling2D(size=(2, 2))(conv2_up)
    conv1_up = layers.Conv2D(32, 3, activation='relu', padding='same')(up1)
    conv1_up = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1_up)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv1_up)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def visualize_validation_results(model, val_ids, num_samples=5):
    # Randomly select validation samples
    selected_ids = np.random.choice(val_ids, min(num_samples, len(val_ids)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for idx, id_ in enumerate(selected_ids):
        try:
            # Load original image and create input
            rgb = load_image(os.path.join(RGB_DIR, f'{id_}.JPG'))
            depth = load_depth(os.path.join(DEPTH_DIR, f'{id_}_depth.png'))
            X = np.concatenate([rgb, depth], axis=-1)
            X = X.astype('float32') / 255.0
            X = np.expand_dims(X, axis=0)
            
            # Get prediction
            pred = model.predict(X, verbose=0)[0]
            pred_mask = np.argmax(pred, axis=-1)
            
            # Create colored overlay
            color_mask = np.zeros_like(rgb)
            colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255)]  # Background, Shelf, Door, Drawer
            for class_idx, color in enumerate(colors):
                color_mask[pred_mask==class_idx] = color
            
            overlay = rgb.copy()
            overlay = np.array(overlay * 0.7 + color_mask * 0.3, dtype=np.uint8)
            
            # Display original image
            axes[idx, 0].imshow(rgb)
            axes[idx, 0].set_title(f'Original Image {id_}')
            axes[idx, 0].axis('off')
            
            # Display predicted mask
            axes[idx, 1].imshow(pred_mask)
            axes[idx, 1].set_title('Predicted Mask')
            axes[idx, 1].axis('off')
            
            # Display overlay
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title('Overlay with Classes')
            axes[idx, 2].axis('off')
            
            # Add class labels
            class_counts = np.bincount(pred_mask.flatten(), minlength=len(CLASSES))
            class_text = '\n'.join([f'{CLASSES[i]}: {count} pixels' for i, count in enumerate(class_counts) if count > 0])
            axes[idx, 2].text(0.02, 0.98, class_text,
                            transform=axes[idx, 2].transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        except Exception as e:
            print(f"Error visualizing {id_}: {str(e)}")
            continue
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved validation results visualization to 'validation_results.png'")

# --- CUSTOM CALLBACKS ---
class ValidationVisualizationCallback(keras.callbacks.Callback):
    def __init__(self, val_ids, num_samples=5):
        super().__init__()
        self.val_ids = val_ids
        self.num_samples = num_samples
        
    def on_epoch_end(self, epoch, logs=None):
        # Randomly select validation samples
        selected_ids = np.random.choice(self.val_ids, min(self.num_samples, len(self.val_ids)), replace=False)
        
        # Create figure
        fig, axes = plt.subplots(self.num_samples, 4, figsize=(20, 5*self.num_samples))
        
        for idx, id_ in enumerate(selected_ids):
            try:
                # Load original image and ground truth
                rgb = load_image(os.path.join(RGB_DIR, f'{id_}.JPG'))
                depth = load_depth(os.path.join(DEPTH_DIR, f'{id_}_depth.png'))
                true_mask = load_mask(os.path.join(ANNOTATIONS_DIR, f'{id_}_annotation.json'))
                
                # Prepare input for prediction
                X = np.concatenate([rgb, depth], axis=-1)
                X = X.astype('float32') / 255.0
                X = np.expand_dims(X, axis=0)
                
                # Get prediction
                pred = self.model.predict(X, verbose=0)[0]
                pred_mask = np.argmax(pred, axis=-1)
                
                # Create overlays
                true_overlay = overlay_mask(rgb, true_mask)
                pred_overlay = overlay_mask(rgb, pred_mask)
                
                # Display original image
                axes[idx, 0].imshow(rgb)
                axes[idx, 0].set_title(f'Original Image {id_}')
                axes[idx, 0].axis('off')
                
                # Display ground truth mask
                axes[idx, 1].imshow(true_mask)
                axes[idx, 1].set_title('Ground Truth Mask')
                axes[idx, 1].axis('off')
                
                # Display ground truth overlay
                axes[idx, 2].imshow(true_overlay)
                axes[idx, 2].set_title('Ground Truth Overlay')
                axes[idx, 2].axis('off')
                
                # Display predicted overlay
                axes[idx, 3].imshow(pred_overlay)
                axes[idx, 3].set_title('Predicted Overlay')
                axes[idx, 3].axis('off')
                
                # Add class labels and pixel counts
                true_counts = np.bincount(true_mask.flatten(), minlength=len(CLASSES))
                pred_counts = np.bincount(pred_mask.flatten(), minlength=len(CLASSES))
                
                class_text = 'Ground Truth vs Predicted:\n'
                for i, class_name in enumerate(CLASSES):
                    if true_counts[i] > 0 or pred_counts[i] > 0:
                        class_text += f'{class_name}: {true_counts[i]} vs {pred_counts[i]} pixels\n'
                
                axes[idx, 3].text(0.02, 0.98, class_text,
                                transform=axes[idx, 3].transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error visualizing {id_}: {str(e)}")
                continue
        
        plt.tight_layout()
        plt.savefig(f'validation_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved validation visualization for epoch {epoch+1}")

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    # Enable mixed precision training
    keras.mixed_precision.set_global_policy('mixed_float16')
    
    ids = get_file_ids()
    train_ids, val_ids = train_val_split(ids)

    # Visualize random training samples
    print("Visualizing random training samples...")
    visualize_samples(train_ids, num_samples=10)

    # Model setup
    BATCH_SIZE = 4  # Reduced batch size due to larger model
    EPOCHS = 50
    input_shape = (IMG_SIZE[1], IMG_SIZE[0], 4)
    num_classes = len(CLASSES)
    
    # Calculate class weights
    print("Calculating class weights...")
    class_weights = {}
    total_pixels = 0
    class_pixels = np.zeros(num_classes)
    
    for id_ in train_ids:
        try:
            mask = load_mask(os.path.join(ANNOTATIONS_DIR, f'{id_}_annotation.json'))
            for i in range(num_classes):
                class_pixels[i] += np.sum(mask == i)
            total_pixels += mask.size
        except Exception as e:
            print(f"Error processing {id_}: {str(e)}")
            continue
    
    # Calculate weights (higher for non-background classes)
    for i in range(num_classes):
        if i == 0:  # Background
            class_weights[i] = 1.0
        else:
            # Higher weight for non-background classes
            class_weights[i] = 5.0 * (total_pixels / (num_classes * class_pixels[i] + 1e-6))
    
    print("Class weights:", class_weights)
    
    # Create and compile model
    model = resnet_unet_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Compile model with custom metrics and optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            IoU(num_classes=num_classes, name='iou'),
            DiceCoefficient(num_classes=num_classes, name='dice'),
            PerClassAccuracy(num_classes=num_classes, name='per_class_accuracy')
        ]
    )

    # Add callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More aggressive learning rate reduction
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_iou',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ValidationVisualizationCallback(val_ids, num_samples=5)
    ]

    train_gen = data_generator(train_ids, BATCH_SIZE, class_weights)
    val_gen = data_generator(val_ids, BATCH_SIZE)  # No class weights for validation
    steps_per_epoch = max(1, len(train_ids) // BATCH_SIZE)
    val_steps = max(1, len(val_ids) // BATCH_SIZE)

    print('Starting training...')
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2
    )

    # Print final metrics
    print("\nFinal Evaluation Metrics:")
    print(f"IoU Score: {history.history['val_iou'][-1]:.4f}")
    print(f"Dice Coefficient: {history.history['val_dice'][-1]:.4f}")
    print(f"Per-Class Accuracy: {history.history['val_per_class_accuracy'][-1]:.4f}")
    print(f"Overall Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    model.save('segmentation_model.h5')
    print('Model saved as segmentation_model.h5')

# --- USAGE ---
# Activate your venv before running:
# $ source venv/bin/activate
# $ python train_segmentation.py 
