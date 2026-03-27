import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import shutil
import glob
import json

# Define constants - REDUCED FOR FASTER TRAINING
BATCH_SIZE = 16  # Reduced from 32 to 16
IMG_HEIGHT = 180
IMG_WIDTH = 180
DATASET_PATH = '/home/purandar/dataset_blood_group'
CHECKPOINT_DIR = 'checkpoints'
MODEL_FILE = 'blood_group_classifier.h5'
METADATA_FILE = 'training_metadata.json'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Batch Size: {BATCH_SIZE} (reduced for faster training)")
print(f"Dataset Location: {DATASET_PATH}")
print(f"{'='*60}\n")

# Check if we're resuming
resume_training = os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE)

if resume_training:
    print("🔄 Resuming previous training...\n")
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    start_epoch = metadata.get('completed_epochs', 0)
    print(f"Previous training completed {start_epoch} epochs")
else:
    print("🆕 Starting fresh training...\n")
    start_epoch = 0

TRAIN_DIR = 'blood_group_dataset/train'
VAL_DIR = 'blood_group_dataset/val'

if not resume_training:
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR):
        shutil.rmtree(VAL_DIR)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

if not resume_training:
    print("Preparing dataset...")
    print("-" * 60)

    total_train = 0
    total_val = 0

    for blood_type in classes:
        source_dir = os.path.join(DATASET_PATH, blood_type)
        
        if not os.path.exists(source_dir):
            continue
        
        train_class_dir = os.path.join(TRAIN_DIR, blood_type)
        val_class_dir = os.path.join(VAL_DIR, blood_type)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        images = []
        for ext in ['*.bmp', '*.BMP', '*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(glob.glob(os.path.join(source_dir, ext)))
        
        images = [os.path.basename(img) for img in images]
        
        if len(images) == 0:
            continue
        
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        copied_train = 0
        for img in train_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(train_class_dir, img)
            try:
                shutil.copy2(src, dst)
                copied_train += 1
            except:
                pass
        
        copied_val = 0
        for img in val_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(val_class_dir, img)
            try:
                shutil.copy2(src, dst)
                copied_val += 1
            except:
                pass
        
        total_train += copied_train
        total_val += copied_val
        
        print(f"✓ {blood_type:5s}: {copied_train:4d} train, {copied_val:4d} val")

    print("-" * 60)
    print(f"Total: {total_train} training images, {total_val} validation images\n")

print("Loading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    verbose=0
)
print(f"✓ Training dataset: {len(train_ds)} batches")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    verbose=0
)
print(f"✓ Validation dataset: {len(val_ds)} batches\n")

print("Normalizing images...")
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
print("✓ Images normalized\n")

print("Building model...")
if resume_training:
    model = keras.models.load_model(MODEL_FILE)
    print(f"✓ Loaded existing model")
else:
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(8, activation='softmax')
    ])
    print("✓ Model created")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*60)
print("Starting training...")
print("="*60 + "\n")

checkpoint = ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR, 'model_epoch_{epoch:02d}.h5'),
    save_freq='epoch',
    verbose=0
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,  # Reduced from 20 to 10 epochs
    initial_epoch=start_epoch,
    callbacks=[checkpoint],
    verbose=1
)

print("\n" + "="*60)
model.save(MODEL_FILE)
print(f"✓ Model saved as '{MODEL_FILE}'")

completed_epochs = start_epoch + len(history.history['loss'])
metadata = {
    'completed_epochs': completed_epochs,
    'total_epochs': 10,
    'final_accuracy': float(history.history['accuracy'][-1]),
    'final_val_accuracy': float(history.history['val_accuracy'][-1])
}

with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
print("="*60 + "\n")
