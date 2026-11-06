
# Install required packages (for notebook, skip in .py)
# %pip install -U tensorflow pillow scipy matplotlib seaborn scikit-learn

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras import layers, models

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pathlib import Path

# Workspace dataset paths (update if needed)
BASE_DIR = Path(r'c:\Users\USER\Downloads\Bone Fractrure')
train_dir = BASE_DIR / 'Dataset' / 'training'
test_dir = BASE_DIR / 'Dataset' / 'testing'

# Image parameters
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Basic checks
assert train_dir.exists(), f"Training directory not found: {train_dir}"
assert test_dir.exists(), f"Testing directory not found: {test_dir}"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation + normalization with validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Clear, well-structured CNN using Functional API
inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=inputs, outputs=outputs, name='Simple_CNN')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy and loss
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
print("\n=== Training Summary ===")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Test Loss: {test_loss:.2f}")

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Get predictions
test_gen.reset()
predictions = model.predict(test_gen, steps=len(test_gen))
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_gen.classes

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fractured', 'Not Fractured'],
            yticklabels=['Fractured', 'Not Fractured'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(true_classes, predicted_classes, 
                          target_names=['Fractured', 'Not Fractured']))

model.save('fracture_classification_model.keras')

loaded_model = tf.keras.models.load_model('fracture_classification_model.keras')

# Detect fracture from a user-selected image using the saved model
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('fracture_classification_model.keras')

# Open file dialog to select an image
root = tk.Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title='Select X-ray Image', filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])

if file_path:
    # Load and preprocess the image
    img = Image.open(file_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict fracture
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        print(f"Prediction: Not Fractured (Confidence: {prediction:.2f})")
    else:
        print(f"Prediction: Fractured (Confidence: {1-prediction:.2f})")
else:
    print('No image selected.')
