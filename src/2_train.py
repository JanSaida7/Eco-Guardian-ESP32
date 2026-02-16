import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import DATA_DIR, CLASSES, MODEL_DIR, INPUT_SHAPE

# Constants
BATCH_SIZE = 128
EPOCHS = 3 # Reduced for faster turnover, dataset is large enough for quick convergence
# INPUT_SHAPE imported from utils

def build_ds_cnn(input_shape, num_classes):
    """
    Builds a Depthwise Separable CNN models optimized for Edge Devices (ESP32).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Standard Conv2D for initial feature extraction
        layers.Conv2D(16, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # DS-CNN Block 1
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (1, 1), padding='same', use_bias=False), # Pointwise
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # DS-CNN Block 2
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (1, 1), padding='same', use_bias=False), # Pointwise
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    print(f"Confusion Matrix saved to {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")

def main():
    # Load Data
    try:
        X = np.load(os.path.join(DATA_DIR, "X.npy"))
        y = np.load(os.path.join(DATA_DIR, "y.npy"))
    except FileNotFoundError:
        print("Error: X.npy or y.npy not found. Run 1_preprocess.py first.")
        return

    # Check Input Shape
    print(f"Data Shape: {X.shape}")
    
    # One-hot encode labels
    y_onehot = to_categorical(y, num_classes=len(CLASSES))

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Build Model
    model = build_ds_cnn(input_shape=X.shape[1:], num_classes=len(CLASSES))
    model.summary()
    
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate
    print("\n--- Evaluation ---")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Generate Report & Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))

    plot_confusion_matrix(y_true_classes, y_pred_classes, CLASSES)

    # Save Model
    model.save(os.path.join(MODEL_DIR, "forest_guard.h5"))
    print(f"Model saved to {os.path.join(MODEL_DIR, 'forest_guard.h5')}")

if __name__ == "__main__":
    main()
