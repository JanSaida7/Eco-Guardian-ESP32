# Eco-Guardian: Edge-AI Acoustic Surveillance

**Project Title**: Eco-Guardian: Real-Time Acoustic Event Detection on ESP32
**Target Device**: ESP32-S3 (Tensilica Xtensa LX7)
**Framework**: TensorFlow Lite for Microcontrollers (TFLM)

## 🌍 Mission & SDG Goals
This project contributes to **SDG 15: Life on Land** by providing a low-cost, scalable solution for real-time forest monitoring.
- **Target 15.2**: Promote the implementation of sustainable management of all types of forests, halt deforestation, restore degraded forests and substantially increase afforestation and reforestation globally.
- **Target 15.7**: Take urgent action to end poaching and trafficking of protected species of flora and fauna and address both demand and supply of illegal wildlife products.

## 🎯 Objectives
1.  **Decect Illegal Activity**: Identify "Gunshots" (poaching) and "Chainsaws" (illegal logging) in real-time.
2.  **Edge Compute**: Run fully on-device (ESP32) without cloud dependency for privacy and battery efficiency.
3.  **Lightweight**: Utilize a **Depthwise Separable CNN** (MobileNet-style) to minimize RAM/Flash usage.

## 📂 Project Structure
- `data/`: Place your raw .wav files here (Structure: `gunshot/`, `chainsaw/`, `background/`).
- `src/`: Python scripts for Preprocessing, Training, and Conversion.
- `models/`: Stores trained models (`.h5`, `.tflite`) and performance graphs.

## 🚀 Setup Instructions
1.  **Install Dependencies**: `pip install -r requirements.txt`
2.  **Add Data**: Download **ESC-50** or **UrbanSound8K** datasets and populate the `data/` folders.
3.  **Run Pipeline**:
    - `python src/4_generate_synthetic_data.py` (Generate synthetic training data with noise augmentation)
    - `python src/1_preprocess.py` (Extract Mel-Spectrograms from synthetic data)
    - `python src/2_train.py` (Train DS-CNN & Generate Confusion Matrix)
    - `python src/3_convert.py` (Quantize & Convert to C++)

## 📊 Experimental Results
The training script automatically generates:
- `models/confusion_matrix.png`: Visual proof of classification performance.
- `models/training_history.csv`: Loss/Accuracy curves.
