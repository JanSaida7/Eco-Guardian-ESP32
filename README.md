# Eco-Guardian: Edge-AI Acoustic Surveillance

**Project Title**: Eco-Guardian: Real-Time Acoustic Event Detection on ESP32
**Target Device**: ESP32-S3 (Tensilica Xtensa LX7)
**Framework**: TensorFlow Lite for Microcontrollers (TFLM)

## 🌍 Mission & SDG Goals
This project contributes to **SDG 15: Life on Land** by providing a low-cost, scalable solution for real-time forest monitoring.
- **Target 15.2**: Promote the implementation of sustainable management of all types of forests, halt deforestation, restore degraded forests and substantially increase afforestation and reforestation globally.
- **Target 15.7**: Take urgent action to end poaching and trafficking of protected species of flora and fauna and address both demand and supply of illegal wildlife products.

## 🎯 Objectives
1.  **Detect Illegal Activity**: Identify "Gunshots" (poaching) and "Chainsaws" (illegal logging) in real-time.
2.  **Edge Compute**: Run fully on-device (ESP32) without cloud dependency for privacy and battery efficiency.
3.  **Lightweight**: Utilize a **Depthwise Separable CNN** (MobileNet-style) to minimize RAM/Flash usage.

## 📂 Project Structure
- `data/`: Place your raw .wav files here (Structure: `gunshot/`, `chainsaw/`, `background/`).
- `src/`: Python scripts for Preprocessing, Training, and Conversion.
  - `0_download_data.py`: Downloads ESC-50 dataset and extracts relevant categories.
  - `1_preprocess.py`: Extracts Mel-Spectrograms.
  - `2_train.py`: Trains the DS-CNN model.
  - `3_convert.py`: Converts model to TFLite/C++.
  - `4_generate_synthetic_data.py`: Generates synthetic training data with noise.
  - `debug_categories.py`: Helper to check ESC-50 categories.
  - `demo_laptop_mic.py`: Real-time demo using laptop microphone.
  - `utils.py`: Shared constants and configuration.
- `models/`: Stores trained models (`.h5`, `.tflite`) and performance graphs.

## 🚀 Setup Instructions
1.  **Install Dependencies**: `pip install -r requirements.txt`
2.  **Add Data**: 
    - **Option A (Automatic)**: Run `python -m src.0_download_data` to download relevant ESC-50 samples.
    - **Option B (Manual)**: Download **ESC-50** or **UrbanSound8K** datasets and populate the `data/` folders manually.
3.  **Run Pipeline**:
    *Note: Run all scripts from the project root using `python -m src.<script_name>` to ensure imports work correctly.*
    - `python -m src.4_generate_synthetic_data` (Generate synthetic training data with noise augmentation)
    - `python -m src.1_preprocess` (Extract Mel-Spectrograms from synthetic data)
    - `python -m src.2_train` (Train DS-CNN & Generate Confusion Matrix)
    - `python -m src.3_convert` (Quantize & Convert to C++)
4.  **Run Demo**:
    - `python -m src.demo_laptop_mic` (Test model in real-time)

## 📊 Experimental Results
The training script automatically generates:
- `models/confusion_matrix.png`: Visual proof of classification performance.
- `models/training_history.csv`: Loss/Accuracy curves.

## ✅ Verification
To verify that the project structure and dependencies are correctly set up, run:
```bash
python -m src.verify_setup
```
This script checks for the existence of necessary files and directories.
