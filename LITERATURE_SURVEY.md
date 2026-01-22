# Eco-Guardian: Literature Survey & Research Strategy

## 1. Introduction & Scope
**Objective**: Detect "Gunshots" (Poaching) and "Chainsaws" (Illegal Logging) in real-time on an ESP32-S3.
**Constraints**: 
- **Offline**: No Cloud/4G.
- **Low Power**: Solar/Battery operation.
- **Hardware**: ESP32-S3 (Limited RAM/Compute compared to Raspberry Pi).

## 2. Literature Survey (Why others failed/succeeded)

### Paper 1: "Automated detection of gunshots using CNNs" (Katsis et al., 2022)
- **Method**: ResNet-18 on Spectrograms.
- **Success**: High Accuracy (95%).
- **Failure for us**: **Too Heavy**. ResNet-18 has ~11M parameters. It cannot run on an ESP32. It requires a Raspberry Pi or Cloud server.

### Paper 2: "SafeGuard Wild: IoT Anti-Poaching" (Ananth et al., 2025)
- **Method**: Raspberry Pi + Telegram Alerts.
- **Success**: Integrated system.
- **Failure for us**: **High Power**. Raspberry Pi consumes Watts. ESP32 consumes Milliwatts. Pi is not viable for long-term solar deployment in deep forests.

### Paper 3: "Trees Have Ears" (Lorenzo et al., 2024)
- **Method**: RP2040 Microcontroller for Chainsaws.
- **Success**: Proven viability on small chips.
- **Gap**: Did not leverage **vector instructions** (ESP32-S3 has DSP instructions for faster AI) and focused only on chainsaws.

## 3. The "Eco-Guardian" Strategy (Our Approach)

We are bridging the gap between "High Accuracy/Heavy Compute" (Paper 1) and "Low Power/Low Compute" (Paper 3).

### Implementation Strategy
1.  **Architecture**: **Depthwise Separable CNN (DS-CNN)**.
    -   *Why?* Reduces parameters by ~90% vs standard CNNs (MobileNet concept). Fits in ESP32 RAM.
2.  **Data Strategy**: **Synthetic Noise Mixing**.
    -   *Why?* Real-world forest audio is noisy. Training on clean data (ESC-50) fails (as seen in our demo).
    -   *Plan*: We will generate **2,000+ samples** by mathematically mixing "clean" chainsaw sounds with "forest background" noise at different volumes. This forces the model to learn the *sound signature*, not just the file.
3.  **Deployment**: **Int8 Quantization**.
    -   *Why?* Reduces model size by 4x, allowing larger/smarter models to fit in the same space.

## 4. Current Status & Next Steps
- **Current Status**: Pipeline working, but model **under-fits** real-world noisy data (Demo failed).
- **Next Step**: Implement `src/4_generate_synthetic_data.py` to blindly create thousands of noisy training examples.
