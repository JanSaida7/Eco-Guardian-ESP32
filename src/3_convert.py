import os
import numpy as np
import tensorflow as tf

# Constants
DATA_DIR = "./data"
MODEL_DIR = "./models"
H5_MODEL_PATH = os.path.join(MODEL_DIR, "forest_guard.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model_quantized.tflite")
CC_MODEL_PATH = os.path.join(MODEL_DIR, "model_data.cc")

def representative_dataset_gen():
    """Generates a representative dataset for Quantization."""
    try:
        X = np.load(os.path.join(DATA_DIR, "X.npy"))
        # Use a subset of data for calibration
        for i in range(min(100, len(X))):
            # Ensure proper shape (1, 64, 63, 1) and dtype (float32)
            yield [X[i].reshape(1, 64, 63, 1).astype(np.float32)]
    except FileNotFoundError:
        print("Error: X.npy not found. Cannot perform quantization without data.")
        return

def convert_to_c_array(tflite_model, output_path):
    """Converts TFLite binary to a C++ header file."""
    hex_array = ', '.join([f'0x{val:02x}' for val in tflite_model])
    
    c_code = f"""
#include <cstdint>

// TFLite Model Data
// Length: {len(tflite_model)} bytes
alignas(16) const unsigned char g_model[] = {{
    {hex_array}
}};

const unsigned int g_model_len = {len(tflite_model)};
"""
    with open(output_path, "w") as f:
        f.write(c_code)
    print(f"C++ Model saved to {output_path}")

def main():
    if not os.path.exists(H5_MODEL_PATH):
        print(f"Error: Model file {H5_MODEL_PATH} not found. Train the model first.")
        return

    # Load Keras Model
    model = tf.keras.models.load_model(H5_MODEL_PATH)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization: Int8 Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure full integer quantization for ESP32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
        
        # Save TFLite Model
        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)
        print(f"Quantized TFLite Model saved to {TFLITE_MODEL_PATH}")
        print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")

        # Convert to C++
        convert_to_c_array(tflite_model, CC_MODEL_PATH)
        
    except Exception as e:
        print(f"Conversion Failed: {e}")
        # Fallback without quantization if data is missing (for testing script logic only)
        if not os.path.exists(os.path.join(DATA_DIR, "X.npy")):
             print("TIP: You need 'X.npy' from the preprocessing step to perform Int8 Quantization.")

if __name__ == "__main__":
    main()
