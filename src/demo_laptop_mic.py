import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import queue
import time
import winsound  # For Windows Beep

# Constants
MODEL_PATH = "./models/forest_guard.h5"
CLASSES = ["background", "chainsaw", "gunshot"]
SAMPLE_RATE = 16000
DURATION = 2.0  # Seconds
BLOCK_SIZE = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.6 # Confidence threshold - lowered for better sensitivity

# Sliding Window Constants
WINDOW_STEP = 0.5 # Seconds (Overlap = DURATION - WINDOW_STEP)
STEP_SIZE = int(SAMPLE_RATE * WINDOW_STEP)

# Audio Buffer (Rolling)
# We initialized it with zeros
audio_buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio."""
    if status:
        print(status)
    # Add incoming audio to the queue
    audio_queue.put(indata.copy())

# ... (preprocess_audio remains same) ...

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found.")
        return

    print("Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Loaded. Starting Real-Time Detection...")
    print("Listening... (Press Ctrl+C to stop)")
    print("-" * 50)

    # Global buffer 
    global audio_buffer

    # Start Recording Stream with smaller blocks
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=STEP_SIZE):
        while True:
            try:
                # Get small block (0.5s)
                new_data = audio_queue.get()
                
                # Update rolling buffer
                # Shift left
                audio_buffer = np.roll(audio_buffer, -STEP_SIZE)
                # Overwrite end with new data (flatten to 1D)
                audio_buffer[-STEP_SIZE:] = new_data.flatten()
                
                # Preprocess current 2s buffer
                input_data = preprocess_audio(audio_buffer)
                
                # Predict
                prediction = model.predict(input_data, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                label = CLASSES[class_idx]

                # Display Result
                if confidence > THRESHOLD:
                    if label in ["gunshot", "chainsaw"]:
                        # DANGER ALERT
                        print(f"\033[91m[DANGER] Detected: {label.upper()} ({confidence:.2f})\033[0m")
                        if confidence > 0.8: # Only beep on high confidence to avoid spam
                             winsound.Beep(1000, 200) 
                    else:
                        pass # Silence for background
                        # print(f"\033[92m[SAFE] Detected: {label.upper()} ({confidence:.2f})\033[0m")
                else:
                    pass # print(f"[UNCERTAIN] ({confidence:.2f})")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
