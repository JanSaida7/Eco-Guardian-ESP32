import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import queue
import time
import winsound  # For Windows Beep

# Constants
from src.utils import MODEL_DIR, CLASSES, SAMPLE_RATE, DURATION

# Constants
MODEL_PATH = os.path.join(MODEL_DIR, "forest_guard.h5")
# CLASSES, SAMPLE_RATE, DURATION imported from utils
BLOCK_SIZE = int(SAMPLE_RATE * DURATION)
THRESHOLD = 0.6 # Confidence threshold - lowered for better sensitivity

# Sliding Window Constants
WINDOW_STEP = 0.5 # Seconds (Overlap = DURATION - WINDOW_STEP)
STEP_SIZE = int(SAMPLE_RATE * WINDOW_STEP)

# Audio Buffer (Rolling)
# We initialized it with zeros
audio_buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)

# Audio Queue
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio."""
    if status:
        print(status)
    # Add incoming audio to the queue
    audio_queue.put(indata.copy())

def preprocess_audio(audio_buffer):
    """Convert raw audio buffer to Mel-Spectrogram."""
    # Flatten buffer
    audio = audio_buffer.flatten()
    
    # Extract Mel-Spectrogram (Same logic as 1_preprocess.py)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SAMPLE_RATE, 
        n_mels=64, 
        n_fft=1024, 
        hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Ensure shape matches model input (1, 64, 63, 1)
    # Resize/Pad if necessary (simple resizing for real-time stability)
    expected_width = 63
    current_width = mel_spec_db.shape[1]
    
    if current_width < expected_width:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, expected_width - current_width)))
    elif current_width > expected_width:
        mel_spec_db = mel_spec_db[:, :expected_width]
        
    return mel_spec_db.reshape(1, 64, 63, 1)

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

                # DEBUG: Calculate volume
                rms = np.sqrt(np.mean(audio_buffer**2))

                # DEBUG: Print raw probabilities to diagnose "Why is it failing?"
                bg_conf = prediction[0][0]
                chain_conf = prediction[0][1]
                gun_conf = prediction[0][2]
                
                # Dynamic print row (overwrites previous line if supported, or just prints)
                print(f"[DEBUG] Vol:{rms:.3f} | BG:{bg_conf:.2f} | CHAIN:{chain_conf:.2f} | GUN:{gun_conf:.2f}")

                # Display Result
                if confidence > THRESHOLD:
                    if label in ["gunshot", "chainsaw"]:
                        print(f"\n\033[91m[DANGER] >>> DETECTED: {label.upper()} ({confidence:.2f}) <<<\033[0m")
                        if confidence > 0.8:
                             winsound.Beep(1000, 200) 
                    else:
                        pass 
                else:
                    pass
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
