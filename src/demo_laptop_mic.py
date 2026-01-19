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

    # Start Recording Stream
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            try:
                # Get audio block from queue
                audio_block = audio_queue.get()
                
                # Preprocess
                input_data = preprocess_audio(audio_block)
                
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
                        print(f"\033[93m>>> Sending Alert to Forest Ranger: 'Illegal Activity: {label}'\033[0m")
                        winsound.Beep(1000, 500) # Frequency 1000Hz, Duration 500ms
                    else:
                        # SAFE
                        print(f"\033[92m[SAFE] Detected: {label.upper()} ({confidence:.2f})\033[0m")
                else:
                    print(f"[UNCERTAIN] ({confidence:.2f})")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
