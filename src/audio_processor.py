import numpy as np
import librosa
import tensorflow as tf
import queue
import sounddevice as sd
import os

# Constants
MODEL_PATH = "./models/forest_guard.h5"
CLASSES = ["background", "chainsaw", "gunshot"]
SAMPLE_RATE = 16000
DURATION = 2.0  # Seconds
BLOCK_SIZE = int(SAMPLE_RATE * DURATION)
WINDOW_STEP = 0.5 # Seconds
STEP_SIZE = int(SAMPLE_RATE * WINDOW_STEP)

class AudioProcessor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)
        self.running = False
        self.stream = None
        
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            print("Loading Model...")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model Loaded.")
        else:
            print(f"Error: Model {self.model_path} not found.")

    def audio_callback(self, indata, frames, time, status):
        """Callback function to capture audio."""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start_stream(self):
        if self.stream is None:
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=STEP_SIZE)
            self.stream.start()
            self.running = True
            print("Audio stream started.")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.running = False
            print("Audio stream stopped.")

    def preprocess_audio(self, audio_buffer):
        """Convert raw audio buffer to Mel-Spectrogram."""
        audio = audio_buffer.flatten()
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=SAMPLE_RATE, 
            n_mels=64, 
            n_fft=1024, 
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        expected_width = 63
        current_width = mel_spec_db.shape[1]
        
        if current_width < expected_width:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, expected_width - current_width)))
        elif current_width > expected_width:
            mel_spec_db = mel_spec_db[:, :expected_width]
            
        return mel_spec_db.reshape(1, 64, 63, 1)

    def process_next_chunk(self, gain=1.0):
        """
        Process the next chunk of audio from the queue.
        Returns: (prediction, confidence, label, rms) or None if queue is empty
        """
        if not self.audio_queue.empty():
            new_data = self.audio_queue.get() * gain
            
            # Update rolling buffer
            self.audio_buffer = np.roll(self.audio_buffer, -STEP_SIZE)
            self.audio_buffer[-STEP_SIZE:] = new_data.flatten()
            
            if self.model:
                input_data = self.preprocess_audio(self.audio_buffer)
                prediction = self.model.predict(input_data, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                label = CLASSES[class_idx]
                rms = np.sqrt(np.mean(self.audio_buffer**2))
                
                return prediction[0], confidence, label, rms
        
        return None
