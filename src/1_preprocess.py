import os
import numpy as np
import librosa
from tqdm import tqdm

# Constants
DATA_DIR = "./data"
SAMPLE_RATE = 16000
DURATION = 2.0  # Seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

CLASSES = ["background", "chainsaw", "gunshot"]

def extract_features(file_path):
    try:
        # Load audio (automatically resamples to 16kHz)
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate to ensure consistent length
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # Extract Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Shape: (n_mels, time_steps) -> (n_mels, time_steps, 1) for CNN
        return mel_spec_db[..., np.newaxis]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    X, y = [], []
    
    print("Starting Feature Extraction...")
    for idx, label in enumerate(CLASSES):
        folder_path = os.path.join(DATA_DIR, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        for file in tqdm(files, desc=f"Processing {label}"):
            file_path = os.path.join(folder_path, file)
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(idx)

    X = np.array(X)
    y = np.array(y)

    print(f"Feature Extraction Complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    print("Features saved to ./data/X.npy and ./data/y.npy")

if __name__ == "__main__":
    main()
