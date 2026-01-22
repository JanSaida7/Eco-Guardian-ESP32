import os
import numpy as np
import librosa
from tqdm import tqdm

# Constants
# Constants
DATA_DIR = "./data/synthetic_train"
OUTPUT_DIR = "./data"
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

def augment_audio(audio, sr):
    """Generate augmented versions of the audio."""
    augmented_data = []
    
    # 1. Original
    augmented_data.append(audio)
    
    # 2. White Noise
    noise_factor = 0.005
    noise = np.random.randn(len(audio))
    augmented_data.append(audio + noise_factor * noise)
    
    # 3. Time Stretch (Speed up and Slow down) - simplified resampling
    # Note: librosa.effects.time_stretch requires spectrogram or complex audio
    # Using simple resampling for speed variation
    for rate in [0.9, 1.1]:
        try:
             # Resample
             new_len = int(len(audio) / rate)
             # Basic interpolation (using numpy for simplicity to avoid heavy librosa dependency in augmentation loop if possible, 
             # but librosa.resample is safer. Let's stick to simple noise/pitch for stability or use reliable methods)
             # Actually, let's stick to safe augmentations that don't change array size unpredictably
             pass
        except:
             pass

    # 3. Pitch Shift (using librosa - might send warning but effective)
    try:
        # Shift -2 and +2 semitones
        augmented_data.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2.0))
        augmented_data.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2.0))
    except Exception as e:
        pass # Skip if fails
        
    return augmented_data

def main():
    X, y = [], []
    
    print("Starting Feature Extraction with Augmentation...")
    for idx, label in enumerate(CLASSES):
        folder_path = os.path.join(DATA_DIR, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        for file in tqdm(files, desc=f"Processing {label}"):
            file_path = os.path.join(folder_path, file)
            
            try:
                # Load audio once
                raw_audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # Pad/Truncate
                target_length = int(SAMPLE_RATE * DURATION)
                if len(raw_audio) < target_length:
                    raw_audio = np.pad(raw_audio, (0, target_length - len(raw_audio)))
                else:
                    raw_audio = raw_audio[:target_length]
                    
                # Augment
                augmented_versions = augment_audio(raw_audio, SAMPLE_RATE)
                
                # Extract Features for each version
                for audio_ver in augmented_versions:
                    mel_spec = librosa.feature.melspectrogram(y=audio_ver, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    feature = mel_spec_db[..., np.newaxis]
                    
                    X.append(feature)
                    y.append(idx)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Feature Extraction Complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print("Features saved to ./data/X.npy and ./data/y.npy")

if __name__ == "__main__":
    main()
