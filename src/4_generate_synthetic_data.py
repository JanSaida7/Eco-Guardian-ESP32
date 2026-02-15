import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import random

# Constants
from utils import DATA_DIR, CLASSES, SAMPLE_RATE, DURATION

# Constants
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic_train")
TARGET_CLASSES = [c for c in CLASSES if c != "background"]
BACKGROUND_CLASS = "background"
SAMPLES_PER_TARGET = 1000  # Generate 1000 samples per target class

def load_audio_files(label):
    folder = os.path.join(DATA_DIR, label)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
    audio_data = []
    for f in files:
        y, _ = librosa.load(f, sr=SAMPLE_RATE, duration=DURATION)
        # Fix length
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        audio_data.append(y)
    return audio_data

def mix_audio(foreground, background, snr_db):
    """Mix foreground signal with background noise at given SNR."""
    # Calculate power
    fg_power = np.mean(foreground ** 2)
    bg_power = np.mean(background ** 2)
    
    if bg_power == 0:
        return foreground

    # Calculate required background power
    target_bg_power = fg_power / (10 ** (snr_db / 10))
    
    # Scale background
    scale = np.sqrt(target_bg_power / bg_power)
    mixed = foreground + background * scale
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
        
    return mixed

def augment_pitch_speed(audio):
    """Apply random pitch shift or time stretch."""
    if random.random() < 0.5:
        # Pitch shift
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
    else:
        # Time stretch (simple resampling for speed)
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)

def main():
    if not os.path.exists(SYNTHETIC_DIR):
        os.makedirs(SYNTHETIC_DIR)
        
    print("Loading Source Audio...")
    background_clips = load_audio_files(BACKGROUND_CLASS)
    target_clips = {label: load_audio_files(label) for label in TARGET_CLASSES}
    
    if not background_clips:
        print("Error: No background clips found!")
        return

    print(f"Generating {SAMPLES_PER_TARGET} samples per target class...")
    
    for label in TARGET_CLASSES:
        output_folder = os.path.join(SYNTHETIC_DIR, label)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        clips = target_clips[label]
        if not clips:
            print(f"Warning: No clips for {label}")
            continue
            
        for i in tqdm(range(SAMPLES_PER_TARGET), desc=f"Generating {label}"):
            # 1. Pick random foreground (target)
            fore = random.choice(clips)
            
            # 2. Pick random background
            back = random.choice(background_clips)
            
            # 3. Augment foreground first? (Optional, let's keep it simple mixing first)
            
            # 4. HEAVY NOISE SCENARIO (FAN/AC):
            # Range: -10dB (Target buried in noise) to 5dB (Target slightly louder)
            snr = random.uniform(-10, 5)
            mixed = mix_audio(fore, back, snr)
            
            # 5. Apply Pitch/Speed Augmentation
            if random.random() < 0.8: # Increased chance of augmentation
                 try:
                    fore_aug = augment_pitch_speed(fore)
                    # Fix length again after aug
                    target_len = int(SAMPLE_RATE * DURATION)
                    if len(fore_aug) < target_len:
                        fore_aug = np.pad(fore_aug, (0, target_len - len(fore_aug)))
                    else:
                        fore_aug = fore_aug[:target_len]
                    mixed = mix_audio(fore_aug, back, snr)
                 except:
                    pass # Fallback to original mix
            
            # Save
            out_name = f"synth_{label}_{i:04d}_snr{int(snr)}.wav"
            sf.write(os.path.join(output_folder, out_name), mixed, SAMPLE_RATE)
            
    # Also generate "pure" background samples (just augment existing backgrounds)
    # We want the model to have a "Background" class too.
    bg_folder = os.path.join(SYNTHETIC_DIR, BACKGROUND_CLASS)
    if not os.path.exists(bg_folder):
        os.makedirs(bg_folder)
        
    for i in tqdm(range(SAMPLES_PER_TARGET), desc="Generating Background"):
        bg = random.choice(background_clips)
        # Augment
        if random.random() < 0.5:
             # Add volume variation
             gain = random.uniform(0.5, 1.5)
             bg = bg * gain
             
        out_name = f"synth_bg_{i:04d}.wav"
        sf.write(os.path.join(bg_folder, out_name), bg, SAMPLE_RATE)

    print("Synthetic Generation Complete.")

if __name__ == "__main__":
    main()
