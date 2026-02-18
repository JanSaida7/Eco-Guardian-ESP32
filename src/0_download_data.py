import os
import urllib.request
import csv
import ssl

from src.utils import METADATA_URL, AUDIO_BASE_URL, METADATA_FILENAME as METADATA_FILE, DATA_DIR, ensure_dir

# Target Categories (Map our folder names to ESC-50 categories)
TARGET_MAP = {
    'gunshot': ['fireworks'], # Proxy for gunshot
    'chainsaw': ['chainsaw'],
    'background': ['wind', 'rain', 'crickets']
}

# Limit samples per category to save time/bandwidth (ESC-50 has 40 per class)
SAMPLES_PER_CLASS = 40 

def download_file(url, save_path):
    try:
        # bypass SSL verification for simplicity in some envs
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, open(save_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    print("Downloading metadata...")
    download_file(METADATA_URL, METADATA_FILE)

    # Read CSV
    files_to_download = {'gunshot': [], 'chainsaw': [], 'background': []}
    
    with open(METADATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row['category']
            filename = row['filename']
            
            # Check mappings
            for folder, target_cats in TARGET_MAP.items():
                if category in target_cats:
                    if len(files_to_download[folder]) < SAMPLES_PER_CLASS:
                        files_to_download[folder].append(filename)

    # Download Audio
    print(f"Starting Download ({SAMPLES_PER_CLASS} samples per class)...")
    for folder, files in files_to_download.items():
        save_dir = os.path.join(DATA_DIR, folder)
        ensure_dir(save_dir)
        
        print(f"Downloading {len(files)} files for '{folder}'...")
        for filename in files:
            url = AUDIO_BASE_URL + filename
            save_path = os.path.join(save_dir, filename)
            if not os.path.exists(save_path):
                print(f"Fetching {filename}...")
                download_file(url, save_path)
            else:
                print(f"Skipping {filename} (exists)")
    
    # Cleanup
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
    print("Download Complete.")

if __name__ == "__main__":
    main()
