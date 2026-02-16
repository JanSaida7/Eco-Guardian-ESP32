import os

DATA_DIR = "data"
CLASSES = ["background", "chainsaw", "gunshot"]
SAMPLE_RATE = 16000
DURATION = 2.0

METADATA_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
AUDIO_BASE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
METADATA_FILENAME = "esc50.csv"

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MODEL_DIR = "models"
INPUT_SHAPE = (64, 63, 1)
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic_train")
