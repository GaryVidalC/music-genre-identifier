from pathlib import Path
import numpy as np
import sys
import librosa
import json
import argparse
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_processing import extract_audio_features, standardize_signal

# path from which to load a specific wav file for testing

parser = argparse.ArgumentParser(description="Test the trained model with a specific audio file.")
parser.add_argument("audio_file", type=str, help="Path to the audio file to test.")
args = parser.parse_args()
audio_file_path = Path(args.audio_file)

# Load metadata

with (PROJECT_ROOT / 'model/genre_encoder.pkl').open('rb') as f:  
    encoder = pickle.load(f)

metadata = json.load((PROJECT_ROOT / 'model/metadata.json').open('r', encoding='utf-8'))

model_name = metadata['model']['model_used'].lower()

with (PROJECT_ROOT / f'model/{model_name}_model.pkl').open('rb') as f:
    model = pickle.load(f)

# Load and preprocess the audio file

signal, sr = librosa.load(audio_file_path, 
                        sr=metadata['preprocessing']['sample_rate'], 
                        mono=metadata['preprocessing']['mono'], 
                        duration=metadata['preprocessing']['duration'])

signal = standardize_signal(signal, 
                            metadata['preprocessing']['sample_rate'], 
                            metadata['preprocessing']['duration'],
                            metadata['preprocessing']['normalization'])

features = extract_audio_features(signal, sr, metadata['preprocessing']['n_mfcc'])

# Extract feature values in the same order as the training data

feature_values = [features[feature] for feature in metadata['preprocessing']['features']]
feature_array = np.array(feature_values).reshape(1, -1)

# load model and make prediction

pred = model.predict(feature_array)
pred_genre = encoder.inverse_transform(pred)[0]

print(pred_genre)