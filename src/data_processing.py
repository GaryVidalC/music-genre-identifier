import librosa
import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle

from audio_processing import extract_audio_features, standardize_signal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

with (PROJECT_ROOT / 'configs/read_config.json').open('r', encoding='utf-8') as f:
    CONFIG = json.load(f)

data_path = PROJECT_ROOT / CONFIG['data_path']
data_records = []
skipped_files = []

for genre_path in data_path.iterdir():
    if not genre_path.is_dir():
        continue
    for audio_file in genre_path.glob('*.wav'):
        try:
            signal, sr = librosa.load(audio_file, sr=CONFIG['sample_rate'],
                                      mono=CONFIG['mono'], duration=CONFIG['duration'])
            signal = standardize_signal(signal, CONFIG['sample_rate'],
                                       CONFIG['duration'], CONFIG['normalization'])

            features = extract_audio_features(signal, sr, CONFIG['n_mfcc'])
            features['genre'] = genre_path.name
            data_records.append(features)
        except Exception as e:
            skipped_files.append(str(audio_file))
            continue

df = pd.DataFrame(data_records)

# encode genres
encoder = LabelEncoder()
df['genre'] = encoder.fit_transform(df['genre'])

# save encoder
with (PROJECT_ROOT / 'model/genre_encoder.pkl').open('wb') as f:
    pickle.dump(encoder, f)

#save dataset
df.to_parquet(PROJECT_ROOT / 'processed_data/features.parquet')
print(df.head())
print(f"saved dataset: {len(df)} samples, {len(df.columns)} features")
print(f"Skipped files: {len(skipped_files)}")


