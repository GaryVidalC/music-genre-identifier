import librosa
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle

with open('configs/read_config.json', 'r') as f:
    CONFIG = json.load(f)

def standardize_signal(signal,sample_rate, duration,normalize):
    objective_samples = sample_rate * duration

    if len(signal) < objective_samples:
        signal = librosa.util.fix_length(signal, size =objective_samples)

    if normalize:
        signal = librosa.util.normalize(signal)

    return signal

def extract_audio_features(signal,sr,n_mfcc):


    features = {}
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    features.update({f'mean_mfcc_{i}': np.mean(mfcc[i]) for i in range(len(mfcc))})
    features.update({f'std_mfcc_{i}': np.std(mfcc[i]) for i in range(len(mfcc))})
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.update({f'mean_chroma_{i}': np.mean(chroma[i]) for i in range(len(chroma))})
    features.update({f'std_chroma_{i}': np.std(chroma[i]) for i in range(len(chroma))})
    
    # Spectral features
    features['mean_spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    features['std_spectral_centroid'] = np.std(librosa.feature.spectral_centroid(y=signal, sr=sr))
    features['mean_spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    features['std_spectral_bandwidth'] = np.std(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    features['mean_spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
    features['std_spectral_rolloff'] = np.std(librosa.feature.spectral_rolloff(y=signal, sr=sr))
    
    return features

data_path = Path(CONFIG['data_path'])
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
with open('configs/genre_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

#save dataset
df.to_parquet('processed_data/features.parquet')
print(df.head())
print(f"saved dataset: {len(df)} samples, {len(df.columns)} features")
print(f"Skipped files: {len(skipped_files)}")


        