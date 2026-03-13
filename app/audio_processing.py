import numpy as np
import librosa


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

def feature_extraction(audio_file, metadata):
    # get metadata info
    preprocessing = metadata["preprocessing"]

    sample_rate = preprocessing["sample_rate"]
    mono = preprocessing["mono"]
    duration = preprocessing["duration"]
    normalization = preprocessing["normalization"]
    n_mfcc = preprocessing["n_mfcc"]

    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Standardize the signal
    y = standardize_signal(y, sample_rate, duration, normalization)

    # Extract features
    features = extract_audio_features(y, sr, n_mfcc)
    
    # Standardize
    feature_values = [features[feature] for feature in metadata['preprocessing']['features']]
    feature_array = np.array(feature_values).reshape(1, -1)

    return feature_array