import numpy as np
import math
import soundfile as sf
from io import BytesIO
from app.audio_processing import standardize_signal, extract_audio_features, feature_extraction

def test_standardize_signal():
    signal = np.ones(5)

    result = standardize_signal(
        signal = signal,
        sample_rate = 10,
        duration = 1,
        normalize = False
    )

    assert len(result) == 10
    assert np.allclose(result[:5], signal)
    assert np.allclose(result[5:], 0)

def test_standardize_signal_normalizes_audio():
    signal = np.array([1, 2, 3])

    result = standardize_signal(
        signal = signal,
        sample_rate = 10,
        duration = 1,
        normalize = True
    )

    assert np.max(np.abs(result)) == 1

def test_exact_length_signal():
    signal = np.array([1, 2, 3, 4, 5])

    result = standardize_signal(
        signal = signal,
        sample_rate = 5,
        duration = 1,
        normalize = False
    )

    assert len(result) == 5
    assert np.allclose(result, signal)

def test_non_normalized_signal():
    signal = np.array([1, 2, 3])

    result = standardize_signal(
        signal = signal,
        sample_rate = 10,
        duration = 1,
        normalize = False
    )

    assert np.max(np.abs(result)) == 3

def test_silent_signal():
    signal = np.zeros(5)

    result = standardize_signal(
        signal = signal,
        sample_rate = 10,
        duration = 1,
        normalize = True
    )

    assert np.allclose(result, 0)
    assert np.isnan(result).sum() == 0  # Ensure no NaN values are present
    
def test_quantity_and_names_audio_features():
    sr = 22050
    time = np.arange(sr)/sr
    signal = 0.5*np.sin(2*np.pi*440*time)  

    n_mfcc = 12

    features = extract_audio_features(signal,sr=sr, n_mfcc=n_mfcc)

    assert isinstance(features, dict)
    assert len(features) == 54
    for i in range(n_mfcc):
        assert f'mean_mfcc_{i}' in features
        assert f"std_mfcc_{i}" in features
    for i in range(12):  # chroma only has 12 features, regardless of n_mfcc
        assert f"mean_chroma_{i}" in features
        assert f"std_chroma_{i}" in features
    
    assert 'mean_spectral_centroid' in features
    assert 'std_spectral_centroid' in features
    assert 'mean_spectral_bandwidth' in features
    assert 'std_spectral_bandwidth' in features
    assert 'mean_spectral_rolloff' in features
    assert 'std_spectral_rolloff' in features

def test_audio_features_values():
    sr = 22050
    time = np.arange(sr)/sr
    signal = 0.5*np.sin(2*np.pi*440*time)  

    n_mfcc = 12
    features = extract_audio_features(signal,sr=sr, n_mfcc=n_mfcc)

    assert all(np.isscalar(value) for value in features.values())
    assert np.all(np.isfinite(list(features.values())))

def test_audio_features_with_silent_signal():
    sr = 22050
    signal = np.zeros(sr)  # Silent signal

    n_mfcc = 12
    features = extract_audio_features(signal, sr=sr, n_mfcc=n_mfcc)

    assert isinstance(features, dict)
    assert len(features) == 54
    for i in range(n_mfcc):
        assert f'mean_mfcc_{i}' in features and math.isfinite(features[f'mean_mfcc_{i}'])
        assert f"std_mfcc_{i}" in features and math.isfinite(features[f'std_mfcc_{i}'])
    for i in range(12):  # Chroma always has 12 features, regardless of n_mfcc
        assert f"mean_chroma_{i}" in features and math.isfinite(features[f'mean_chroma_{i}'])
        assert f"std_chroma_{i}" in features and math.isfinite(features[f'std_chroma_{i}'])
    
    assert 'mean_spectral_centroid' in features and math.isfinite(features['mean_spectral_centroid'])
    assert 'std_spectral_centroid' in features and math.isfinite(features['std_spectral_centroid'])
    assert 'mean_spectral_bandwidth' in features and math.isfinite(features['mean_spectral_bandwidth'])
    assert 'std_spectral_bandwidth' in features and math.isfinite(features['std_spectral_bandwidth'])
    assert 'mean_spectral_rolloff' in features and math.isfinite(features['mean_spectral_rolloff'])
    assert 'std_spectral_rolloff' in features and math.isfinite(features['std_spectral_rolloff'])

def test_audio_features_with_other_n_mfcc_values():
    sr = 22050
    time = np.arange(sr)/sr
    signal = 0.5*np.sin(2*np.pi*440*time)  

    for n_mfcc in [5, 20]:
        features = extract_audio_features(signal, sr=sr, n_mfcc=n_mfcc)

        assert isinstance(features, dict)
        assert len(features) == 2*n_mfcc + 24 + 6
        for i in range(n_mfcc):
            assert f'mean_mfcc_{i}' in features
            assert f"std_mfcc_{i}" in features
        for i in range(12):
            assert f"mean_chroma_{i}" in features
            assert f"std_chroma_{i}" in features
        
        assert 'mean_spectral_centroid' in features
        assert 'std_spectral_centroid' in features
        assert 'mean_spectral_bandwidth' in features
        assert 'std_spectral_bandwidth' in features
        assert 'mean_spectral_rolloff' in features
        assert 'std_spectral_rolloff' in features

def test_feature_extraction():
    # valid .wave file generated
    sr = 22050
    duration = 1

    time = np.arange(sr*duration)/sr
    signal = 0.5*np.sin(2*np.pi*440*time)  #

    wav = BytesIO()

    sf.write(wav, signal, sr, format='WAV')
    wav.seek(0)

    metadata = {
        "preprocessing": {
            "sample_rate": sr,
            "mono": True,
            "duration": duration,
            "normalization": True,
            "n_mfcc": 12,
            "features": [
                f"mean_mfcc_{i}" for i in range(12)
            ] + [
                f"std_mfcc_{i}" for i in range(12)
            ] + [
                f"mean_chroma_{i}" for i in range(12)
            ] + [
                f"std_chroma_{i}" for i in range(12)
            ] + [
                'mean_spectral_centroid',
                'std_spectral_centroid',
                'mean_spectral_bandwidth',
                'std_spectral_bandwidth',
                'mean_spectral_rolloff',
                'std_spectral_rolloff'
            ]
        }
    }   
    results = feature_extraction(wav, metadata)

    assert isinstance(results, np.ndarray)
    assert results.shape == (1, 54)  # 1 sample, 54 features
    assert np.all(np.isfinite(results))  # Ensure all features are finite

def test_resampling_and_mono_conversion(monkeypatch):
    # Create a stereo signal with a different sample rate
    sr_original = 44100
    duration = 1
    time = np.arange(sr_original*duration)/sr_original
    signal_stereo = np.array([0.5*np.sin(2*np.pi*440*time), 0.5*np.sin(2*np.pi*440*time)]).T  # Stereo signal

    wav = BytesIO()
    sf.write(wav, signal_stereo, sr_original, format='WAV')
    wav.seek(0)

    metadata = {
        "preprocessing": {
            "sample_rate": 22050,  # Target sample rate
            "mono": True,           # Convert to mono
            "duration": duration,
            "normalization": True,
            "n_mfcc": 12,
            "features": [
                f"mean_mfcc_{i}" for i in range(12)
            ] + [
                f"std_mfcc_{i}" for i in range(12)
            ] + [
                f"mean_chroma_{i}" for i in range(12)
            ] + [
                f"std_chroma_{i}" for i in range(12)
            ] + [
                'mean_spectral_centroid',
                'std_spectral_centroid',
                'mean_spectral_bandwidth',
                'std_spectral_bandwidth',
                'mean_spectral_rolloff',
                'std_spectral_rolloff'
            ]
        }
    }   
    received_audio = {}

    def capture_audio(signal, sr, n_mfcc):
        received_audio["signal"] = signal
        received_audio["sample_rate"] = sr
        received_audio["n_mfcc"] = n_mfcc
        return {feature: 0.0 for feature in metadata["preprocessing"]["features"]}

    monkeypatch.setattr("app.audio_processing.extract_audio_features", capture_audio)

    results = feature_extraction(wav, metadata)

    assert isinstance(results, np.ndarray)
    assert results.shape == (1, 54)  # 1 sample, 54 features
    assert np.all(np.isfinite(results))  # Ensure all features are finite
    assert received_audio["sample_rate"] == 22050
    assert received_audio["signal"].ndim == 1
    assert len(received_audio["signal"]) == 22050
    assert received_audio["n_mfcc"] == 12

def test_feature_extraction_feature_order(monkeypatch):
    # Create a simple sine wave signal
    sr = 22050
    duration = 1
    time = np.arange(sr*duration)/sr
    signal = 0.5*np.sin(2*np.pi*440*time)  # A4 note

    wav = BytesIO()
    sf.write(wav, signal, sr, format='WAV')
    wav.seek(0)

    metadata = {
        "preprocessing": {
            "sample_rate": sr,
            "mono": True,
            "duration": duration,
            "normalization": True,
            "n_mfcc": 12,
            "features": ["feature_c", "feature_a", "feature_b"]
        }
    }

    extracted_features = {
        "feature_a": 10.0,
        "feature_b": 20.0,
        "feature_c": 30.0,
    }

    def return_known_features(signal, sr, n_mfcc):
        return extracted_features

    monkeypatch.setattr(
        "app.audio_processing.extract_audio_features",
        return_known_features,
    )

    results = feature_extraction(wav, metadata)

    assert results.shape == (1, 3)
    assert np.array_equal(results, np.array([[30.0, 10.0, 20.0]]))
