import librosa
import numpy as np


def standardize_signal(
    signal: np.ndarray,
    sample_rate: int,
    duration: int,
    normalize: bool,
) -> np.ndarray:
    """Pad an audio signal and optionally normalize it.

    Args:
        signal: Audio samples to process.
        sample_rate: Number of samples per second.
        duration: Target duration in seconds.
        normalize: Whether to normalize the signal amplitude.

    Returns:
        The standardized audio signal.
    """
    objective_samples = sample_rate * duration

    if len(signal) < objective_samples:
        signal = librosa.util.fix_length(signal, size=objective_samples)

    if normalize:
        signal = librosa.util.normalize(signal)

    return signal


def extract_audio_features(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int,
) -> dict[str, float]:
    """Extract audio features used by the classifier.

    Args:
        signal: Audio samples from which to extract features.
        sr: Audio sample rate.
        n_mfcc: Number of MFCC coefficients to calculate.

    Returns:
        Extracted feature names and values.
    """

    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    features.update({f'mean_mfcc_{i}': np.mean(
        mfcc[i]) for i in range(len(mfcc))})
    features.update({f'std_mfcc_{i}': np.std(
        mfcc[i]) for i in range(len(mfcc))})

    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.update({f'mean_chroma_{i}': np.mean(
        chroma[i]) for i in range(len(chroma))})
    features.update({f'std_chroma_{i}': np.std(
        chroma[i]) for i in range(len(chroma))})

    # Spectral features
    features['mean_spectral_centroid'] = np.mean(
        librosa.feature.spectral_centroid(y=signal, sr=sr))
    features['std_spectral_centroid'] = np.std(
        librosa.feature.spectral_centroid(y=signal, sr=sr))
    features['mean_spectral_bandwidth'] = np.mean(
        librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    features['std_spectral_bandwidth'] = np.std(
        librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    features['mean_spectral_rolloff'] = np.mean(
        librosa.feature.spectral_rolloff(y=signal, sr=sr))
    features['std_spectral_rolloff'] = np.std(
        librosa.feature.spectral_rolloff(y=signal, sr=sr))

    return features
