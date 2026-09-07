import numpy as np
from app.audio_processing import standardize_signal

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