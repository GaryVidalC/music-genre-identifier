import io
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from backend import inference


def make_wav(duration: float, sample_rate: int = 1000) -> io.BytesIO:
    """Create a silent WAV file in memory."""
    audio = io.BytesIO()
    samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
    sf.write(audio, samples, sample_rate, format="WAV")
    audio.seek(0)
    return audio


def test_predict_short_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extract features once and return model probabilities."""
    audio = make_wav(5)
    metadata = {"preprocessing": {}}
    features = np.ones((1, 54))
    expected = np.array([[0.4, 0.6]])
    extraction = Mock(return_value=features)
    model = Mock()
    model.predict_proba.return_value = expected
    monkeypatch.setattr(inference, "feature_extraction", extraction)

    result = inference.predict_all_audio(model, metadata, audio)

    np.testing.assert_array_equal(result, expected)
    extraction.assert_called_once_with(audio, metadata)
    model.predict_proba.assert_called_once_with(features)


def test_predict_long_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Average chunk probabilities and respect the chunk limit."""
    audio = make_wav(95)
    extraction = Mock(return_value=np.ones((1, 54)))
    model = Mock()
    model.predict_proba.side_effect = [
        np.array([[0.2, 0.8]]),
        np.array([[0.6, 0.4]]),
    ]
    monkeypatch.setattr(inference, "feature_extraction", extraction)

    result = inference.predict_all_audio(
        model,
        {},
        audio,
        n_max_chunks=2,
    )

    np.testing.assert_allclose(result, np.array([[0.4, 0.6]]))
    assert extraction.call_count == 2
    assert all(call.args[0].closed for call in extraction.call_args_list)
