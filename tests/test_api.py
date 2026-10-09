import io
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

from backend import main


TEST_METADATA = {
    "model": {
        "model_used": "svc",
        "model_name": "test_model",
        "version": "1",
    }
}
TEST_PROBABILITIES = np.array([[0.25, 0.75]])


def make_wav(
    duration: float = 1,
    sample_rate: int = 8000,
    audio_format: str = "WAV",
) -> io.BytesIO:
    """Create an audio file in memory."""
    audio = io.BytesIO()
    samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
    sf.write(audio, samples, sample_rate, format=audio_format)
    audio.seek(0)
    return audio


@pytest.fixture
def prediction_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Return deterministic probabilities from the inference workflow."""
    prediction = Mock(return_value=TEST_PROBABILITIES)
    monkeypatch.setattr(main, "predict_all_audio", prediction)
    return prediction


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    prediction_mock: Mock,
) -> Iterator[TestClient]:
    """Create an API client with mocked model resources."""
    model = Mock()
    model.classes_ = np.array([0, 1])

    encoder = LabelEncoder()
    encoder.classes_ = np.array(["country", "rock"])

    loader = Mock(return_value=(model, encoder, TEST_METADATA))
    monkeypatch.setattr(main, "load_model_resources", loader)

    with TestClient(main.app) as test_client:
        yield test_client


def test_status_endpoints(client: TestClient) -> None:
    """Expose health, readiness, and model metadata."""
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {
        "model_loaded": True,
        "encoder_loaded": True,
        "metadata_loaded": True,
    }
    assert client.get("/model-info").json() == TEST_METADATA


def test_predict_audio(
    client: TestClient,
    prediction_mock: Mock,
) -> None:
    """Return the winning genre and its class-aligned probabilities."""
    files = {"file": ("test.wav", make_wav(), "audio/wav")}

    response = client.post("/predict-audio", files=files)

    assert response.status_code == 200
    assert response.json() == {
        "predicted_genre": "rock",
        "probabilities": {
            "country": 0.25,
            "rock": 0.75,
        },
    }
    prediction_mock.assert_called_once()


def test_predict_youtube_cleans_temporary_files(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use client and monkeypatch to verify isolation and error cleanup."""
    directories = []
    opened_files = []

    for failure, status in (
        ("none", 200),
        ("download", 500),
        ("http", 413),
        ("prediction", 500),
    ):
        def download(url: str, output_dir: str) -> str:
            """Write url's files in output_dir; return the MP3 or fail."""
            directory = Path(output_dir)
            directories.append(directory)
            (directory / "track.part").write_bytes(b"partial download")
            if failure == "download":
                raise RuntimeError("Download failed")
            audio_path = directory / "track.mp3"
            audio_path.write_bytes(b"audio")
            return str(audio_path)

        def predict(request: main.fastapi.Request, file: UploadFile) -> dict:
            """Read file; return a fake prediction or the selected error."""
            opened_files.append(file.file)
            assert file.file.read() == b"audio"
            assert file.size == 5
            if failure == "http":
                raise HTTPException(413, "Audio too large")
            if failure == "prediction":
                raise RuntimeError("Prediction failed")
            return {"predicted_genre": "rock", "probabilities": {"rock": 1.0}}

        monkeypatch.setattr(main, "youtube_downloader", download)
        monkeypatch.setattr(main, "predict_audio", predict)
        response = client.post("/predict-youtube", params={"url": "mock-url"})
        assert response.status_code == status
        if failure == "none":
            assert response.json()["predicted_genre"] == "rock"
        elif failure == "http":
            assert response.json()["detail"] == "Audio too large"
        assert not directories[-1].exists()
        assert all(file.closed for file in opened_files)

    assert len(set(directories)) == len(directories)


def test_predict_audio_rejects_invalid_uploads(client: TestClient) -> None:
    """Reject corrupt audio, non-WAV formats, and missing files."""
    corrupt = {"file": (
        "corrupt.wav",
        io.BytesIO(b"not a WAV file"),
        "audio/wav",
    )}
    flac = {"file": (
        "test.flac",
        make_wav(audio_format="FLAC"),
        "audio/flac",
    )}

    assert client.post("/predict-audio", files=corrupt).status_code == 400
    assert client.post("/predict-audio", files=flac).status_code == 400
    assert client.post("/predict-audio", files={}).status_code == 422


def test_predict_audio_rejects_large_file() -> None:
    """Reject files larger than the upload limit."""
    upload = UploadFile(
        file=io.BytesIO(),
        filename="large.wav",
        size=100 * 1024 * 1024 + 1,
    )

    with pytest.raises(HTTPException) as error:
        main.validate_wav(upload)

    assert error.value.status_code == 413
