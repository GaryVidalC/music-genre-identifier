import pytest
import io
import numpy as np
import soundfile as sf

from app import main
from fastapi.testclient import TestClient
from fastapi import UploadFile, HTTPException


TEST_METADATA = {
    "model": {
        "model_used" : "xgboost",
        "model_name" : "test_model",
        "version" : "1.0.0",
    }
}

@pytest.fixture
def client(monkeypatch):

    monkeypatch.setattr(
        main, 
        "load_metadata", 
        lambda: TEST_METADATA
    )
    monkeypatch.setattr(main, 
        "load_encoder", 
        lambda: object() 
    )  
    monkeypatch.setattr(
        main, 
        "load_model", 
        lambda model_used: object() 
    )
    monkeypatch.setattr(
        main,
        "feature_extraction", 
        lambda audio_file, metadata: np.zeros((1,54)) 
    )  

    monkeypatch.setattr(
        main, 
        "predict_genre",
        lambda features, model, encoder : "rock",
    )

    with TestClient(main.app) as test_client:
        yield test_client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "model_loaded": True,
        "encoder_loaded": True,
        "metadata_loaded": True
    }

def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == TEST_METADATA


def test_predict_audio_valid_wav(client):
    # Create a dummy .wav file in memory
    sample_rate = 44100
    duration = 30  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Save the audio data to a BytesIO object as a .wav file
    wav_file = io.BytesIO()
    sf.write(wav_file, audio_data, sample_rate, format='WAV')
    wav_file.seek(0)  # Reset the pointer to the beginning of the file

    files = {"file": ("test.wav", wav_file, "audio/wav")}
    response = client.post("/predict-audio", files=files)

    assert response.status_code == 200
    assert "predicted_genre" in response.json()
    assert response.json() == {
        "predicted_genre" : "rock",
        "model_name" : "test_model",
        "model_used" : "xgboost",
        "model_version" : "1.0.0"
    }

def test_predict_audio_empty_file(client):
    # Create an empty .wav file in memory
    empty_wav_file = io.BytesIO(b"")
    empty_wav_file.seek(0)  # Reset the pointer to the beginning of the file

    files = {"file": ("empty.wav", empty_wav_file, "audio/wav")}
    response = client.post("/predict-audio", files=files)

    assert response.status_code == 400

def test_predict_audio_corrupt_file(client):
    # Create a corrupt .wav file in memory
    corrupt_wav_file = io.BytesIO(b"This is not a valid WAV file.")
    corrupt_wav_file.seek(0)  # Reset the pointer to the beginning of the file

    files = {"file": ("corrupt.wav", corrupt_wav_file, "audio/wav")}
    response = client.post("/predict-audio", files=files)

    assert response.status_code == 400

def test_predict_audio_invalid_format(client):
    sample_rate = 44100
    duration = 30  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Save the audio data to a BytesIO object as a .flac file
    flac_file = io.BytesIO()
    sf.write(flac_file, audio_data, sample_rate, format='FLAC')
    flac_file.seek(0)  # Reset the pointer to the beginning of the file

    files = {"file": ("test.flac", flac_file, "audio/flac")}
    response = client.post("/predict-audio", files=files)

    assert response.status_code == 400

def test_predict_shorter_audio(client):
    # Create a dummy .wav file shorter than 29 seconds
    sample_rate = 44100
    duration_short = 28  # seconds
    frequency = 440  # Hz (A4 note)
    t_short = np.linspace(0, duration_short, int(sample_rate * duration_short), endpoint=False)
    audio_data_short = 0.5 * np.sin(2 * np.pi * frequency * t_short)

    wav_file_short = io.BytesIO()
    sf.write(wav_file_short, audio_data_short, sample_rate, format='WAV')
    wav_file_short.seek(0)

    files_short = {"file": ("short.wav", wav_file_short, "audio/wav")}
    response_short = client.post("/predict-audio", files=files_short)
    assert response_short.status_code == 400

def test_predict_longer_audio(client):
    sample_rate = 44100
    frequency = 440  # Hz (A4 note)
    # Create a dummy .wav file longer than 31 seconds
    duration_long = 32  # seconds
    t_long = np.linspace(0, duration_long, int(sample_rate * duration_long), endpoint=False)
    audio_data_long = 0.5 * np.sin(2 * np.pi * frequency * t_long)

    wav_file_long = io.BytesIO()
    sf.write(wav_file_long, audio_data_long, sample_rate, format='WAV')
    wav_file_long.seek(0)

    files_long = {"file": ("long.wav", wav_file_long, "audio/wav")}
    response_long = client.post("/predict-audio", files=files_long)
    assert response_long.status_code == 400 

def test_predict_audio_no_file(client):
    response = client.post("/predict-audio", files={})
    assert response.status_code == 422  # Unprocessable Entity due to missing file

def test_predict_audio_large_file():
    upload = UploadFile(
        file = io.BytesIO(b""),
        filename = "large.wav",
        size = 100 * 1024 * 1024 + 1, 
    )
    with pytest.raises(HTTPException) as error:
        main.validate_wav(upload)

    assert error.value.status_code == 413
    

def test_predict_audio_extraction_error(client, monkeypatch):
    # Create a valid .wav file
    sample_rate = 44100
    duration = 30  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    wav_file = io.BytesIO()
    sf.write(wav_file, audio_data, sample_rate, format='WAV')
    wav_file.seek(0)

    # Patch feature_extraction to raise an exception
    monkeypatch.setattr(main, "feature_extraction", lambda audio_file, metadata: (_ for _ in ()).throw(Exception("Feature extraction error")))

    files = {"file": ("test.wav", wav_file, "audio/wav")}
    response = client.post("/predict-audio", files=files)

    assert response.status_code == 500

def test_predict_audio_prediction_error(client, monkeypatch):
    # Create a valid .wav file
    sample_rate = 44100
    duration = 30  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    wav_file = io.BytesIO()
    sf.write(wav_file, audio_data, sample_rate, format='WAV')
    wav_file.seek(0)

    # Patch predict_genre to raise an exception
    monkeypatch.setattr(main, "predict_genre", lambda features, model, encoder: (_ for _ in ()).throw(Exception("Prediction error")))

    files = {"file": ("test.wav", wav_file, "audio/wav")}
    response = client.post("/predict-audio", files=files)

    assert response.status_code == 500