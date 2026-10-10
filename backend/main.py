import fastapi
import numpy as np
import logging
from fastapi import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import asynccontextmanager
from backend.model_loader import load_model_resources
from backend.inference import predict_all_audio
from backend.audio_processing import youtube_downloader
from urllib.parse import urlparse
from threading import BoundedSemaphore
from typing import BinaryIO
from yt_dlp.utils import DownloadError
import soundfile as sf


MAX_AUDIO_SIZE = 100 * 1024 * 1024
prediction_slot = BoundedSemaphore(value=1)
logger = logging.getLogger(__name__)

# Define lifespan for the app


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Perform any startup tasks here (e.g., load model, encoder, etc.)
    print("Starting up the app...")
    (
        app.state.model,
        app.state.encoder,
        app.state.metadata,
    ) = load_model_resources()
    yield
    # Perform any shutdown tasks here (if needed)
    print("Shutting down the app...")
    del app.state.metadata
    del app.state.encoder
    del app.state.model

# Create app
app = fastapi.FastAPI(lifespan=lifespan)


@app.get("/health")
def read_health():
    return {"status": "ok"}


@app.get("/model-info")
def read_model_info(request: fastapi.Request):
    return request.app.state.metadata


@app.get("/ready")
def read_ready(request: fastapi.Request):
    data_loaded = {}

    if request.app.state.model is not None:
        data_loaded["model_loaded"] = True
    else:
        data_loaded["model_loaded"] = False

    if request.app.state.encoder is not None:
        data_loaded["encoder_loaded"] = True
    else:
        data_loaded["encoder_loaded"] = False

    if request.app.state.metadata is not None:
        data_loaded["metadata_loaded"] = True
    else:
        data_loaded["metadata_loaded"] = False

    return data_loaded


def validate_wav(audio: BinaryIO, size: int | None) -> None:
    """Validate audio and size; raise HTTP errors for invalid input."""
    # verify that file is nonempty
    if not size:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded. Please upload a valid .wav file.",
        )

    # verify that file is not too large
    if size > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. Please upload a .wav file smaller than "
                "100 MB."
            ),
        )

    try:
        audio.seek(0)  # Ensure the file pointer is at the beginning
        with sf.SoundFile(audio) as audio_file:
            sample_rate = audio_file.samplerate
            channels = audio_file.channels
            frames = audio_file.frames
    except sf.SoundFileError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file. Please upload a valid .wav file.",
        ) from error

    finally:
        # Reset the file pointer to the beginning after reading
        audio.seek(0)

    if sample_rate <= 0 or frames <= 0 or channels <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file. Please upload a non empty .wav file.",
        )


def predict_file(
    request: fastapi.Request,
    audio: BinaryIO,
    size: int | None,
) -> dict:
    """Validate audio and size; return prediction using request's resources."""

    validate_wav(audio, size)

    probabilities = predict_all_audio(
        model=request.app.state.model,
        metadata=request.app.state.metadata,
        audio=audio,
        n_max_chunks=10,
    )

    # predicted genre with encoder
    prediction_index = np.argmax(probabilities, axis=1)
    encoded_prediction = request.app.state.model.classes_[
        prediction_index
    ]

    # Get the ordered list of genres
    encoded_classes = request.app.state.model.classes_
    ordered_genres = request.app.state.encoder.inverse_transform(
        encoded_classes
    )
    predicted_genre = request.app.state.encoder.inverse_transform(
        encoded_prediction
    )

    # store the probabilities in a dictionary with genre names as keys
    probabilities_dict = {
        genre: prob for genre, prob in zip(ordered_genres, probabilities[0])
    }

    return {
        "predicted_genre": predicted_genre[0],
        "probabilities": probabilities_dict,
    }


@app.post("/predict-audio")
def predict_audio(
    request: fastapi.Request,
    file: fastapi.UploadFile = fastapi.File(...),
) -> dict:
    """Predict the uploaded file's genre using the request's model."""
    if not prediction_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                "Another prediction is processing. Please try again shortly."
            ),
        )
    try:
        return predict_file(request, file.file, file.size)
    finally:
        prediction_slot.release()


@app.post("/predict-youtube")
def predict_youtube(
    request: fastapi.Request,
    url: str = fastapi.Query(...),
) -> dict:
    """Download url in isolation and return request's model prediction."""

    # check if URL is from youtube
    parsed_url = urlparse(url)
    allowed_domains = {
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
        "m.youtube.com"
    }
    if (
        parsed_url.scheme not in {"http", "https"}
        or parsed_url.hostname not in allowed_domains
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Please provide a valid YouTube URL.",
        )

    # reject playlist URLs
    if parsed_url.path.rstrip('/') == '/playlist':
        raise HTTPException(
            status_code=400,
            detail=(
                "Playlist URLs are not supported. "
                "Please provide a single video URL."
            ),
        )
    # check if another prediction is processing
    if not prediction_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                "Another prediction is processing. Please try again shortly."
            ),
        )

    # Download the audio from the YouTube URL and predict its genre
    try:
        with TemporaryDirectory(prefix="music-genre-") as temp_dir:
            try:
                audio_path = Path(
                    youtube_downloader(url, temp_dir, MAX_AUDIO_SIZE)
                )
            except DownloadError as e:
                logger.exception("YouTube download failed")
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Unable to download the video. It may be unavailable."
                    ),
                ) from e
            with audio_path.open("rb") as audio:
                return predict_file(request, audio, audio_path.stat().st_size)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("YouTube prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the audio. Please try again later.",
        ) from e
    finally:
        prediction_slot.release()
