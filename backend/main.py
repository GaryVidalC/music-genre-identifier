import fastapi
import numpy as np
from fastapi import HTTPException
from contextlib import asynccontextmanager
from backend.model_loader import load_model_resources
from backend.inference import predict_all_audio
import soundfile as sf

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


def validate_wav(file: fastapi.UploadFile):

    MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB
    # verify that file is nonempty
    if not file.size:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded. Please upload a valid .wav file.",
        )

    # verify that file is not too large
    if file.size > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. Please upload a .wav file smaller than "
                "100 MB."
            ),
        )

    try:
        file.file.seek(0)  # Ensure the file pointer is at the beginning
        with sf.SoundFile(file.file) as audio_file:
            audio_format = audio_file.format
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
        file.file.seek(0)

    if audio_format != 'WAV':
        raise HTTPException(
            status_code=400,
            detail="Invalid audio format. Please upload a .wav file.",
        )

    if sample_rate <= 0 or frames <= 0 or channels <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file. Please upload a valid .wav file.",
        )


@app.post("/predict-audio")
def predict_audio(
    request: fastapi.Request,
    file: fastapi.UploadFile = fastapi.File(...),
):
    # verify that file is valid

    validate_wav(file)

    probabilities = predict_all_audio(
        model=request.app.state.model,
        metadata=request.app.state.metadata,
        audio=file.file,
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
