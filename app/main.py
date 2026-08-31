import fastapi
from fastapi import HTTPException
from contextlib import asynccontextmanager
from app.model_loader import load_metadata, load_encoder, load_model
from app.audio_processing import feature_extraction
from app.inference import predict_genre
import soundfile as sf

# Define lifespan for the app
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Perform any startup tasks here (e.g., load model, encoder, etc.)
    print("Starting up the app...")
    app.state.metadata = load_metadata()
    app.state.encoder = load_encoder()
    model_used = app.state.metadata["model"]["model_used"]
    app.state.model = load_model(model_used)
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
    else:        data_loaded["model_loaded"] = False

    if request.app.state.encoder is not None:
        data_loaded["encoder_loaded"] = True
    else:        data_loaded["encoder_loaded"] = False

    if request.app.state.metadata is not None:
        data_loaded["metadata_loaded"] = True
    else:        data_loaded["metadata_loaded"] = False

    return data_loaded

def validate_wav(file: fastapi.UploadFile):

    MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB
    # verify that file is nonempty
    if not file.size:
        raise HTTPException(status_code=400, detail="Empty file uploaded. Please upload a valid .wav file.")

    # verify that file is not too large
    if file.size > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Please upload a .wav file smaller than 100 MB.")

    try:
        file.file.seek(0)  # Ensure the file pointer is at the beginning
        with sf.SoundFile(file.file) as audio_file:
            audio_format = audio_file.format
            sample_rate = audio_file.samplerate
            channels = audio_file.channels
            frames = audio_file.frames
    except sf.SoundFileError as error:
        raise HTTPException(status_code=400, detail="Invalid audio file. Please upload a valid .wav file.") from error

    finally:
        file.file.seek(0)  # Reset the file pointer to the beginning after reading

    if audio_format != 'WAV':
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a .wav file.")

    if sample_rate <= 0 or frames <= 0 or channels <= 0:
        raise HTTPException(status_code=400, detail="Invalid audio file. Please upload a valid .wav file.")

    duration = frames / sample_rate

    if not 29 <= duration <= 31:
        raise HTTPException(status_code=400, detail="Invalid audio duration. Please upload a .wav with aprox 30 seconds.")

    return {"duration": duration, "sample_rate": sample_rate, "channels": channels, "frames": frames}



@app.post("/predict-audio")
def predict_audio(request: fastapi.Request, file: fastapi.UploadFile = fastapi.File(...)):
    # verify that file is valid

    validate_wav(file)

    # Extract features from the audio file
    try:
        file.file.seek(0)  # Ensure the file pointer is at the beginning
        features = feature_extraction(file.file, request.app.state.metadata)
    except Exception:
        raise HTTPException(status_code=500, detail="Error extracting features from audio file.")

    # Predict genre using the model and encoder
    try:
        predicted_genre = predict_genre(features, request.app.state.model, request.app.state.encoder)
    except Exception:
        raise HTTPException(status_code=500, detail="Error predicting genre from features.")

    result = {"predicted_genre": predicted_genre,
            "model_name": request.app.state.metadata["model"]["model_name"],
            "model_used": request.app.state.metadata["model"]["model_used"],
            "model_version": request.app.state.metadata["model"]["version"],
            }

    return result
