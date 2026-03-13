import fastapi
import json
from pathlib import Path
from contextlib import asynccontextmanager
from app.model_loader import load_metadata, load_encoder, load_model
from app.audio_processing import feature_extraction
from app.inference import predict_genre

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

# Root directory of the project
# APP_DIR = Path(__file__).parent
# ROOT_DIR = APP_DIR.parent

# @app.get("/")
# def read_root(): 
#     return {"Hello": "World"}

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

@app.post("/predict-audio")
def predict_audio(request: fastapi.Request, file: fastapi.UploadFile = fastapi.File(...)):
    # verify that file is nonempty
    if file.filename == "":
        return {"error": "No file uploaded."}
    
    # verify that file is .wav
    if not file.filename.endswith(".wav"):
        return {"error": "Invalid file format. Please upload a .wav file."}
    

    # Save the uploaded file to a temporary location
    temp_file_path = Path("temp_audio_file")
    
    with open(temp_file_path, "wb") as f:
        f.write(file.file.read())

    # Extract features from the audio file
    features = feature_extraction(temp_file_path, request.app.state.metadata)

    # Predict genre using the model and encoder
    predicted_genre = predict_genre(features, request.app.state.model, request.app.state.encoder)

    # Remove the temporary file
    temp_file_path.unlink()

    result = {"predicted_genre": predicted_genre,
              "model_name": request.app.state.metadata["model"]["model_name"],
              "model_used": request.app.state.metadata["model"]["model_used"],
              "model_version": request.app.state.metadata["model"]["version"],
              }

    return result