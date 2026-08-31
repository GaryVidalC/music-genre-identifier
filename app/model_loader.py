import json
from pathlib import Path
import pickle

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent

def load_metadata():
    # Load modelmetadata from JSON file
    with open(ROOT_DIR / "model" / "metadata.json", "r") as f:
        model_metadata = json.load(f)

    return model_metadata

def load_encoder():
    # load genre encoder
    with open(ROOT_DIR / "model" / "genre_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return encoder


def load_model(model_used: str):
    # load model
    with open(ROOT_DIR / "model" / f"{model_used.lower()}_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model
