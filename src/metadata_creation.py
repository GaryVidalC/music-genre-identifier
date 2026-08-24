import pandas as pd
from pathlib import Path
import pickle
import json
import datetime
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# data metadata
processed_data_path = PROJECT_ROOT / 'processed_data'

def load_genre_encoder(encoder_path: Path):
    """Load genre encoder (LabelEncoder) from pickle file."""
    with encoder_path.open("rb") as file:
        encoder = pickle.load(file)
    return encoder

data = pd.read_parquet(processed_data_path/'features.parquet')

number_of_samples = len(data)
number_of_features = data.shape[1] - 1

# load enconder
encoder = load_genre_encoder(PROJECT_ROOT / 'model/genre_encoder.pkl')
genre_classes = encoder.classes_

data_info = {
    "number_of_samples": number_of_samples,
    "number_of_features": number_of_features,
    "genre_classes": genre_classes.tolist()
}

# Preprocessing metadata
preprocessing_metadata_path = PROJECT_ROOT / 'configs'

with open(preprocessing_metadata_path/ 'read_config.json', 'r', encoding='utf-8') as f:
    read_config = json.load(f)

preprocessing_data = read_config

preprocessing_data['features'] = data.columns[:-1].tolist()

# model metadata

model_used = "XGBoost"
model_used_lower = model_used.lower()
source_model_path = PROJECT_ROOT / f"model/model_selection/{model_used_lower}_model.pkl"
target_model_path = PROJECT_ROOT / f"model/{model_used_lower}_model.pkl"

if source_model_path.exists():
    shutil.copy2(source_model_path, target_model_path)
else:
    raise FileNotFoundError(f"Model file not found: {source_model_path}")

model_metadata = {
    "model_name": "Metal",
    "model_used": model_used,
    "version": "1.0",
    "training_date": datetime.date.today().isoformat(),
    'model_pkl_path': f"model/{model_used_lower}_model.pkl",
}

# model stats
stats_path = PROJECT_ROOT / 'model/model_selection'
stats = pd.read_csv(stats_path / "model_results.csv")

model_stats_df = stats[stats['Model'] == model_used]
model_stats = {}

for col in model_stats_df.columns:
    model_stats[col] = model_stats_df.iloc[0][col]

# Combine all metadata into a single dictionary

metadata = {
    "dataset": "GTZAN",
    "data_info": data_info,
    "preprocessing": preprocessing_data,
    "model": model_metadata,
    "model_stats": model_stats
}

# Save metadata to a JSON file
metadata_output_path = PROJECT_ROOT / 'model'

with open(metadata_output_path / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)

# copy model pkl file to model directory
