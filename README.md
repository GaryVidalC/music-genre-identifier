# Problem

Trying to implement a music identifier with fast-api, then adding a recomendation system. I only have the first block "working". Need to improve and generalize it a bit first.

## Dataset and Model

Used the [GZTAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) for training. 

- Librosa: for processing the audiofiles.
- XGBoost: for prediction (had an F1-score of 0.66 aprox).

I evaluated SVM and RandomForest, but the best performing model at the moment was XGBoost.

# First block: Music Genre Identifier API 
## Model "Metal-v1"

API in FastAPI to detect the genre of a .wav file.

**IMPORTANT**: It only works with .wav files that are 30 seconds long.

## Endpoints

- `GET /health`: Verifies the status of the API
- `GET /ready`: Verifies that data and model were loaded.
- `GET /model-info`: Gets the metadata of the current model.
- `POST /predict-audio`: inputs a .wav and outputs a genre.

## To start:

```bash
uvicorn app.main:app --reload
```

Full Documentation:

- `http://127.0.0.1:8000/docs`

## Structure

- `app/main.py`: endpoint and `lifespan`.
- `app/model_loader.py`: loads metadata, encoder and model.
- `app/audio_processing.py`: extracts audio features.
- `app/inference.py`: predicts genre.
- `model/`: model's (`.pkl`, `metadata.json`).

# To do:

- Add other audio filetypes (.mp3, .FLAC, etc)
- Implement inference model for files with >30 seconds
