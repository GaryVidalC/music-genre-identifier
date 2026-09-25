# Music Genre Classification API

This project classifies 30-second WAV clips into one of ten music genres. It covers the full path from audio preprocessing and feature extraction to model comparison and inference through a FastAPI service.

The supported genres are blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

Use Docker to start the API.

## Results

Three classifiers were trained and evaluated on the same split. XGBoost produced the best test macro F1 and is the model served by the API.

| Model | Test macro F1 | Test accuracy | Inference per sample |
|---|---:|---:|---:|
| SVM | 0.6520 | 0.6533 | 0.0534 ms |
| Random Forest | 0.6500 | 0.6533 | 0.1391 ms |
| XGBoost | **0.6645** | **0.6667** | **0.0449 ms** |

## Data and preprocessing

The project uses the [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The processed dataset contains 999 usable clips across the ten genres.

Each clip is converted to mono at 22,050 Hz, normalized, and standardized to 30 seconds. The model uses 54 summary features extracted with librosa:

- Mean and standard deviation of 12 MFCC coefficients: 24 features
- Mean and standard deviation of 12 chroma bins: 24 features
- Mean and standard deviation of spectral centroid, bandwidth, and rolloff: 6 features

SVM, Random Forest, and XGBoost were tuned with 5-fold cross-validation. The final comparison uses a held-out test split.

## How inference works

```text
WAV upload
    -> format, size, and duration validation
    -> mono conversion and resampling to 22,050 Hz
    -> audio feature extraction
    -> XGBoost prediction
    -> genre label
```

The upload must be a valid WAV file between 29 and 31 seconds long and no larger than 100 MiB. Audio with a different sample rate is resampled before feature extraction.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Confirms that the API process is running |
| `GET` | `/ready` | Reports whether the model, encoder, and metadata are loaded |
| `GET` | `/model-info` | Returns model and preprocessing metadata |
| `POST` | `/predict-audio` | Accepts a WAV upload and returns the predicted genre |

FastAPI also exposes interactive documentation at `http://localhost:8080/docs`.

## Running the API with Docker

The Docker image contains the inference API and model artifacts. Training code and data remain outside the runtime image.

### Building the image

To build the image, run from the root of the repo:

```bash
docker build -t music-ai .
```

### Running the container

To run it, use the following:

```bash
docker run --rm -p 8080:8080 music-ai:latest
```

```bash
curl -X POST http://localhost:8080/predict-audio \
  -F "file=@path/to/audio.wav"
```

Example response:

```json
{
  "predicted_genre": "hiphop",
  "model_name": "Metal",
  "model_used": "XGBoost",
  "model_version": "1.0"
}
```

## Tests

Install the development dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Training dependencies

The API environment does not include the data and plotting packages used during training. Install the extended dependency set before running the scripts in `src/`:

```bash
pip install -r requirements-training.txt
```

## Project structure

```text
app/             FastAPI service and inference code
configs/         Preprocessing and model-search configuration
model/           Trained model, label encoder, and metadata
src/             Data processing, training, and metadata scripts
processed_data/  Extracted features and dataset splits
tests/           Unit and API tests
Dockerfile       Container image definition for the inference API
.dockerignore    Files excluded from the Docker build context
```

## Limitations

The classifier was trained on GTZAN and inherits the limitations of that dataset. It predicts one genre per clip, does not return calibrated confidence scores, and currently accepts only WAV input close to 30 seconds long.
