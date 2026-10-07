# Music Genre Classification API

This project classifies WAV clips into one of ten music genres. It covers the full path from audio preprocessing and feature extraction to model comparison and inference through a FastAPI service.

The supported genres are blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

Use Docker Compose to start MLflow and the API.

## Architecture

Docker Compose runs MLflow and FastAPI, with an optional `cloudflared` service
for public access. The live API runs on a Debian homeserver and is exposed
through Cloudflare Tunnel at `https://api.gvidal.cl`.

FastAPI loads
`music-genre-classifier@champion` from MLflow during startup and keeps the
model in memory for inference. MLflow persists runs, registry data, and model
artifacts in `data/mlflow/`. MLflow is not exposed through the public tunnel.

The API code, production dependencies, and Docker build files live in
`backend/`. Compose builds the API using that directory as its context.
Training remains separate in `src/`.

```text
Client -> Cloudflare -> cloudflared -> FastAPI -> model in memory
                                         |
                                         └── MLflow during startup
                                                |
                                                └── data/mlflow/
```

## Data and preprocessing

The project uses the [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The processed dataset contains 999 usable clips across the ten genres.

Each clip is converted to mono at 22,050 Hz, normalized, and standardized to 30 seconds. The model uses 54 summary features extracted with librosa:

- Mean and standard deviation of 12 MFCC coefficients: 24 features
- Mean and standard deviation of 12 chroma bins: 24 features
- Mean and standard deviation of spectral centroid, bandwidth, and rolloff: 6 features

SVC, Random Forest, and XGBoost were hyperparameter-tuned with Optuna and tracked with MLflow using a fixed random seed and 30 trials per model.

## Results

Three classifiers were trained and evaluated on the same validation split.

| Model | F1 score | Precision | Recall | Inference time (seconds) |
|---|---:|---:|---:|---:|
| SVC | **0.73** | **0.75** | **0.73** | **0.008** |
| Random Forest | 0.69 | 0.71 | 0.69 | 0.011 |
| XGBoost | 0.68 | 0.70 | 0.69 | 0.005 |

The script automatically selects the champion model SVC with parameters:
- C = 2.223
- gamma = scale

## How inference works

```text
WAV upload
    -> format and size validation
    -> mono conversion and resampling to 22,050 Hz
    -> audio feature extraction
    -> champion model prediction
    -> genre label and probabilities for every class
```

The model processes audio in 30-second segments. For longer audio, it averages the class probabilities across up to ten segments.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Confirms that the API process is running |
| `GET` | `/ready` | Reports whether the model, encoder, and metadata are loaded |
| `GET` | `/model-info` | Returns model and preprocessing metadata |
| `POST` | `/predict-audio` | Accepts a WAV upload and returns the predicted genre and probabilities by class |

FastAPI also exposes interactive documentation at `http://localhost:8080/docs`.

### Live API

- [Interactive docs](https://api.gvidal.cl/docs)
- [Model information](https://api.gvidal.cl/model-info)
- Base URL: `https://api.gvidal.cl`

In the interactive docs, open `POST /predict-audio`, select **Try it out**,
upload a WAV file, and select **Execute**. You can also send a request with:

```bash
curl -F "file=@clip.wav" https://api.gvidal.cl/predict-audio
```

The JSON response contains `predicted_genre`, the class with the highest
probability, and `probabilities`, a mapping from each genre to a value between
0 and 1.

### Previous Cloud Run deployment (outdated)

The previous deployment used Google Cloud Run. These links are retained as
historical references; the current API uses the homeserver deployment above.

- [Base URL](https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app)
- [Interactive docs](https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app/docs)

## Running locally with Docker Compose

MLflow stores experiment data and model artifacts in the `data/mlflow/` directory. This directory is persistent but is not committed to Git.

### First Setup

Install the extended dependency set before running the training scripts in `src/`:

```bash
pip install -r requirements-training.txt
```

Start the MLflow tracking server for the models and artifacts.

```bash
docker compose up -d mlflow
```
Then train and register the models. The script selects the *champion* using the highest F1 score.

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python src/model_selection.py
```

Build and start the FastAPI service.

```bash
docker compose up -d --build api
```

The services are available by default at:

- MLflow: http://localhost:5000
- FastAPI: http://localhost:8080
- FastAPI documentation: http://localhost:8080/docs

After the first setup, you can start the services with:

```bash
docker compose up -d
```

## Public access with Cloudflare Tunnel

Once the API is running, you can create a remotely managed tunnel in Cloudflare and
store its token in a `.env` file at the repository root on the server:

```dotenv
CLOUDFLARE_TUNNEL_TOKEN=your_tunnel_token
```

The `.env` file is ignored by Git. Compose passes this value to `cloudflared`
as `TUNNEL_TOKEN`.

Start the optional connector:

```bash
docker compose --profile public up -d cloudflared
```
## Tests

Install the development dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Run lint from the repository root with:

```bash
python -m flake8 backend src tests
```

## Project structure

```text
backend/         FastAPI code, Dockerfile, .dockerignore, and requirements.txt
frontend/        Empty placeholder for the future React interface
configs/         Preprocessing configuration
model/           Label encoder used during training
src/             Data processing and MLflow training
processed_data/  Extracted features used during training
tests/           Unit and API tests
scripts/         Homeserver deployment script
requirements-dev.txt       Development dependencies, including backend requirements
requirements-training.txt  Training dependencies, including backend requirements
compose.yaml     MLflow, FastAPI, and optional Cloudflare Tunnel services
```

Install only the API dependencies with `pip install -r backend/requirements.txt`.
The development and training requirements include that same file; package
versions are unchanged.

`frontend/` is intentionally empty locally. Git does not track empty directories,
so it will not appear in a fresh clone until frontend files are added.

## Limitations

The API accepts WAV files only. For audio longer than 30 seconds, predictions
are averaged across up to ten 30-second chunks.
