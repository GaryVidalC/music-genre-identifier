# Music Genre Identifier

Classify music from WAV uploads or YouTube links through a React interface and a FastAPI backend. Training and model comparison use Optuna and MLflow.

The supported genres are blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

Use Docker Compose to start the frontend and API. A model export is included; MLflow and retraining are not required to run predictions.


## Project structure

```text
backend/         FastAPI, inference, local model loader, Dockerfile and dependencies
frontend/        React + TypeScript, Vite, tests, Dockerfile and Nginx configuration
configs/         Preprocessing configuration
model/           Training encoder and exported deployment model/metadata
src/             Data processing, MLflow training and model export
processed_data/  Extracted features used during training
tests/           Unit and API tests
scripts/         Homeserver deployment script
requirements-dev.txt       Development dependencies, including backend requirements
requirements-training.txt  Training dependencies, including backend requirements
compose.yaml     Frontend, FastAPI, optional MLflow and Cloudflare Tunnel
```

Install only the API dependencies with `pip install -r backend/requirements.txt`.
The development and training requirements include that same file; package
versions are unchanged.

## Data and Model

The project uses the [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The processed dataset contains 999 usable clips across the ten genres.

Each clip is converted to mono at 22,050 Hz, normalized, and standardized to 30 seconds. The model uses 54 summary features extracted with librosa:

- Mean and standard deviation of 12 MFCC coefficients: 24 features
- Mean and standard deviation of 12 chroma bins: 24 features
- Mean and standard deviation of spectral centroid, bandwidth, and rolloff: 6 features

SVC, Random Forest, and XGBoost were hyperparameter-tuned with Optuna and tracked with MLflow using a fixed random seed and 30 trials per model.

### Results

Three classifiers were trained and evaluated on the same validation split.

| Model | F1 score | Precision | Recall | Inference time (seconds) |
|---|---:|---:|---:|---:|
| SVC | **0.73** | **0.75** | **0.73** | **0.008** |
| Random Forest | 0.69 | 0.71 | 0.69 | 0.011 |
| XGBoost | 0.68 | 0.70 | 0.69 | 0.005 |

The current exported champion is SVC with parameters:

- C = 2.223
- gamma = scale

### How inference works

```text
WAV upload or downloaded YouTube audio
    -> format and size validation
    -> mono conversion and resampling to 22,050 Hz
    -> audio feature extraction
    -> champion model prediction
    -> genre label and probabilities for every class
```

The model processes audio in 30-second segments. For longer audio, it averages the class probabilities across up to ten segments.

### Model API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Confirms that the API process is running |
| `GET` | `/ready` | Reports whether the model, encoder, and metadata are loaded |
| `GET` | `/model-info` | Returns model and preprocessing metadata |
| `POST` | `/predict-audio` | Accepts a WAV upload and returns the predicted genre and probabilities by class |
| `POST` | `/predict-youtube?url=...` | Downloads a single YouTube video's audio and returns the same prediction response |

FastAPI also exposes interactive documentation at `http://localhost:8080/docs`.

### Live application

Hosted on a Debian homeserver with 4 GB of RAM. Cloudflare Tunnel routes
`music-genre-identifier.gvidal.cl` to `http://frontend:80`. Nginx serves the
interface and forwards `/api/` to FastAPI. MLflow stays stopped during inference.

- [Try the application](https://music-genre-identifier.gvidal.cl)
- [Model information](https://music-genre-identifier.gvidal.cl/api/model-info)

WAV uploads and YouTube predictions have been verified through the public
frontend. The previous `api.gvidal.cl` route is no longer needed.

The JSON response contains `predicted_genre`, the class with the highest
probability, and `probabilities`, a mapping from each genre to a value between
0 and 1.

### Previous Cloud Run deployment (outdated)

The previous deployment used Google Cloud Run. These links are retained as
historical references; the current API uses the homeserver deployment above.

- [Base URL](https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app)
- [Interactive docs](https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app/docs)

## Running locally

See [Running locally](RUN_LOCAL.md) for requirements, Docker Compose startup,
development without Docker, and verification commands.

## Training and model export

Only needed to train or change the deployed model:

```bash
pip install -r requirements-training.txt
docker compose --profile training up -d mlflow
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.model_selection
```

Each Optuna trial logs parameters and metrics, not a model. Training registers
one final model per family, assigns `champion` to the highest-F1 candidate,
and automatically exports its model and metadata to `model/exported/`.

To choose a different champion without retraining, change the `champion` alias
in the MLflow registry, then export it again with the local MLflow server running:

```bash
python -m src.export_model
```

Include the updated `model/exported/` in your commit before deployment. Rebuild
locally with:

```bash
docker compose up -d --build api frontend
docker compose stop mlflow
```

Running training again replaces a manual champion selection with the
highest-F1 candidate. The API does not reload model changes until restarted
with the updated image.

## CI/CD

CI runs Python lint/tests, frontend tests and TypeScript/build checks, validates
Compose, and builds both images using the committed model export.

After CI succeeds for a push to `main`, CD connects to the homeserver through
Tailscale and restricted SSH, checks out the approved commit, builds and
recreates both services, and checks the frontend and `/api/ready`.

When the deployment script changes, reinstall it manually on the homeserver:

```bash
sudo install -m 0755 -o root -g root \
  scripts/deploy-homeserver.sh /usr/local/bin/deploy-music-genre
```

## Limitations

- Uploads and downloaded audio are limited to 100 MiB. Downloads are checked
  by blocks and can exceed the limit by a block before being aborted.
- YouTube playlists are not supported. The full audio is downloaded, but only
  up to ten 30-second chunks are analyzed.
- The API uses a single worker to reduce RAM usage on the 4 GB homeserver.
  WAV and YouTube share one processing slot; overlapping requests receive
  HTTP `429` to avoid simultaneous downloads and inference.
- Oversized audio returns `413`, download failures `502`, and internal YouTube
  processing errors `500` without exposing technical details.
