# Running locally

Run commands from the repository root unless noted otherwise. The included
`model/exported/` package is enough for predictions; no MLflow or retraining
is needed.

## Docker Compose

Requires Docker Engine and Docker Compose.

```bash
docker compose up -d --build api frontend
```

- Frontend: http://localhost:8081
- API documentation: http://localhost:8080/docs
- Readiness through Nginx: http://localhost:8081/api/ready

Ports are bound to localhost. MLflow and Cloudflare are optional profiles,
not started by this command.

```bash
docker compose logs --tail=50 api frontend
docker compose stop api frontend
```

## Development without Docker

Requires Python 3.12, Node 24, and FFmpeg for YouTube conversion. On Linux,
the backend also uses `libsndfile1` and `libgomp1`.

Start the API in one terminal:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
```

Start Vite in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173. Vite forwards `/api/` to the backend on port 8080.
Stop any Docker API using that port before starting the manual API.

## Checks

With the virtual environment activated, from the repository root:

```bash
python -m pytest -q
python -m flake8 backend src tests
docker compose config --quiet
```

From `frontend/` after `npm ci`:

```bash
npm test
npm run build
```

The frontend build includes TypeScript checks. Tests use mocked predictions
and downloads; they do not need a running MLflow server.
