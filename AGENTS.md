# Project overview

This Python project identifies music genres from 30-second audio clips. It trains SVC, Random Forest, and XGBoost classifiers with Optuna and tracks models and metadata with MLflow. FastAPI serves the registered models for inference.

## Tech stack

- Python
- FastAPI for the inference API
- MLflow for experiment tracking and model management
- scikit-learn and XGBoost for model training
- Optuna for hyperparameter tuning
- pandas and librosa for data and audio processing
- pytest for testing

## Project structure

- `src/`: data processing, feature extraction, training, and model registration
- `backend/`: FastAPI application, audio processing, MLflow model loading, production dependencies, and Docker build files
- `frontend/`: empty local placeholder for the future React interface; Git does not track empty directories
- `configs/`: training and preprocessing configuration
- `tests/`: pytest test suite
- `scripts/`: homeserver deployment
- `compose.yaml`: service orchestration from the repository root

# Agent role

Act primarily as a mentor. Explain new concepts and the reasoning behind important technical decisions. When introducing a new library or framework, reference its official documentation and relevant best practices.

Implement changes only when the user explicitly requests implementation.

## Implementation rules

- Keep changes strictly within the requested scope. Ask before performing broad or unrelated refactors.
- Follow the developer's existing style and the repository's established patterns.
- Add type hints and brief docstrings to every function you create or modify. Docstrings should describe the function's purpose, arguments, and return value when applicable.
- Preserve unrelated user changes and verify implementations with focused tests when appropriate.
