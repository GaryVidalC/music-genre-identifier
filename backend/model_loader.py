import json
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.models import Model
from mlflow import sklearn as mlflow_sklearn
from mlflow import xgboost as mlflow_xgboost
from sklearn.preprocessing import LabelEncoder

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "model" / "exported"
)


def _load_native_model(model_path: str, flavors: dict) -> Any:
    """Load model_path using the native sklearn or XGBoost flavor."""
    if "sklearn" in flavors:
        return mlflow_sklearn.load_model(model_path)

    if "xgboost" in flavors:
        return mlflow_xgboost.load_model(model_path)

    supported_flavors = ", ".join(sorted(flavors))
    raise ValueError(
        "Unsupported MLflow model flavor. "
        f"Available flavors: {supported_flavors}"
    )


def load_model_resources() -> tuple[Any, LabelEncoder, dict]:
    """Load the local export; return model, encoder and API metadata."""
    with (DEFAULT_MODEL_DIR / "metadata.json").open(
        "r", encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    required_fields = {
        "data_info", "preprocessing", "model", "model_stats", "model_params",
    }
    missing_fields = required_fields - metadata.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Exported model metadata is missing: {missing}")

    genre_classes = metadata["data_info"].get("genre_classes")
    if not genre_classes:
        raise ValueError("Exported model metadata contains no genre classes")
    if not metadata["preprocessing"].get("features"):
        raise ValueError("Exported model metadata contains no feature names")

    model_path = str(DEFAULT_MODEL_DIR / "model")
    model_config = Model.load(str(Path(model_path) / "MLmodel"))
    model = _load_native_model(model_path, model_config.flavors)

    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(genre_classes)

    return model, encoder, metadata
