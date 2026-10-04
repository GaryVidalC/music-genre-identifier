import os

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow import sklearn as mlflow_sklearn
from mlflow import xgboost as mlflow_xgboost
from sklearn.preprocessing import LabelEncoder

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MODEL_NAME = "music-genre-svc"
DEFAULT_MODEL_ALIAS = "champion"
REQUIRED_METADATA_FIELDS = {
    "preprocessing",
    "genre_classes",
    "feature_names",
    "number_of_samples",
    "number_of_features",
}


def _load_native_model(model_uri, flavors):
    """Load a registered model with its native MLflow flavor."""
    if "sklearn" in flavors:
        return mlflow_sklearn.load_model(model_uri)

    if "xgboost" in flavors:
        return mlflow_xgboost.load_model(model_uri)

    supported_flavors = ", ".join(sorted(flavors))
    raise ValueError(
        "Unsupported MLflow model flavor. "
        f"Available flavors: {supported_flavors}"
    )


def _build_metadata(custom_metadata, model_version, run, model_alias):
    """Convert MLflow model and run data into the FastAPI metadata contract."""
    missing_fields = REQUIRED_METADATA_FIELDS - custom_metadata.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"MLflow model metadata is missing: {missing}")

    preprocessing = custom_metadata["preprocessing"]
    feature_names = custom_metadata["feature_names"]
    genre_classes = custom_metadata["genre_classes"]

    if not isinstance(preprocessing, dict):
        raise ValueError("MLflow preprocessing metadata must be a dictionary")
    if not feature_names:
        raise ValueError("MLflow model metadata contains no feature names")
    if not genre_classes:
        raise ValueError("MLflow model metadata contains no genre classes")

    preprocessing = {
        **preprocessing,
        "features": list(feature_names),
    }

    return {
        "dataset": run.data.tags.get("dataset"),
        "data_info": {
            "number_of_samples": custom_metadata["number_of_samples"],
            "number_of_features": custom_metadata["number_of_features"],
            "genre_classes": list(genre_classes),
        },
        "preprocessing": preprocessing,
        "model": {
            "model_name": model_version.name,
            "model_used": run.data.tags.get(
                "model_type",
                model_version.name,
            ),
            "version": model_version.version,
            "alias": model_alias,
            "run_id": model_version.run_id,
        },
        "model_stats": dict(run.data.metrics),
        "model_params": dict(run.data.params),
    }


def load_model_resources():
    """Load a pinned MLflow model, its metadata, and its label encoder."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = client.get_model_version_by_alias(
        model_name,
        model_alias,
    )

    model_uri = f"models:/{model_name}/{model_version.version}"
    model_info = mlflow.models.get_model_info(model_uri)
    run = client.get_run(model_version.run_id)

    metadata = _build_metadata(
        model_info.metadata or {},
        model_version,
        run,
        model_alias,
    )
    model = _load_native_model(model_uri, model_info.flavors)

    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(
        metadata["data_info"]["genre_classes"],
    )

    return model, encoder, metadata
