import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion


REQUIRED_METADATA_FIELDS = {
    "preprocessing",
    "genre_classes",
    "feature_names",
    "number_of_samples",
    "number_of_features",
}


def _build_metadata(
    custom_metadata: dict,
    model_version: ModelVersion,
    run: Run,
    model_alias: str,
) -> dict:
    """Convert model metadata, version, run and alias to API metadata."""
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


def export_model(
    output_dir: Path,
    tracking_uri: str = "http://localhost:5000",
    model_name: str = "music-genre-classifier",
    alias: str = "champion",
) -> Path:
    """Export name/alias from tracking_uri to output_dir; return its path."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    version = client.get_model_version_by_alias(
        name=model_name,
        alias=alias
    )
    model_uri = f"models:/{model_name}/{version.version}"

    model_info = mlflow.models.get_model_info(model_uri)
    run = client.get_run(model_info.run_id)

    metadata = _build_metadata(
        model_info.metadata or {},
        version,
        run,
        alias,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=temp_dir
        )
        shutil.copytree(
            downloaded_path,
            output_dir / "model",
            dirs_exist_ok=True
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return output_dir


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    exported_path = export_model(project_root / "model" / "exported")
    print(f"Model exported to: {exported_path}")
