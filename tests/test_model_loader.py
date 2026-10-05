from types import SimpleNamespace

import pytest

from app import model_loader


CUSTOM_METADATA = {
    "preprocessing": {
        "sample_rate": 22050,
        "mono": True,
        "duration": 30,
        "normalization": True,
        "n_mfcc": 12,
    },
    "genre_classes": ["blues", "jazz", "rock"],
    "feature_names": ["mean_mfcc_0", "std_mfcc_0"],
    "number_of_samples": 999,
    "number_of_features": 2,
}


def configure_mlflow(
    monkeypatch,
    flavors,
    custom_metadata=None,
    model_name="music-genre-classifier",
    model_alias="champion",
):
    """Configure mocked MLflow registry, run, and model information."""
    version = SimpleNamespace(
        name=model_name,
        version=7,
        run_id="run-123",
    )
    run = SimpleNamespace(
        data=SimpleNamespace(
            metrics={"f1_score": 0.75},
            params={"C": "2.2", "gamma": "scale"},
            tags={"dataset": "GTZAN", "model_type": "SVC"},
        ),
    )
    model_info = SimpleNamespace(
        metadata=(
            CUSTOM_METADATA
            if custom_metadata is None
            else custom_metadata
        ),
        flavors=flavors,
    )

    class FakeClient:
        def __init__(self, tracking_uri):
            self.tracking_uri = tracking_uri

        def get_model_version_by_alias(self, name, alias):
            assert name == model_name
            assert alias == model_alias
            return version

        def get_run(self, run_id):
            assert run_id == version.run_id
            return run

    monkeypatch.setattr(model_loader, "MlflowClient", FakeClient)
    monkeypatch.setattr(
        model_loader.mlflow.models,
        "get_model_info",
        lambda uri: model_info,
    )

    return version


def test_load_model_resources_uses_defaults_and_sklearn(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_MODEL_ALIAS", raising=False)
    configure_mlflow(monkeypatch, {"python_function": {}, "sklearn": {}})

    tracking_uris = []
    loaded_uris = []
    expected_model = object()
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        tracking_uris.append,
    )
    monkeypatch.setattr(
        model_loader.mlflow_sklearn,
        "load_model",
        lambda uri: loaded_uris.append(uri) or expected_model,
    )

    model, encoder, metadata = model_loader.load_model_resources()

    assert model is expected_model
    assert tracking_uris == ["http://localhost:5000"]
    assert loaded_uris == ["models:/music-genre-classifier/7"]
    assert encoder.inverse_transform([0, 2]).tolist() == ["blues", "rock"]
    assert metadata == {
        "dataset": "GTZAN",
        "data_info": {
            "number_of_samples": 999,
            "number_of_features": 2,
            "genre_classes": ["blues", "jazz", "rock"],
        },
        "preprocessing": {
            "sample_rate": 22050,
            "mono": True,
            "duration": 30,
            "normalization": True,
            "n_mfcc": 12,
            "features": ["mean_mfcc_0", "std_mfcc_0"],
        },
        "model": {
            "model_name": "music-genre-classifier",
            "model_used": "SVC",
            "version": 7,
            "alias": "champion",
            "run_id": "run-123",
        },
        "model_stats": {"f1_score": 0.75},
        "model_params": {"C": "2.2", "gamma": "scale"},
    }


def test_load_model_resources_uses_environment_and_xgboost(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "music-genre-xgboost")
    monkeypatch.setenv("MLFLOW_MODEL_ALIAS", "challenger")
    configure_mlflow(
        monkeypatch,
        {"python_function": {}, "xgboost": {}},
        model_name="music-genre-xgboost",
        model_alias="challenger",
    )

    expected_model = object()
    loaded_uris = []
    tracking_uris = []
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        tracking_uris.append,
    )
    monkeypatch.setattr(
        model_loader.mlflow_xgboost,
        "load_model",
        lambda uri: loaded_uris.append(uri) or expected_model,
    )

    model, _, metadata = model_loader.load_model_resources()

    assert model is expected_model
    assert tracking_uris == ["http://mlflow:5000"]
    assert loaded_uris == ["models:/music-genre-xgboost/7"]
    assert metadata["model"]["alias"] == "challenger"


def test_load_model_resources_rejects_missing_metadata(monkeypatch):
    configure_mlflow(
        monkeypatch,
        {"python_function": {}, "sklearn": {}},
        custom_metadata={"preprocessing": {}},
    )
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        lambda uri: None,
    )

    with pytest.raises(ValueError, match="metadata is missing"):
        model_loader.load_model_resources()


def test_load_model_resources_rejects_unsupported_flavor(monkeypatch):
    configure_mlflow(monkeypatch, {"python_function": {}})
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        lambda uri: None,
    )

    with pytest.raises(ValueError, match="Unsupported MLflow model flavor"):
        model_loader.load_model_resources()
