import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import mlflow

from backend import model_loader
from src.export_model import _build_metadata


EXPORTED_METADATA = {
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
        "version": "7",
        "alias": "champion",
        "run_id": "run-123",
    },
    "model_stats": {"f1_score": 0.75},
    "model_params": {"C": "2.2", "gamma": "scale"},
}


def configure_local_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flavors: dict,
    metadata: dict | None = None,
) -> str:
    """Write metadata under tmp_path and mock flavors; return model path."""
    metadata = EXPORTED_METADATA if metadata is None else metadata
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8",
    )
    model_path = str(tmp_path / "model")
    monkeypatch.setattr(model_loader, "DEFAULT_MODEL_DIR", tmp_path)

    def load_config(path: str) -> SimpleNamespace:
        """Check the requested local path; return mocked flavor details."""
        assert path == str(tmp_path / "model" / "MLmodel")
        return SimpleNamespace(flavors=flavors)

    def reject_tracking(uri: str) -> None:
        """Fail if the local loader attempts to configure a tracking URI."""
        pytest.fail("Local loading must not configure MLflow tracking")

    monkeypatch.setattr(
        model_loader.Model, "load", load_config,
    )
    monkeypatch.setattr(
        mlflow, "set_tracking_uri", reject_tracking,
    )
    return model_path


def test_load_model_resources_uses_local_sklearn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Verify local sklearn loading, encoder and unchanged export metadata."""
    model_path = configure_local_model(
        monkeypatch, tmp_path, {"python_function": {}, "sklearn": {}},
    )
    expected_model = object()
    loaded_paths = []
    monkeypatch.setattr(
        model_loader.mlflow_sklearn,
        "load_model",
        lambda path: loaded_paths.append(path) or expected_model,
    )

    model, encoder, metadata = model_loader.load_model_resources()

    assert model is expected_model
    assert loaded_paths == [model_path]
    assert encoder.inverse_transform([0, 2]).tolist() == ["blues", "rock"]
    assert metadata == EXPORTED_METADATA

    custom_metadata = {
        "preprocessing": {
            key: value
            for key, value in metadata["preprocessing"].items()
            if key != "features"
        },
        "genre_classes": metadata["data_info"]["genre_classes"],
        "feature_names": metadata["preprocessing"]["features"],
        "number_of_samples": 999,
        "number_of_features": 2,
    }
    version = SimpleNamespace(
        name="music-genre-classifier", version="7", run_id="run-123",
    )
    run = SimpleNamespace(data=SimpleNamespace(
        metrics=metadata["model_stats"],
        params=metadata["model_params"],
        tags={"dataset": "GTZAN", "model_type": "SVC"},
    ))
    assert _build_metadata(custom_metadata, version, run, "champion")\
        == EXPORTED_METADATA


def test_load_model_resources_uses_local_xgboost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Verify native XGBoost loading without consulting a remote registry."""
    model_path = configure_local_model(
        monkeypatch, tmp_path, {"python_function": {}, "xgboost": {}},
    )
    expected_model = object()
    loaded_paths = []
    monkeypatch.setattr(
        model_loader.mlflow_xgboost,
        "load_model",
        lambda path: loaded_paths.append(path) or expected_model,
    )

    model, _, metadata = model_loader.load_model_resources()

    assert model is expected_model
    assert loaded_paths == [model_path]
    assert metadata == EXPORTED_METADATA


def test_load_model_resources_rejects_missing_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Reject incomplete/missing local metadata and invalid export input."""
    configure_local_model(
        monkeypatch, tmp_path, {"sklearn": {}},
        metadata={"preprocessing": {}},
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        model_loader.load_model_resources()

    (tmp_path / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError):
        model_loader.load_model_resources()

    with pytest.raises(ValueError, match="metadata is missing"):
        _build_metadata({}, SimpleNamespace(), SimpleNamespace(), "champion")


def test_load_model_resources_rejects_unsupported_flavor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Reject a local model that exposes no supported native flavor."""
    configure_local_model(monkeypatch, tmp_path, {"python_function": {}})
    with pytest.raises(ValueError, match="Unsupported MLflow model flavor"):
        model_loader.load_model_resources()
