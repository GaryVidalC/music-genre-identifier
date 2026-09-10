
import pickle
import json
import pytest
from app import model_loader


def test_load_metadata(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metadata = {
        "model": {"model_used" : "svm" },
        "preprocessing": {"sample_rate": 16000, "duration": 1.0, "normalize": True},
    }

    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    monkeypatch.setattr(model_loader, "ROOT_DIR", tmp_path)

    result = model_loader.load_metadata()

    assert result == metadata

def test_load_encoder(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = {"class_to_index": {"Rock": 0, "Pop": 1, "Jazz": 2}}

    with open(model_dir / "genre_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    monkeypatch.setattr(model_loader, "ROOT_DIR", tmp_path)

    result = model_loader.load_encoder()

    assert result == encoder

def test_load_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = {"model": "dummy_model"}

    with open(model_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    monkeypatch.setattr(model_loader, "ROOT_DIR", tmp_path)

    result = model_loader.load_model("XGboost")

    assert result == model

def test_load_model_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "ROOT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        model_loader.load_model("XGboost")