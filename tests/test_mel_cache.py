"""Check power-mel caching with tiny temporary WAVs, without training."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch

from src import cache_jamendo_mels as cache_module
from src import jamendo_dataset as data
from src.neural import calculate_log_mel_spectrogram


@pytest.fixture
def files(tmp_path: Path) -> SimpleNamespace:
    """Create two short audio sources and official-style annotations."""
    annotations = tmp_path / "annotations"
    audio = tmp_path / "audio"
    annotations.mkdir()
    audio.mkdir()
    genres = ["rock", "jazz", "pop"]
    records = []
    rng = np.random.default_rng(42)
    for i in range(2):
        samples = rng.normal(0, 0.05, 44100 * (i + 2)).astype(np.float32)
        path = f"song{i}.wav"
        sf.write(audio / path, samples, 44100, subtype="FLOAT")
        records.append({"track_id": str(i), "path": path,
                        "labels": [1, i, 0]})
    (annotations / "genres.json").write_text(json.dumps(genres))
    for split, index in (("train", 0), ("validation", 1)):
        (annotations / f"{split}.json").write_text(
            json.dumps([records[index]]),
        )
    manifest = tmp_path / "subset.json"
    manifest.write_text(json.dumps({
        "genres": genres, "records": records[:1],
        "evaluation_records": records[1:],
    }))
    return SimpleNamespace(
        annotations=annotations, audio=audio, manifest=manifest,
        cache=tmp_path / "cache", records=records,
    )


def build(files: SimpleNamespace) -> dict:
    """Build the fixture cache and return its metadata."""
    return cache_module.build_cache(
        files.manifest, files.cache, files.audio, files.annotations,
    )


def test_full_song_equivalence_and_resume(
    files: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve full-song log-mels, labels, inputs and completed files."""
    original = [
        cache_module.source_info(files.audio / r["path"])
        for r in files.records
    ]
    index = build(files)
    assert index["complete"] and index["tracks"] == 2
    assert index["format"] == "mel_power_v1"
    assert original == [cache_module.source_info(files.audio / r["path"])
                        for r in files.records]
    dataset = data.JamendoDataset(
        "train", files.annotations, files.audio, "log_mel",
        mel_cache_dir=files.cache,
    )
    audio, sr = librosa.load(files.audio / "song0.wav", sr=44100, mono=True)
    expected = calculate_log_mel_spectrogram(audio, sr)
    cached, labels = dataset[0]
    np.testing.assert_allclose(cached.numpy(), expected, atol=1e-5, rtol=1e-5)
    assert cached.dtype == torch.float32 and cached.device.type == "cpu"
    assert labels.tolist() == files.records[0]["labels"]
    stamps = [p.stat().st_mtime_ns for p in sorted(files.cache.glob("*.npy"))]
    forbidden = Mock(side_effect=AssertionError("Unexpected audio processing"))
    monkeypatch.setattr(librosa, "load", forbidden)
    monkeypatch.setattr(librosa.feature, "melspectrogram", forbidden)
    monkeypatch.setattr(librosa, "get_duration", forbidden)
    dataset[0]
    assert build(files)["complete"]
    assert stamps == [p.stat().st_mtime_ns
                      for p in sorted(files.cache.glob("*.npy"))]
    forbidden.assert_not_called()


def test_windows_and_window_normalization(
    files: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep random train crops reproducible and central evaluation fixed."""
    assert data.MAX_LOG_MEL_DURATION == 20 * 60
    build(files)
    monkeypatch.setattr(data, "MAX_LOG_MEL_DURATION", 1)
    train = data.JamendoDataset(
        "train", files.annotations, files.audio, "log_mel",
        mel_cache_dir=files.cache,
    )
    torch.manual_seed(42)
    first = train[0][0]
    second = train[0][0]
    assert first.shape == (64, 87)
    assert not torch.equal(first, second)
    torch.manual_seed(42)
    torch.testing.assert_close(first, train[0][0])
    path = files.cache / "song1.npy"
    power = np.load(path)
    power[:, 0] = 1e12  # Outside the central window; must not set its dB ref.
    np.save(path, power, allow_pickle=False)
    validation = data.JamendoDataset(
        "validation", files.annotations, files.audio, "log_mel",
        mel_cache_dir=files.cache,
    )
    state = torch.get_rng_state().clone()
    actual = validation[0][0]
    torch.testing.assert_close(actual, validation[0][0])
    assert torch.equal(state, torch.get_rng_state())
    start = (power.shape[1] - 87) // 2
    expected = librosa.power_to_db(power[:, start:start + 87], ref=np.max)
    np.testing.assert_allclose(actual.numpy(), expected)
    assert actual.max().item() == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize("problem", [
    "missing", "corrupt", "shape", "dtype", "nan", "negative", "source",
    "parameters",
])
def test_invalid_cache(
    files: SimpleNamespace, problem: str,
) -> None:
    """Reject corrupt, incompatible or stale files without recomputation."""
    build(files)
    path = files.cache / "song0.npy"
    if problem == "missing":
        path.unlink()
    elif problem == "corrupt":
        path.write_bytes(b"truncated npy")
    elif problem == "source":
        (files.audio / "song0.wav").touch()
    elif problem == "parameters":
        index = json.loads((files.cache / "index.json").read_text())
        index["parameters"]["hop_length"] = 256
        cache_module.write_json(files.cache / "index.json", index)
    else:
        power = np.load(path)
        if problem == "dtype":
            power = power.astype(np.float64)
        elif problem == "shape":
            power = power[:, :-1]
        else:
            power[0, 0] = np.nan if problem == "nan" else -1
        np.save(path, power, allow_pickle=False)
    with pytest.raises(ValueError):
        data.JamendoDataset(
            "train", files.annotations, files.audio, "log_mel",
            mel_cache_dir=files.cache,
        )


def test_interrupted_generation(
    files: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed files and regenerate a partial temporary output."""
    save = np.save
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> None:
        """Write the first song, then simulate failure writing the second."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted write")
        save(*args, **kwargs)

    monkeypatch.setattr(np, "save", interrupt)
    with pytest.raises(RuntimeError, match="1: cache generation failed"):
        build(files)
    first_stamp = (files.cache / "song0.npy").stat().st_mtime_ns
    monkeypatch.setattr(np, "save", save)
    assert build(files)["complete"]
    assert (files.cache / "song0.npy").stat().st_mtime_ns == first_stamp
    assert not list(files.cache.glob("*.tmp"))


def test_raw_mode_and_loader(files: SimpleNamespace) -> None:
    """Keep uncached log-mels working and pass cache settings to loaders."""
    build(files)
    uncached = data.JamendoDataset(
        "train", files.annotations, files.audio, "log_mel",
    )
    assert uncached.mel_cache is None
    assert uncached[0][0].shape[0] == 64
    loader = data.create_jamendo_dataloader(
        "train", annotations_dir=files.annotations, audio_dir=files.audio,
        representation="log_mel", mel_cache_dir=files.cache,
    )
    songs, labels = next(iter(loader))
    assert len(songs) == 1 and labels.shape == (1, 3)
    with pytest.raises(ValueError, match="only supported for log_mel"):
        data.JamendoDataset(
            "train", files.annotations, files.audio,
            mel_cache_dir=files.cache,
        )
