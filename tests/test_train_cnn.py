"""Exercise CNN orchestration with simulated epochs, GPU and MLflow."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from src import train_cnn_test as trainer


class AnnotationOnlyDataset:
    """Provide annotations while forbidding audio loading in these tests."""

    def __init__(self, split: str, count: int) -> None:
        """Build unique track IDs and a shared genre vocabulary."""
        self.genres = [f"genre_{i}" for i in range(87)]
        self.records = [
            {"track_id": f"{split}_{i}", "path": f"{i}.mp3",
             "labels": [1] + [0] * 86}
            for i in range(count)
        ]

    def __len__(self) -> int:
        """Return the annotation count without accessing audio."""
        return len(self.records)

    def __getitem__(self, index: int) -> None:
        """Fail if simulated orchestration attempts to decode audio."""
        raise AssertionError("Tests must not load songs.")


@pytest.fixture
def harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> SimpleNamespace:
    """Mock tracking, CUDA and model allocation; retain real file saving."""
    tracking = MagicMock()
    tracking.__version__ = "test"
    tracking.start_run.return_value.__enter__.return_value.info.run_id = "run1"
    monkeypatch.setattr(trainer, "mlflow", tracking)
    monkeypatch.setattr(trainer, "__file__", str(tmp_path / "src" / "run.py"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    for name, value in (
        ("is_available", True), ("get_device_name", "fake GPU"),
        ("reset_peak_memory_stats", None), ("manual_seed_all", None),
        ("max_memory_allocated", 100), ("max_memory_reserved", 200),
    ):
        monkeypatch.setattr(torch.cuda, name, Mock(return_value=value))

    def select(
        split: str, count: int, random_crop: bool | None = None,
    ) -> AnnotationOnlyDataset:
        """Return annotations only, preserving split/count selection."""
        return AnnotationOnlyDataset(split, count)

    monkeypatch.setattr(trainer, "select_dataset", select)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(model, "to", Mock(return_value=model))
    constructor = Mock(return_value=model)
    monkeypatch.setattr(trainer, "MGI", constructor)
    optimizer = Mock()
    optimizer.state_dict.return_value = {"fake_optimizer": True}
    monkeypatch.setattr(torch.optim, "AdamW", Mock(return_value=optimizer))
    saves = Mock(wraps=trainer.save_checkpoint)
    monkeypatch.setattr(trainer, "save_checkpoint", saves)
    return SimpleNamespace(
        tracking=tracking, model=model, constructor=constructor,
        saves=saves, root=tmp_path,
    )


def simulate(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
    scores: list[float], failure: BaseException | None = None,
) -> None:
    """Replace epoch work with scores and marker weights; perform no fit."""
    position = 0

    def epoch(
        model: torch.nn.Module, loader: object, device: object,
        optimizer: object = None, evaluation_prefix: str = "train_eval",
    ) -> tuple[dict, list]:
        """Return fake losses/scores; optionally interrupt a later epoch."""
        nonlocal position
        if optimizer is not None:
            if failure is not None and position == 2:
                raise failure
            with torch.no_grad():
                model.weight.fill_(position)
            return {"train_bce": 0.1}, []
        score = scores[min(position, len(scores) - 1)]
        position += 1
        return {
            f"{evaluation_prefix}_map_macro": score,
            f"{evaluation_prefix}_bce": 0.2,
            f"{evaluation_prefix}_f1_micro": 0.3,
        }, [[score]]

    monkeypatch.setattr(trainer, "run_epoch", epoch)


def test_best_and_patience(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
) -> None:
    """Ignore baseline, reject ties, reset patience and stop at its limit."""
    simulate(monkeypatch, harness, [0.95, 0.2, 0.2, 0.3, 0.3, 0.3])
    output = trainer.train(20, train_songs=17, val_songs=3, patience=2)
    best = torch.load(output / "best.pt", weights_only=True)
    last = torch.load(output / "last.pt", weights_only=True)
    assert best["epoch"] == best["best_epoch"] == 3
    assert best["epochs_without_improvement"] == 0
    assert last["epoch"] == 5
    assert last["best_epoch"] == 3
    assert last["epochs_without_improvement"] == 2
    assert best["model"]["weight"].item() == 3
    assert last["model"]["weight"].item() == 5
    assert [call.args[1]["epoch"] for call in harness.saves.call_args_list
            if call.args[0].name == "best.pt"] == [1, 3]
    baseline = harness.saves.call_args_list[0].args[1]
    assert baseline["epoch"] == 0 and baseline["best_map"] is None
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_reason"] == "early_stopping"
    assert summary["best_epoch"] == 3
    assert summary["last_epoch"] == summary["probabilities_epoch"] == 5
    assert json.loads((output / "probabilities.json").read_text()) == [[0.3]]
    assert [call.kwargs["step"] for call in
            harness.tracking.log_metrics.call_args_list] == list(range(6))
    assert {Path(call.args[0]).name for call in
            harness.tracking.log_artifact.call_args_list} == {
        "config.json", "history.json", "summary.json", "probabilities.json",
        "best.pt", "last.pt",
    }
    params = harness.tracking.log_params.call_args.args[0]
    assert params["patience"] == 2 and params["songs"] == 17
    assert "records" not in params and "genres" not in params
    config = json.loads((output / "config.json").read_text())
    assert len(config["records"]) == 17 and len(config["genres"]) == 87
    assert config["run_id"] == "run1" and len(config["subset_sha256"]) == 64
    assert config["max_audio_duration_seconds"] == 20 * 60


def test_default_patience(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
) -> None:
    """Stop after exactly ten non-improving epochs by default."""
    simulate(monkeypatch, harness, [0.9, 0.2])
    output = trainer.train(30, train_songs=17, val_songs=2)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["last_epoch"] == 11
    assert summary["epochs_without_improvement"] == 10
    assert summary["stop_reason"] == "early_stopping"


def test_memorization_does_not_stop(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
) -> None:
    """Keep all requested epochs without independent validation."""
    simulate(monkeypatch, harness, [0.9, 0.2])
    output = trainer.train(4, train_songs=17, patience=1)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["last_epoch"] == 4
    assert summary["best_epoch"] == 1
    assert summary["stop_reason"] == "max_epochs"
    config = json.loads((output / "config.json").read_text())
    assert config["early_stopping_enabled"] is False
    assert config["selection_metric"] == "train_eval_map_macro"


@pytest.mark.parametrize("stage", ["set_experiment", "start_run"])
def test_tracking_failure_prevents_work(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace, stage: str,
) -> None:
    """Fail before reading annotations/audio or assigning a model to GPU."""
    getattr(harness.tracking, stage).side_effect = RuntimeError("offline")
    selection = Mock(side_effect=AssertionError("Must fail before data"))
    monkeypatch.setattr(trainer, "select_dataset", selection)
    with pytest.raises(RuntimeError, match="offline"):
        trainer.train(2)
    selection.assert_not_called()
    harness.constructor.assert_not_called()
    torch.cuda.is_available.assert_not_called()


@pytest.mark.parametrize("failure", [
    RuntimeError("oops"), KeyboardInterrupt(),
])
def test_failure_preserves_completed_checkpoints(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
    failure: BaseException,
) -> None:
    """Propagate failures to the run context and preserve the last epoch."""
    simulate(monkeypatch, harness, [0.9, 0.2], failure)
    with pytest.raises(type(failure)):
        trainer.train(5, train_songs=17, val_songs=2)
    output = next((harness.root / "data/experiments/cnn_subset").iterdir())
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert (output / "best.pt").is_file()
    summary = json.loads((output / "summary.json").read_text())
    expected = (
        "interrupted" if isinstance(failure, KeyboardInterrupt) else "error"
    )
    assert summary["stop_reason"] == expected
    assert summary["last_epoch"] == 1
    exit_call = harness.tracking.start_run.return_value.__exit__.call_args
    assert exit_call.args[0] is type(failure)


@pytest.mark.parametrize("argument,environment,expected", [
    ("http://argument", "http://environment", "http://argument"),
    (None, "http://environment", "http://environment"),
    (None, None, "http://localhost:5000"),
])
def test_tracking_uri_precedence(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
    argument: str | None, environment: str | None, expected: str,
) -> None:
    """Resolve tracking URI without starting any model or epoch work."""
    if environment:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", environment)
    work = Mock(return_value=Path("unused"))
    monkeypatch.setattr(trainer, "_train_run", work)
    trainer.train(1, tracking_uri=argument)
    harness.tracking.set_tracking_uri.assert_called_once_with(expected)
    harness.tracking.set_experiment.assert_called_once_with(
        "jamendo_cnn_transformer",
    )


def test_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass flags through, preserve the default rate and validate patience."""
    work = Mock()
    monkeypatch.setattr(trainer, "train", work)
    monkeypatch.setattr("sys.argv", ["train_cnn_test"])
    trainer.main()
    assert work.call_args.kwargs["learning_rate"] == 1e-4
    work.reset_mock()
    monkeypatch.setattr("sys.argv", [
        "train_cnn_test", "--patience", "3", "--tracking-uri", "http://local",
        "--mel-cache-dir", "cache",
        "--learning-rate", "3e-5",
    ])
    trainer.main()
    assert work.call_args.kwargs["patience"] == 3
    assert work.call_args.kwargs["tracking_uri"] == "http://local"
    assert work.call_args.kwargs["mel_cache_dir"] == Path("cache")
    assert work.call_args.kwargs["learning_rate"] == 3e-5
    work.reset_mock()
    monkeypatch.setattr("sys.argv", ["train_cnn_test", "--patience", "0"])
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2
    work.assert_not_called()


@pytest.mark.parametrize("rate", [None, 3e-5])
def test_learning_rate_recorded_and_used(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
    rate: float | None,
) -> None:
    """Use the same default/custom rate in AdamW, tracking and checkpoints."""
    simulate(monkeypatch, harness, [0.1, 0.2])
    options = {} if rate is None else {"learning_rate": rate}
    expected = 1e-4 if rate is None else rate
    output = trainer.train(1, train_songs=17, val_songs=2, **options)
    assert torch.optim.AdamW.call_args.kwargs["lr"] == expected
    assert harness.tracking.log_params.call_args.args[0][
        "learning_rate"
    ] == expected
    config = json.loads((output / "config.json").read_text())
    assert config["learning_rate"] == expected
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["config"]["learning_rate"] == expected


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf"),
                                  float("-inf")])
def test_invalid_learning_rate(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace, rate: float,
) -> None:
    """Reject invalid rates before tracking, audio or GPU allocation."""
    with pytest.raises(ValueError, match="finite and positive"):
        trainer.train(1, learning_rate=rate)
    monkeypatch.setattr("sys.argv", [
        "train_cnn_test", f"--learning-rate={rate}",
    ])
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2
    harness.tracking.set_tracking_uri.assert_not_called()
    harness.constructor.assert_not_called()
    torch.cuda.is_available.assert_not_called()


def test_cache_failure_prevents_gpu_model(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
) -> None:
    """Validate cache coverage before assigning model parameters to CUDA."""
    validator = Mock(side_effect=ValueError("missing cache"))
    monkeypatch.setattr(
        AnnotationOnlyDataset, "enable_mel_cache", validator, raising=False,
    )
    with pytest.raises(ValueError, match="missing cache"):
        trainer.train(1, train_songs=17, val_songs=2,
                      mel_cache_dir=Path("cache"))
    validator.assert_called_once_with(Path("cache"))
    harness.constructor.assert_not_called()


def test_cache_metadata_recorded(
    monkeypatch: pytest.MonkeyPatch, harness: SimpleNamespace,
) -> None:
    """Record cache provenance without uploading its per-song file index."""
    metadata = {
        "format": "mel_power_v1", "parameters": {"hop_length": 512},
        "entries": {"track": "not part of config"},
    }

    def enable(dataset: AnnotationOnlyDataset, directory: Path) -> None:
        """Simulate validated cache selection without reading features."""
        dataset.mel_cache = SimpleNamespace(index=metadata)

    monkeypatch.setattr(
        AnnotationOnlyDataset, "enable_mel_cache", enable, raising=False,
    )
    simulate(monkeypatch, harness, [0.1, 0.2])
    output = trainer.train(
        1, train_songs=17, val_songs=2, mel_cache_dir=Path("cache"),
    )
    config = json.loads((output / "config.json").read_text())
    assert config["mel_cache_metadata"] == {
        key: value for key, value in metadata.items() if key != "entries"
    }
    params = harness.tracking.log_params.call_args.args[0]
    assert params["feature_source"] == "mel_power_cache_v1"
    assert params["mel_cache_dir"] == str(Path("cache").resolve())
