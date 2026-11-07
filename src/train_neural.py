"""Train and validate the Jamendo Transformer with manual MLflow tracking."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.jamendo_dataset import (
    MAX_SNIPPETS,
    SAMPLE_RATE,
    SNIPPET_DURATION,
    create_jamendo_dataloader,
)
from src.neural import MusicGenreTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "d_model": 512,
    "dim_feedforward": 2048,
    "num_layers": 4,
    "num_heads": 8,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "seed": 42,
    "max_train_songs": None,
    "max_val_songs": None,
    "annotations_dir": str(PROJECT_ROOT / "data" / "jamendo"),
    "audio_dir": str(PROJECT_ROOT / "data" / "audio-low"),
}


def parse_args() -> argparse.Namespace:
    """Return CLI settings; omitted options inherit checkpoint settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total epochs, including completed epochs.")
    for name in (
        "d-model", "dim-feedforward", "num-layers", "num-heads",
        "batch-size", "seed", "max-train-songs", "max-val-songs",
    ):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment")
    parser.add_argument("--resume", type=Path,
                        help="Resume last.pt from a completed epoch.")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    return args


def build_config(args: argparse.Namespace, saved: dict) -> dict:
    """Resolve CLI/checkpoint defaults and reject incompatible resumption."""
    config = {}
    for name, default in DEFAULTS.items():
        value = getattr(args, name)
        value = saved.get(name, default) if value is None else value
        if name.endswith("_dir"):
            value = str(Path(value).resolve())
        config[name] = value
    for name in (
        "d_model", "dim_feedforward", "num_layers", "num_heads",
        "batch_size", "max_train_songs", "max_val_songs",
    ):
        if config[name] is not None and config[name] <= 0:
            raise ValueError(f"{name} must be positive.")
    if config["d_model"] % config["num_heads"]:
        raise ValueError("num_heads must divide d_model.")
    if config["seed"] < 0 or config["seed"] >= 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1.")
    for name in ("learning_rate", "weight_decay"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if config["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive.")
    if saved and config != saved:
        raise ValueError("Resume must preserve the checkpoint configuration.")
    return config


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader, dict]:
    """Build loaders and annotation fingerprints without loading audio."""
    loaders = []
    fingerprints = {}
    for split, limit_name in (
        ("train", "max_train_songs"), ("validation", "max_val_songs"),
    ):
        loader = create_jamendo_dataloader(
            split, batch_size=config["batch_size"], seed=config["seed"],
            annotations_dir=Path(config["annotations_dir"]),
            audio_dir=Path(config["audio_dir"]),
        )
        limit = config[limit_name]
        if limit is not None and limit < len(loader.dataset):
            generator = torch.Generator().manual_seed(config["seed"])
            indices = torch.randperm(
                len(loader.dataset), generator=generator,
            )[:limit].tolist()
            loader.dataset.records = [
                loader.dataset.records[index] for index in indices
            ]
        if not len(loader.dataset):
            raise ValueError(f"{split} contains no songs.")
        serialized = json.dumps(loader.dataset.records, sort_keys=True)
        fingerprints[split] = hashlib.sha256(serialized.encode()).hexdigest()
        loaders.append(loader)
    if loaders[0].dataset.genres != loaders[1].dataset.genres:
        raise ValueError("Train and validation must share the genre order.")
    return loaders[0], loaders[1], fingerprints


def validation_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict:
    """Return macro AP on supported genres and F1 at threshold 0.5.

    Args:
        labels: CPU binary targets [songs, genres].
        scores: CPU sigmoid probabilities with the same shape.
    """
    targets = labels.numpy()
    probabilities = scores.numpy()
    supported = targets.sum(axis=0) > 0
    if not supported.any():
        raise ValueError("Validation needs at least one positive label.")
    predictions = probabilities >= 0.5
    return {
        "val_map_macro": float(average_precision_score(
            targets[:, supported], probabilities[:, supported],
            average="macro",
        )),
        "val_ap_genres": int(supported.sum()),
        "val_f1_macro": float(f1_score(
            targets, predictions, average="macro", zero_division=0,
        )),
        "val_f1_micro": float(f1_score(
            targets, predictions, average="micro", zero_division=0,
        )),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Train or validate one loader; return song-weighted BCE and metrics.

    Passing an optimizer enables training; otherwise use inference mode and
    collect validation probabilities on CPU, without retaining GPU graphs.
    """
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    song_count = 0
    targets, scores = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for patches, labels, padding_mask in loader:
            patches = patches.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            padding_mask = padding_mask.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(patches, padding_mask)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite loss or predictions.")
            if training:
                loss.backward()
                if any(
                    not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ):
                    raise RuntimeError("Non-finite training gradients.")
                optimizer.step()
            else:
                targets.append(labels.cpu())
                scores.append(logits.sigmoid().cpu())
            total_loss += loss.item() * labels.shape[0]
            song_count += labels.shape[0]
    if not song_count:
        raise ValueError("Cannot run an epoch on an empty loader.")
    if training:
        optimizer.zero_grad(set_to_none=True)
        return {"train_bce": total_loss / song_count}
    return {
        "val_bce": total_loss / song_count,
        **validation_metrics(torch.cat(targets), torch.cat(scores)),
    }


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomically replace a checkpoint with the supplied training state."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> Path:
    """Run configured epochs with MLflow; return the local run directory.

    Resume only trusted checkpoints produced by this script. Checkpoints
    represent completed epochs; partial epochs are deliberately not resumed.
    """
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume is not None else {}
    )
    config = build_config(args, checkpoint.get("config", {}))
    start_epoch = checkpoint.get("epoch", 0)
    if args.epochs <= start_epoch:
        raise ValueError("--epochs must exceed the completed epoch count.")
    tracking_uri = (
        args.tracking_uri or checkpoint.get("tracking_uri")
        or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name = (
        args.experiment or checkpoint.get("experiment", "jamendo_transformer")
    )
    if checkpoint and (
        tracking_uri != checkpoint["tracking_uri"]
        or experiment_name != checkpoint["experiment"]
    ):
        raise ValueError("Resume must use the same MLflow server/experiment.")
    # Contact tracking before decoding audio or allocating the model on GPU.
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    run_id = checkpoint.get("run_id")
    if run_id is not None:
        run = mlflow.get_run(run_id)
        if run.info.experiment_id != experiment.experiment_id:
            raise ValueError("Checkpoint run belongs to another experiment.")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    train_loader, val_loader, fingerprints = build_loaders(config)
    genres = train_loader.dataset.genres
    if checkpoint and (
        genres != checkpoint["genres"]
        or fingerprints != checkpoint["fingerprints"]
    ):
        raise ValueError("Dataset or genre order changed since checkpoint.")
    model_config = {
        name: config[name] for name in (
            "d_model", "dim_feedforward", "num_layers", "num_heads",
        )
    }
    model_config["n_genres"] = len(genres)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = MusicGenreTransformer(**model_config)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        random.setstate(checkpoint["python_rng_state"])
        numpy_state = checkpoint["numpy_rng_state"]
        np.random.set_state((
            numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
            *numpy_state[2:],
        ))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        train_loader.generator.set_state(checkpoint["train_loader_rng_state"])
        val_loader.generator.set_state(checkpoint["val_loader_rng_state"])
    history = checkpoint.get("history", [])
    best_map = checkpoint.get("best_map", -1.0)
    # Release CPU checkpoint tensors before training, especially on resume.
    checkpoint.clear()

    with mlflow.start_run(run_id=run_id) as run:
        output_dir = PROJECT_ROOT / "data" / "experiments" / run.info.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if start_epoch and not (output_dir / "best.pt").is_file():
            raise FileNotFoundError("Resume requires the run's local best.pt.")
        mlflow.log_params({
            **config, "n_genres": len(genres), "dtype": "float32",
            "train_songs": len(train_loader.dataset),
            "validation_songs": len(val_loader.dataset),
            "optimizer": "AdamW", "loss": "BCEWithLogitsLoss",
            "sample_rate": SAMPLE_RATE, "snippet_seconds": SNIPPET_DURATION,
            "max_snippets": MAX_SNIPPETS, "f1_threshold": 0.5,
            "parameters": sum(p.numel() for p in model.parameters()),
        })
        mlflow.set_tags({
            "dataset": "MTG-Jamendo", "model_type": "MusicGenreTransformer",
            "selection_metric": "val_map_macro", "target_epochs": args.epochs,
            "device": str(device), "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "mlflow_version": mlflow.__version__,
            "sklearn_version": sklearn.__version__,
            "gpu": (torch.cuda.get_device_name(device)
                    if device.type == "cuda" else "none"),
            "limited_data": (config["max_train_songs"] is not None
                             or config["max_val_songs"] is not None),
        })
        (output_dir / "config.json").write_text(
            json.dumps({**config, "fingerprints": fingerprints}, indent=2)
            + "\n", encoding="utf-8",
        )
        (output_dir / "genres.json").write_text(
            json.dumps(genres, indent=2) + "\n", encoding="utf-8",
        )
        print(f"Run: {run.info.run_id} | Device: {device}", flush=True)
        print(f"Checkpoints: {output_dir}", flush=True)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            started = time.perf_counter()
            # First epoch includes the initial model allocation in its peaks.
            if device.type == "cuda" and epoch != start_epoch + 1:
                torch.cuda.reset_peak_memory_stats(device)
            metrics = run_epoch(model, train_loader, device, optimizer)
            metrics.update(run_epoch(model, val_loader, device))
            metrics["epoch_seconds"] = time.perf_counter() - started
            if device.type == "cuda":
                metrics["peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                )
                metrics["peak_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(device)
                )
            improved = metrics["val_map_macro"] > best_map
            best_map = max(best_map, metrics["val_map_macro"])
            history.append({"epoch": epoch, **metrics})
            numpy_state = np.random.get_state()
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config, "model_config": model_config,
                "genres": genres, "fingerprints": fingerprints,
                "epoch": epoch, "best_map": best_map, "history": history,
                "run_id": run.info.run_id, "tracking_uri": tracking_uri,
                "experiment": experiment_name,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": (
                    numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:],
                ),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (torch.cuda.get_rng_state(device)
                                   if device.type == "cuda" else None),
                "train_loader_rng_state": train_loader.generator.get_state(),
                "val_loader_rng_state": val_loader.generator.get_state(),
            }
            if improved:
                save_checkpoint(output_dir / "best.pt", state)
            save_checkpoint(output_dir / "last.pt", state)
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8",
            )
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"BCE {metrics['train_bce']:.4f}/{metrics['val_bce']:.4f} | "
                f"val mAP {metrics['val_map_macro']:.4f} | "
                f"{metrics['epoch_seconds']:.1f}s", flush=True,
            )
        for name in ("config.json", "genres.json", "history.json",
                     "best.pt", "last.pt"):
            mlflow.log_artifact(str(output_dir / name))
    return output_dir


def main() -> None:
    """Parse CLI arguments and start or resume a tracked training run."""
    train(parse_args())


if __name__ == "__main__":
    main()
