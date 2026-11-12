"""Test CNN learning on fixed train songs, optionally using validation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import mlflow
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.jamendo_dataset import (
    MAX_LOG_MEL_DURATION, SAMPLE_RATE, JamendoDataset, collate_log_mels,
)
from src.neural_cnn import MGI
from src.train_neural import save_checkpoint, validation_metrics


def run_epoch(
    model: MGI,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    evaluation_prefix: str = "train_eval",
) -> tuple[dict, list]:
    """Train or evaluate songs; return metrics and eval probabilities.

    Evaluation disables dropout and gradients. evaluation_prefix identifies
    the evaluated split. BCE averages songs; mAP excludes absent genres.
    """
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss, count = 0.0, 0
    targets, probabilities = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for songs, labels in loader:
            songs = [song.to(device) for song in songs]
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(songs)
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
                    raise RuntimeError("Non-finite gradients.")
                optimizer.step()
            else:
                targets.append(labels.cpu())
                probabilities.append(logits.sigmoid().cpu())
            total_loss += loss.item() * labels.shape[0]
            count += labels.shape[0]
    if training:
        optimizer.zero_grad(set_to_none=True)
        return {"train_bce": total_loss / count}, []
    labels, scores = torch.cat(targets), torch.cat(probabilities)
    metrics = {
        key.replace("val_", f"{evaluation_prefix}_"): value
        for key, value in validation_metrics(labels, scores).items()
    }
    metrics.update(
        **{f"{evaluation_prefix}_bce": total_loss / count},
        positive_probability_mean=scores[labels.bool()].mean().item(),
        negative_probability_mean=scores[~labels.bool()].mean().item(),
        mean_probability_std_between_songs=(
            scores.std(dim=0, unbiased=False).mean().item()
        ),
    )
    return metrics, scores.tolist()


def select_dataset(
    split: str, count: int, random_crop: bool | None = None,
) -> JamendoDataset:
    """Select fixed songs from split, optionally overriding crop randomness."""
    dataset = JamendoDataset(
        split, representation="log_mel", random_crop=random_crop,
    )
    if count <= 0 or len(dataset) < count or len(dataset.genres) != 87:
        raise ValueError(f"Need {count} {split} songs and 87 genres.")
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(42),
    )[:count].tolist()
    dataset.records = [dataset.records[index] for index in indices]
    return dataset


def load_subset(path: Path) -> tuple[JamendoDataset, JamendoDataset]:
    """Load fixed train/validation records, verifying official membership."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    datasets = []
    for split, key in (("train", "records"),
                       ("validation", "evaluation_records")):
        dataset = JamendoDataset(split, representation="log_mel")
        official = {record["track_id"]: record for record in dataset.records}
        records = manifest[key]
        ids = [record["track_id"] for record in records]
        if (
            not records or len(set(ids)) != len(ids)
            or dataset.genres != manifest["genres"]
            or any(official.get(r["track_id"]) != r for r in records)
        ):
            raise ValueError(f"Invalid fixed subset for {split}.")
        dataset.records = records
        datasets.append(dataset)
    return datasets[0], datasets[1]


def train(
    epochs: int, batch_size: int = 2,
    train_songs: int = 16, val_songs: int = 0,
    subset: Path | None = None,
    patience: int = 10, tracking_uri: str | None = None,
    mel_cache_dir: Path | None = None,
) -> Path:
    """Open mandatory MLflow tracking, then run a fixed CNN experiment.

    Resolve tracking_uri before loading audio/model: argument, environment,
    then localhost. Validation stops after patience epochs without mAP
    improvement; memorization always uses the requested epoch count.
    mel_cache_dir enables precomputed power mels, checked before GPU loading.
    """
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size and patience must be positive.")
    if train_songs <= 0 or val_songs < 0:
        raise ValueError("train_songs must be positive; val_songs >= 0.")
    tracking_uri = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        or "http://localhost:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("jamendo_cnn_transformer")
    with mlflow.start_run() as run:
        return _train_run(
            epochs, batch_size, train_songs, val_songs, subset, patience,
            tracking_uri, run.info.run_id,
            mel_cache_dir,
        )


def _train_run(
    epochs: int, batch_size: int, train_songs: int, val_songs: int,
    subset: Path | None, patience: int, tracking_uri: str, run_id: str,
    mel_cache_dir: Path | None = None,
) -> Path:
    """Run inside an open tracking context and return the local directory.

    val_songs=0 evaluates on train for memorization; otherwise use the
    official validation partition. Small memorization tests keep log-mels
    in RAM; larger subsets load MP3 or optional cached power mels per batch.
    subset overrides song counts with a verified frozen train/val manifest.
    Save last.pt each completed epoch and best.pt on strict mAP improvement
    after baseline epoch 0. No previous run or partial epoch is resumed.
    """
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training was not started.")
    device = torch.device("cuda")
    if subset is not None:
        dataset, validation = load_subset(subset)
        train_songs, val_songs = len(dataset), len(validation)
    else:
        dataset = select_dataset("train", train_songs)
        validation = (
            select_dataset("validation", val_songs) if val_songs
            else select_dataset("train", train_songs, random_crop=False)
        )
    if dataset.genres != validation.genres:
        raise ValueError("Train and validation must share the genre order.")
    if mel_cache_dir is not None:
        dataset.enable_mel_cache(mel_cache_dir)
        validation.enable_mel_cache(mel_cache_dir)
    if val_songs and (
        {r["track_id"] for r in dataset.records}
        & {r["track_id"] for r in validation.records}
    ):
        raise ValueError("Train and validation contain overlapping tracks.")
    records = dataset.records
    prefix = "val" if val_songs else "train_eval"
    selection_metric = f"{prefix}_map_macro"
    root = Path(__file__).resolve().parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = "cnn_subset" if val_songs else "cnn_memorization"
    output = root / "data" / "experiments" / experiment / stamp
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "purpose": "Validation" if val_songs else "Memorization only",
        "epochs": epochs, "seed": 42, "batch_size": batch_size,
        "songs": train_songs, "val_songs": val_songs,
        "evaluation_split": "validation" if val_songs else "train",
        "patience": patience, "early_stopping_enabled": bool(val_songs),
        "selection_metric": selection_metric,
        "run_id": run_id, "tracking_uri": tracking_uri,
        "experiment": "jamendo_cnn_transformer",
        "subset_path": str(subset.resolve()) if subset is not None else None,
        "feature_source": "mel_power_cache_v1" if mel_cache_dir else "mp3",
        "mel_cache_dir": (
            str(mel_cache_dir.resolve()) if mel_cache_dir is not None else None
        ),
        "mel_cache_metadata": (
            {key: value for key, value in dataset.mel_cache.index.items()
             if key != "entries"} if mel_cache_dir is not None else None
        ),
        "max_audio_duration_seconds": MAX_LOG_MEL_DURATION,
        "train_crop": "random", "evaluation_crop": "central",
        "learning_rate": 1e-4, "weight_decay": 0.01,
        "optimizer": "AdamW", "loss": "BCEWithLogitsLoss",
        "precision": "float32", "conv_channels": 32, "temporal_pool": 8,
        "d_model": 256, "layers": 4, "heads": 8, "feedforward": 1024,
        "activation": "relu", "positions": "learned", "max_positions": 4096,
        "dropout": 0.1, "f1_threshold": 0.5, "n_genres": 87,
        "sample_rate": SAMPLE_RATE, "n_mels": 64,
        "n_fft": 2048, "hop_length": 512,
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "mlflow_version": mlflow.__version__,
        "sklearn_version": sklearn.__version__,
        "genres": dataset.genres, "records": records,
        "evaluation_records": validation.records,
    }
    config["subset_sha256"] = hashlib.sha256(json.dumps({
        "genres": dataset.genres, "records": records,
        "evaluation_records": validation.records,
    }, sort_keys=True).encode()).hexdigest()
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8",
    )
    mlflow.log_params({
        key: value for key, value in config.items()
        if isinstance(value, (str, int, float, bool))
    })
    mlflow.set_tags({
        "dataset": "MTG-Jamendo", "model_type": "CNNTransformer",
        "evaluation_mode": config["purpose"],
    })
    mlflow.log_artifact(str(output / "config.json"))
    print(f"MLflow run: {run_id}", flush=True)
    print(f"Results: {output}", flush=True)
    samples = dataset
    eval_samples = validation
    if not val_songs and train_songs <= 16:
        samples = []
        for index in range(len(dataset)):
            mel, labels = dataset[index]
            if (mel.shape[1] + 31) // 32 + 1 > 4096:
                track_id = records[index]["track_id"]
                raise ValueError(f"{track_id}: exceeds position capacity.")
            samples.append((mel, labels))
            print(f"Loaded {index + 1}/{train_songs}", flush=True)
        # Long-song windows must be resampled, not frozen in this cache.
        max_frames = MAX_LOG_MEL_DURATION * SAMPLE_RATE // 512 + 1
        if any(mel.shape[1] >= max_frames for mel, _ in samples):
            samples = dataset
        else:
            eval_samples = samples
    loader_options = dict(
        batch_size=batch_size, collate_fn=collate_log_mels, num_workers=0,
        pin_memory=False, drop_last=False,
    )
    training = DataLoader(
        samples, shuffle=True,
        generator=torch.Generator().manual_seed(42), **loader_options,
    )
    evaluation = DataLoader(eval_samples, shuffle=False, **loader_options)
    model = MGI().to(device)
    mlflow.log_param("parameters", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01,
    )
    history = []
    summary = {
        "run_id": run_id, "selection_metric": selection_metric,
        "best_map": None, "best_epoch": None, "epochs_without_improvement": 0,
        "last_epoch": None, "probabilities_epoch": None,
        "stop_reason": "in_progress",
    }
    try:
        for epoch in range(epochs + 1):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            metrics = {}
            if epoch:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Training {len(training)} batches", flush=True,
                )
                metrics, _ = run_epoch(model, training, device, optimizer)
            evaluated, probabilities = run_epoch(
                model, evaluation, device, evaluation_prefix=prefix,
            )
            metrics.update(evaluated)
            metrics.update(
                epoch=epoch, elapsed_seconds=time.perf_counter() - started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            improved = epoch > 0 and (
                summary["best_map"] is None
                or metrics[selection_metric] > summary["best_map"]
            )
            if improved:
                summary.update(
                    best_map=metrics[selection_metric], best_epoch=epoch,
                    epochs_without_improvement=0,
                )
            elif epoch:
                summary["epochs_without_improvement"] += 1
            stop = bool(val_songs) and (
                summary["epochs_without_improvement"] >= patience
            )
            summary.update(
                last_epoch=epoch, probabilities_epoch=epoch,
                stop_reason=("early_stopping" if stop else
                             "max_epochs" if epoch == epochs else
                             "in_progress"),
            )
            state = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": config,
                "metrics": metrics, "best_map": summary["best_map"],
                "best_epoch": summary["best_epoch"],
                "epochs_without_improvement": (
                    summary["epochs_without_improvement"]
                ),
            }
            if improved:
                save_checkpoint(output / "best.pt", state)
            save_checkpoint(output / "last.pt", state)
            history.append(metrics)
            for filename, payload in (
                ("history.json", history),
                ("probabilities.json", probabilities),
                ("summary.json", summary),
            ):
                (output / filename).write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8",
                )
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"Epoch {epoch}/{epochs} | {prefix} "
                f"({len(eval_samples)} songs) | "
                f"BCE {metrics[f'{prefix}_bce']:.4f} | "
                f"mAP {metrics[selection_metric]:.4f} | "
                f"F1 micro {metrics[f'{prefix}_f1_micro']:.4f} | "
                f"{metrics['elapsed_seconds']:.1f}s", flush=True,
            )
            if stop:
                print(
                    f"Early stopping: {patience} epochs without improvement; "
                    f"best epoch {summary['best_epoch']}.", flush=True,
                )
                break
        mlflow.set_tags({
            "stop_reason": summary["stop_reason"],
            "best_epoch": summary["best_epoch"],
        })
        for name in ("history.json", "summary.json", "probabilities.json",
                     "best.pt", "last.pt"):
            mlflow.log_artifact(str(output / name))
    except BaseException as error:
        summary["stop_reason"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        raise
    return output


def main() -> None:
    """Parse training and subset sizes; start only when invoked directly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--mel-cache-dir", type=Path)
    parser.add_argument("--train-songs", type=int)
    parser.add_argument(
        "--val-songs", type=int,
        help="Validation songs; 0 evaluates on train for memorization.",
    )
    parser.add_argument(
        "--subset", type=Path,
        help="Frozen subset manifest; replaces --train-songs/--val-songs.",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.subset is not None and (
        args.train_songs is not None or args.val_songs is not None
    ):
        parser.error("--subset cannot be combined with song counts.")
    train_songs = args.train_songs if args.train_songs is not None else 16
    val_songs = args.val_songs if args.val_songs is not None else 0
    if train_songs <= 0 or val_songs < 0:
        parser.error("--train-songs must be positive; --val-songs >= 0.")
    train(
        args.epochs, batch_size=args.batch_size,
        train_songs=train_songs, val_songs=val_songs, subset=args.subset,
        patience=args.patience, tracking_uri=args.tracking_uri,
        mel_cache_dir=args.mel_cache_dir,
    )


if __name__ == "__main__":
    main()
