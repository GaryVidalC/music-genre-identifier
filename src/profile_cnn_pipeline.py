"""Time real audio preparation and a few disposable CNN training steps.

No checkpoints, MLflow calls or dataset edits. Synchronized phase timings
are diagnostic, not a throughput guarantee; OS disk caches are not cleared.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src import jamendo_dataset as data
from src.neural_cnn import MGI
from src.train_cnn_test import load_subset


def timed(
    function: Callable[[], Any], cuda: bool = False,
) -> tuple[Any, float]:
    """Return a call's value and elapsed seconds, synchronizing CUDA if set."""
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    if cuda:
        torch.cuda.synchronize()
    return value, time.perf_counter() - started


def summarize(rows: list[dict]) -> dict:
    """Aggregate measured phases, excluding the initial warmup batch."""
    measured = [row for row in rows if not row["warmup"]]
    totals = {
        key: sum(row["seconds"][key] for row in measured)
        for key in measured[0]["seconds"]
    } if measured else {}
    wall = sum(row["total_seconds"] for row in measured)
    return {
        "batches": len(measured), "total_seconds": wall,
        "phases": {
            key: {"total_seconds": value,
                  "mean_seconds_per_batch": value / len(measured),
                  "percent_of_wall": value / wall * 100 if wall else 0}
            for key, value in totals.items()
        },
    }


def profile(subset: Path, batch_size: int, batches: int) -> dict:
    """Measure one warmup and batches real batches using the current model.

    Instrument the unchanged Dataset only within this process. The model
    is randomly initialized and discarded; these steps do not assess mAP.
    """
    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_size and batches must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; profile was not started.")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    dataset, _ = load_subset(subset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42),
        collate_fn=data.collate_log_mels, num_workers=0,
        pin_memory=False, drop_last=False,
    )
    if batches + 1 > len(loader):
        raise ValueError("Not enough batches for warmup plus measurements.")
    lookup = {str(dataset.audio_dir / r["path"]): r["track_id"]
              for r in dataset.records}
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "librosa_version": data.librosa.__version__,
        "subset": str(subset.resolve()), "batch_size": batch_size,
        "batches_requested": batches, "warmup_batches": 1,
        "seed": 42, "cpu_threads": 4, "num_workers": 0,
        "precision": "float32", "learning_rate": 1e-4,
        "weight_decay": 0.01, "status": "ok", "rows": [],
        "notes": (
            "Fresh disposable model, same pipeline as trainer. Audio load "
            "includes disk read, decode, mono conversion and resampling. "
            "Duration metadata is timed separately. Warmup is excluded "
            "from aggregates. CUDA phases synchronize; disk cache is not "
            "cleared. No checkpoints or MLflow runs are created."
        ),
    }
    setup = time.perf_counter()
    model = MGI().cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    torch.cuda.synchronize()
    result["setup_seconds"] = time.perf_counter() - setup
    seconds, songs = {}, []
    original_load = data.librosa.load

    def wrap(name: str, function: Callable) -> Callable:
        """Wrap a CPU phase and accumulate time in the current batch."""
        def call(*args: Any, **kwargs: Any) -> Any:
            """Execute the original function and record its wall time."""
            value, elapsed = timed(lambda: function(*args, **kwargs))
            seconds[name] = seconds.get(name, 0) + elapsed
            return value
        return call

    def load(path: Path, **kwargs: Any) -> tuple:
        """Decode using the Dataset's crop options and record song duration."""
        value = wrap("audio_load", original_load)(path, **kwargs)
        songs.append({
            "track_id": lookup[str(path)], "offset": kwargs.get("offset", 0),
            "loaded_seconds": len(value[0]) / value[1],
        })
        return value

    try:
        with patch.object(data.librosa, "load", load), patch.object(
            data.librosa, "get_duration",
            wrap("duration_metadata", data.librosa.get_duration),
        ), patch.object(
            data, "calculate_log_mel_spectrogram",
            wrap("log_mel", data.calculate_log_mel_spectrogram),
        ):
            iterator = iter(loader)
            for batch in range(batches + 1):
                seconds, songs = {}, []
                started = time.perf_counter()
                (mels, labels), data_time = timed(lambda: next(iterator))
                seconds["other_data"] = max(
                    0, data_time - sum(seconds.values()),
                )
                (mels, labels), seconds["transfer"] = timed(
                    lambda: ([m.cuda() for m in mels], labels.cuda()), True,
                )
                _, seconds["zero_grad"] = timed(
                    lambda: optimizer.zero_grad(set_to_none=True), True,
                )
                loss, seconds["forward_loss"] = timed(
                    lambda: criterion(model(mels), labels), True,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss.")
                _, seconds["backward"] = timed(loss.backward, True)
                _, seconds["gradient_checks"] = timed(
                    lambda: check_gradients(model), True,
                )
                _, seconds["optimizer"] = timed(optimizer.step, True)
                total = time.perf_counter() - started
                seconds["other_step"] = max(0, total - sum(seconds.values()))
                result["rows"].append({
                    "batch": batch, "warmup": batch == 0, "songs": songs,
                    "seconds": seconds,
                    "total_seconds": total,
                })
                print(
                    f"Batch {batch}/{batches} | "
                    f"audio {seconds['audio_load']:.3f}s | "
                    f"mel {seconds['log_mel']:.3f}s | "
                    f"forward/backward "
                    f"{seconds['forward_loss'] + seconds['backward']:.3f}s",
                    flush=True,
                )
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="out_of_memory", error=str(error))
    result["summary"] = summarize(result["rows"])
    return result


def check_gradients(model: torch.nn.Module) -> None:
    """Perform the same finite-gradient check as the current trainer."""
    if any(not torch.isfinite(p.grad).all() for p in model.parameters()
           if p.grad is not None):
        raise RuntimeError("Non-finite gradients.")


def main() -> None:
    """Run the short profile and save a timestamped local JSON report."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument(
        "--subset", type=Path,
        default=root / "data/jamendo/subsets/train2048_val64_seed42.json",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        parser.error("--batch-size and --batches must be positive.")
    result = profile(args.subset, args.batch_size, args.batches)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"pipeline_b{args.batch_size}_{stamp}.json"
    output = root / "data/benchmarks/cnn" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Results: {output}")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
