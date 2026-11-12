"""Measure synthetic CNN/Transformer memory in one fresh process per case.

Inputs exclude MP3 decoding and log-mel computation. RSS is the whole
process high-water mark; CUDA peaks describe PyTorch's allocator only.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import torch
from torch import nn

from src.neural_cnn import MGI


def parse_args() -> argparse.Namespace:
    """Return the device, duration and batch size of one benchmark case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--duration-seconds", type=int, choices=(30, 180, 600), default=30,
    )
    args = parser.parse_args()
    if args.device == "cpu" and args.batch_size != 1:
        parser.error("CPU inference uses batch size 1.")
    return args


def measure(args: argparse.Namespace) -> dict:
    """Measure three GPU training steps or CPU inference; return metrics.

    Each invocation must run in its own process. No dataset is loaded.
    CUDA OOM is recorded without retrying with a smaller configuration.
    """
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no benchmark was written.")
    frames = args.duration_seconds * 44100 // 512 + 1
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "mode": "training" if cuda else "inference",
        "batch_size": args.batch_size,
        "duration_seconds": args.duration_seconds,
        "input_shape_per_song": [64, frames],
        "tokens_per_song_including_cls": (frames + 31) // 32 + 1,
        "architecture": {
            "conv_channels": 32, "temporal_pool": 8, "d_model": 256,
            "layers": 4, "heads": 8, "dim_feedforward": 1024,
            "activation": "relu", "max_positions": 4096,
        },
        "precision": "float32", "seed": 42, "cpu_threads": 4,
        "optimizer": {
            "name": "AdamW", "lr": 1e-4, "weight_decay": 0.01,
        } if cuda else None,
        "steps_requested": 3, "steps_completed": 0,
        "step_seconds": [], "status": "ok",
        "cuda_free_initial_bytes": None, "cuda_total_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
        "measurement_notes": (
            "Synthetic inputs exclude MP3 decoding and log-mel computation. "
            "RSS includes the whole process, not only model weights. "
            "CUDA peaks cover PyTorch's allocator, not all GPU memory."
        ),
    }
    if cuda:
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_free_initial_bytes=free, cuda_total_bytes=total)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        model = MGI()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["weights_float32_bytes"] = result["parameters"] * 4
        model = model.to(args.device)
        model.train(cuda)
        songs = [
            torch.rand(64, frames, device=args.device) * -80
            for _ in range(args.batch_size)
        ]
        labels = torch.randint(
            0, 2, (args.batch_size, 87), device=args.device,
        ).float() if cuda else None
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01,
        ) if cuda else None
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(3):
            if cuda:
                torch.cuda.synchronize()
            step_started = time.perf_counter()
            if cuda:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(songs), labels)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
            else:
                with torch.inference_mode():
                    model(songs)
            result["steps_completed"] += 1
            result["step_seconds"].append(time.perf_counter() - step_started)
    except (torch.cuda.OutOfMemoryError, MemoryError) as error:
        result["status"] = "out_of_memory"
        result["error"] = str(error)
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["process_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        if cuda:
            result["cuda_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            result["cuda_peak_reserved_bytes"] = (
                torch.cuda.max_memory_reserved()
            )
    return result


def main() -> None:
    """Run one case and overwrite its JSON, exiting nonzero on OOM."""
    args = parse_args()
    try:
        result = measure(args)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    output_dir = (
        Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "cnn"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{args.device}_b{args.batch_size}_{args.duration_seconds}s.json"
    )
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Results: {path}")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
