"""Measure peak CUDA memory for three synthetic training steps."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from src.neural import MusicGenreTransformer


def measure_vram(
    d_model: int,
    dim_feedforward: int,
    num_layers: int = 2,
    num_heads: int = 4,
    batch_size: int = 1,
) -> dict:
    """Return CUDA memory peaks and configuration for one training trial.

    Args:
        d_model: Model embedding width, divisible by num_heads.
        dim_feedforward: Hidden width of both Transformers' feedforward.
        num_layers: Number of layers in each encoder.
        num_heads: Head count for all attention modules.
        batch_size: Number of synthetic songs, each with 40 real snippets.

    Returns:
        JSON-compatible results, including partial measurements on CUDA OOM.
        Requires CUDA; call once per process to isolate configurations.
    """
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    optimizer_config = {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "foreach": None,
        "fused": None,
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "d_model": d_model,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "n_genres": 87,
        "input_shape": [batch_size, 40, 320, 16, 16],
        "dtype": "float32",
        "seed": 42,
        "loss": "BCEWithLogitsLoss",
        "optimizer": {"name": "AdamW", **optimizer_config},
        "steps_requested": 3,
        "steps_completed": 0,
        "initial_free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "status": "ok",
    }
    model = MusicGenreTransformer(
        n_genres=87,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    result["parameters"] = sum(p.numel() for p in model.parameters())

    try:
        model = model.to(device=device, dtype=torch.float32).train()
        inputs = torch.randn(
            result["input_shape"], device=device, dtype=torch.float32,
        )
        labels = torch.randint(
            0, 2, (batch_size, 87), device=device, dtype=torch.float32,
        )
        padding_mask = torch.zeros(
            (batch_size, 40), device=device, dtype=torch.bool,
        )
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(result["steps_requested"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, padding_mask)
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("The trial produced non-finite outputs.")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            result["steps_completed"] += 1
            del logits, loss
    except torch.cuda.OutOfMemoryError:
        result["status"] = "out_of_memory"

    torch.cuda.synchronize(device)
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result


def main() -> int:
    """Parse dimensions, run one CUDA trial, print and save its result.

    Returns:
        Zero on success; one for unavailable CUDA or an out-of-memory trial.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.num_heads <= 0:
        parser.error("--num-heads must be positive.")
    if args.d_model <= 0 or args.d_model % args.num_heads:
        parser.error(
            "--d-model must be positive and divisible by --num-heads."
        )
    if args.dim_feedforward <= 0:
        parser.error("--dim-feedforward must be positive.")
    if args.num_layers <= 0:
        parser.error("--num-layers must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if not torch.cuda.is_available():
        print("CUDA is unavailable. No benchmark was run or saved.")
        return 1

    result = measure_vram(
        args.d_model, args.dim_feedforward,
        num_layers=args.num_layers, num_heads=args.num_heads,
        batch_size=args.batch_size,
    )
    output_dir = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"vram_d{args.d_model}_ff{args.dim_feedforward}"
        f"_l{args.num_layers}_h{args.num_heads}_b{args.batch_size}.json"
    )
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")

    print(json.dumps(result, indent=2))
    print(
        f"Peak allocated: {result['peak_allocated_bytes'] / 1024**2:.2f} MiB; "
        f"peak reserved: {result['peak_reserved_bytes'] / 1024**2:.2f} MiB."
    )
    print("These peaks only include memory managed by PyTorch's allocator.")
    print(f"Saved to: {output_path}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
