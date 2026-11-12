"""Audit Jamendo genre coverage and freeze train/validation subsets."""

import argparse
import json
from pathlib import Path

import torch


def label_counts(records: list[dict], genres: list[str]) -> list[int]:
    """Count positive songs per genre, rejecting invalid label vectors."""
    counts = [0] * len(genres)
    for record in records:
        labels = record["labels"]
        if len(labels) != len(genres) or any(v not in (0, 1) for v in labels):
            raise ValueError(f"Invalid labels: {record['track_id']}")
        for index, present in enumerate(labels):
            counts[index] += present
    return counts


def random_subset(records: list[dict], count: int, seed: int = 42) -> list:
    """Return seeded songs without replacement, matching the old sampler."""
    if count <= 0 or count > len(records):
        raise ValueError(f"Requested {count} songs from {len(records)}.")
    indices = torch.randperm(
        len(records), generator=torch.Generator().manual_seed(seed),
    )[:count].tolist()
    return [records[index] for index in indices]


def coverage_subset(
    records: list[dict], genres: list[str], count: int, seed: int = 42,
    minimum_support: int = 3,
) -> list[dict]:
    """Greedily cover training genres, then fill remaining slots randomly.

    Aim for minimum_support positive songs per available genre, subject to
    the size budget. Multi-label songs can cover several deficits. Seeded
    ordering resolves ties. This is not guaranteed to balance rare genres.
    """
    ordered = random_subset(records, len(records), seed)
    if count <= 0 or count > len(records) or minimum_support <= 0:
        raise ValueError("Invalid subset size or minimum genre support.")
    totals = label_counts(records, genres)
    targets = [min(minimum_support, value) for value in totals]
    counts = [0] * len(genres)
    remaining = [
        (record, [g for g, value in enumerate(record["labels"]) if value])
        for record in ordered
    ]
    selected = []
    while len(selected) < count and remaining:
        gains = [
            sum(counts[g] < targets[g] for g in positive)
            for _, positive in remaining
        ]
        best = max(range(len(gains)), key=gains.__getitem__)
        if gains[best] == 0:
            break
        record, positive = remaining.pop(best)
        selected.append(record)
        for genre in positive:
            counts[genre] += 1
    selected.extend(record for record, _ in remaining[:count - len(selected)])
    return selected


def coverage_summary(records: list[dict], genres: list[str]) -> dict:
    """Return song/genre counts, absent genres and genres with <3 positives."""
    counts = label_counts(records, genres)
    return {
        "songs": len(records), "genres_present": sum(v > 0 for v in counts),
        "missing_genres": [g for g, n in zip(genres, counts) if n == 0],
        "genres_under_three": [g for g, n in zip(genres, counts) if n < 3],
    }


def prepare(
    annotations_dir: Path, output_dir: Path,
    train_songs: int = 192, val_songs: int = 64,
) -> Path:
    """Write a frozen manifest and audit report without reading audio/test.

    Keep validation sampled naturally from its official partition. Improve
    training coverage only; neither move tracks nor change their labels.
    Refuse to overwrite existing artifacts so comparisons remain fixed.
    """
    genres = json.loads((annotations_dir / "genres.json").read_text())
    train = json.loads((annotations_dir / "train.json").read_text())
    validation = json.loads((annotations_dir / "validation.json").read_text())
    train_ids = [r["track_id"] for r in train]
    val_ids = [r["track_id"] for r in validation]
    if (
        len(set(train_ids)) != len(train_ids)
        or len(set(val_ids)) != len(val_ids)
        or set(train_ids) & set(val_ids)
    ):
        raise ValueError("Official partitions have duplicate/overlapping IDs.")
    original = random_subset(train, train_songs)
    selected = coverage_subset(train, genres, train_songs)
    selected_val = random_subset(validation, val_songs)
    populations = {
        "full_train": train, "full_validation": validation,
        "old_random_train": original, "selected_train": selected,
        "selected_validation": selected_val,
    }
    counts = {key: label_counts(rows, genres)
              for key, rows in populations.items()}
    report = {
        "summary": {key: coverage_summary(rows, genres)
                    for key, rows in populations.items()},
        "per_genre": {
            genre: {key: values[i] for key, values in counts.items()}
            for i, genre in enumerate(genres)
        },
        "validation_genres_absent_from_train": {
            key: [genre for i, genre in enumerate(genres)
                  if counts["selected_validation"][i] and not counts[key][i]]
            for key in ("old_random_train", "selected_train")
        },
        "notes": (
            "Training coverage is intentionally enriched; validation keeps "
            "natural random sampling, not balanced quotas. Results cover "
            "only genres represented in this fixed validation sample. "
            "No test annotations or audio were loaded."
        ),
    }
    manifest = {
        "seed": 42, "train_selection": "coverage_min3_then_random",
        "validation_selection": "random", "genres": genres,
        "records": selected, "evaluation_records": selected_val,
    }
    stem = f"train{train_songs}_val{val_songs}_seed42"
    manifest_path = output_dir / f"{stem}.json"
    report_path = output_dir / f"{stem}_report.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("Subset/report already exists; not overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in ((manifest_path, manifest), (report_path, report)):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Manifest: {manifest_path}\nReport: {report_path}")
    return manifest_path


def main() -> None:
    """Parse subset sizes and output paths, then prepare annotations only."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-songs", type=int, default=192)
    parser.add_argument("--val-songs", type=int, default=64)
    parser.add_argument(
        "--annotations-dir", type=Path, default=root / "data" / "jamendo",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "data" / "jamendo" / "subsets",
    )
    args = parser.parse_args()
    prepare(args.annotations_dir, args.output_dir,
            args.train_songs, args.val_songs)


if __name__ == "__main__":
    main()
