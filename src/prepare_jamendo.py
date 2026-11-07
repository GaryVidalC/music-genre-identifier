"""Prepare multilabel genre annotations for available Jamendo audio."""

import csv
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
GENRE_PREFIX = "genre---"


def prepare_jamendo(
    metadata_dir: Path,
    audio_dir: Path,
    output_dir: Path,
) -> None:
    """Save the shared genre order and available tracks for each split.

    Args:
        metadata_dir: Directory containing the official split-0 genre TSVs.
        audio_dir: Root of extracted audio, with paths like 00/12300.low.mp3.
        output_dir: Destination for genres.json and the three split JSONs.
    """
    rows_by_split = {}
    for split in SPLITS:
        metadata_path = metadata_dir / f"autotagging_genre-{split}.tsv"
        with metadata_path.open(encoding="utf-8", newline="") as source:
            reader = csv.reader(source, delimiter="\t")
            next(reader)  # Official header; tags occupy columns 5 onward.
            rows_by_split[split] = [row for row in reader if row]

    # Determine the vocabulary before excluding any unavailable audio.
    genres = sorted({
        tag.removeprefix(GENRE_PREFIX)
        for rows in rows_by_split.values()
        for row in rows
        for tag in row[5:]
        if tag.startswith(GENRE_PREFIX)
    })

    records_by_split = {}
    for split, rows in rows_by_split.items():
        records = []
        for row in rows:
            audio_path = Path(row[3]).with_suffix(".low.mp3")
            if not (audio_dir / audio_path).is_file():
                continue

            track_genres = {
                tag.removeprefix(GENRE_PREFIX) for tag in row[5:]
            }
            records.append({
                "track_id": row[0],
                "path": audio_path.as_posix(),
                "labels": [int(genre in track_genres) for genre in genres],
            })

        records_by_split[split] = records
        print(
            f"{split}: {len(records)} included, "
            f"{len(rows) - len(records)} missing audio files."
        )

    print(f"Genre vocabulary: {len(genres)} genres.")
    if not any(records_by_split.values()):
        print(
            f"No extracted audio found in {audio_dir}. "
            "Extract a TAR first. No output files were written."
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "genres.json").open("w", encoding="utf-8") as output:
        json.dump(genres, output, indent=2)
        output.write("\n")

    for split, records in records_by_split.items():
        with (output_dir / f"{split}.json").open(
            "w", encoding="utf-8",
        ) as output:
            json.dump(records, output)
            output.write("\n")

    print(f"Annotations saved to: {output_dir}")


def main() -> None:
    """Prepare annotations using paths relative to this repository."""
    project_root = Path(__file__).resolve().parent.parent
    metadata_dir = (
        project_root.parent / "mtg-jamendo-dataset"
        / "data" / "splits" / "split-0"
    )
    prepare_jamendo(
        metadata_dir=metadata_dir,
        audio_dir=project_root / "data" / "audio-low",
        output_dir=project_root / "data" / "jamendo",
    )


if __name__ == "__main__":
    main()
