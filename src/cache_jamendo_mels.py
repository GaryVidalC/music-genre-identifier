"""Build and read a resumable float32 power-mel cache, without MLflow/GPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import librosa
import numpy as np


PARAMETERS = {
    "sr": 44100, "n_fft": 2048, "hop_length": 512, "n_mels": 64,
    "win_length": 2048, "window": "hann", "center": True,
    "pad_mode": "constant", "power": 2.0, "fmin": 0.0, "fmax": None,
    "htk": False, "norm": "slaney",
}
FORMAT_VERSION = "mel_power_v1"
AUDIO_DECODE = {"mono": True, "res_type": "soxr_hq"}


def write_json(path: Path, payload: dict) -> None:
    """Atomically replace a cache metadata file with JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def relative_path(record: dict) -> Path:
    """Return a safe relative audio path from an annotation record."""
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid audio path: {relative}")
    return relative


def source_info(path: Path) -> dict:
    """Return size and modification time used to detect changed inputs."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_values(array: np.ndarray) -> None:
    """Check finite nonnegative power values using bounded temporary arrays."""
    for start in range(0, array.shape[1], 8192):
        block = array[:, start:start + 8192]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("Power mel contains non-finite/negative values.")


class MelCache:
    """Validate and memory-map cached power mels without decoding audio."""

    def __init__(self, directory: Path, audio_dir: Path) -> None:
        """Read the index and reject incompatible formats or parameters."""
        self.directory = Path(directory)
        self.audio_dir = Path(audio_dir)
        self.index = json.loads(
            (self.directory / "index.json").read_text(encoding="utf-8"),
        )
        if (
            self.index.get("format") != FORMAT_VERSION
            or self.index.get("parameters") != PARAMETERS
            or self.index.get("dtype") != "float32"
            or self.index.get("audio_decode") != AUDIO_DECODE
        ):
            raise ValueError("Incompatible mel cache format/parameters/dtype.")

    def open(self, record: dict, full_check: bool = False) -> np.memmap:
        """Validate cached mel and return a read-only memory map."""
        track_id = record["track_id"]
        try:
            entry = self.index["entries"][track_id]
            relative = relative_path(record)
            if (
                entry["path"] != relative.as_posix()
                or entry["source"] != source_info(self.audio_dir / relative)
            ):
                raise ValueError("Audio source changed or path mismatches.")
            path = self.directory / relative.with_suffix(".npy")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32 or array.ndim != 2
                or array.shape[0] != 64 or array.shape[1] == 0
                or list(array.shape) != entry["shape"]
                or array.shape[1] != entry["samples"] // 512 + 1
                or path.stat().st_size != entry["npy_bytes"]
            ):
                raise ValueError("Invalid cached mel shape/dtype/size.")
            if full_check:
                validate_values(array)
            return array
        except (KeyError, OSError, ValueError, EOFError, TypeError) as error:
            raise ValueError(
                f"{track_id}: invalid/missing mel cache: {error}"
            ) from error


def manifest_records(subset: Path, annotations_dir: Path) -> list[dict]:
    """Return frozen train/validation records after membership checks."""
    manifest = json.loads(subset.read_text(encoding="utf-8"))
    genres = json.loads((annotations_dir / "genres.json").read_text())
    if manifest["genres"] != genres:
        raise ValueError("Subset genre order does not match annotations.")
    result = []
    for split, key in (("train", "records"),
                       ("validation", "evaluation_records")):
        official = {
            r["track_id"]: r for r in json.loads(
                (annotations_dir / f"{split}.json").read_text(),
            )
        }
        records = manifest[key]
        if not records or any(
            official.get(r["track_id"]) != r for r in records
        ):
            raise ValueError(f"Invalid official {split} subset records.")
        result.extend(records)
    if len({r["track_id"] for r in result}) != len(result):
        raise ValueError("Duplicate/overlapping train and validation tracks.")
    return result


def build_cache(
    subset: Path, output_dir: Path, audio_dir: Path, annotations_dir: Path,
) -> dict:
    """Generate one complete power mel per song, resume and verify all files.

    Preserve audio, labels and completed compatible cache files. Fail with
    the offending track on corruption or source changes; no silent fallback.
    A file written before an interrupted index update can be regenerated.
    """
    records = manifest_records(subset, annotations_dir)
    fingerprint = hashlib.sha256(subset.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    if index_path.exists():
        cache = MelCache(output_dir, audio_dir)
        if cache.index["subset_sha256"] != fingerprint:
            raise ValueError("Cache belongs to another subset manifest.")
        if (
            cache.index["librosa_version"] != librosa.__version__
            or cache.index["numpy_version"] != str(np.__version__)
        ):
            raise ValueError("Cannot resume with different library versions.")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "Refuse a nonempty unindexed cache directory."
            )
        index = {
            "format": FORMAT_VERSION, "parameters": PARAMETERS,
            "dtype": "float32", "numpy_version": str(np.__version__),
            "librosa_version": librosa.__version__,
            "subset_sha256": fingerprint, "entries": {}, "complete": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "audio_decode": AUDIO_DECODE,
        }
        write_json(index_path, index)
        cache = MelCache(output_dir, audio_dir)
    started = time.perf_counter()
    reused = 0
    for number, record in enumerate(records, 1):
        track_id = record["track_id"]
        try:
            if track_id in cache.index["entries"]:
                cache.open(record, full_check=True)
                reused += 1
            else:
                relative = relative_path(record)
                audio_path = audio_dir / relative
                source = source_info(audio_path)
                audio, _ = librosa.load(
                    audio_path, sr=44100, mono=True, dtype=np.float32,
                    res_type="soxr_hq",
                )
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError("Audio is empty or non-finite.")
                samples = len(audio)
                mel = librosa.feature.melspectrogram(y=audio, **PARAMETERS)
                mel = np.asarray(mel, dtype=np.float32)
                validate_values(mel)
                if source_info(audio_path) != source:
                    raise ValueError("Audio changed during extraction.")
                path = output_dir / relative.with_suffix(".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("wb") as file:
                    np.save(file, mel, allow_pickle=False)
                temporary.replace(path)
                cache.index["entries"][track_id] = {
                    "path": relative.as_posix(), "source": source,
                    "samples": samples, "shape": list(mel.shape),
                    "npy_bytes": path.stat().st_size,
                }
                cache.index["complete"] = False
                write_json(index_path, cache.index)
                del audio, mel
            elapsed = time.perf_counter() - started
            print(
                f"{number}/{len(records)} | {track_id} | "
                f"reused {reused} | elapsed {elapsed:.1f}s", flush=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"{track_id}: cache generation failed: {error}"
            ) from error
    print("Verifying every cached file...", flush=True)
    for record in records:
        cache.open(record, full_check=True)
    cache.index.update(
        complete=True, tracks=len(records),
        total_npy_bytes=sum(
            e["npy_bytes"] for e in cache.index["entries"].values()
        ),
    )
    write_json(index_path, cache.index)
    print(f"Verified {len(records)} tracks: {output_dir}", flush=True)
    return cache.index


def main() -> None:
    """Parse paths and build only the requested manifest's power-mel cache."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/jamendo/mel_cache")
    parser.add_argument("--audio-dir", type=Path,
                        default=root / "data/audio-low")
    parser.add_argument("--annotations-dir", type=Path,
                        default=root / "data/jamendo")
    args = parser.parse_args()
    build_cache(
        args.subset, args.output_dir, args.audio_dir, args.annotations_dir,
    )


if __name__ == "__main__":
    main()
