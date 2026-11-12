"""Load Jamendo patches or log-mels with a 20-minute audio window limit."""

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.cache_jamendo_mels import MelCache
from src.neural import (
    calculate_log_mel_spectrogram,
    extract_patches,
    snippets,
)


SAMPLE_RATE = 44100
SNIPPET_DURATION = 15
MAX_SNIPPETS = 40
MAX_LOG_MEL_DURATION = 20 * 60
logger = logging.getLogger(__name__)


class JamendoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read annotations eagerly, but decode and process audio on demand."""

    def __init__(
        self,
        split: str,
        annotations_dir: Path | None = None,
        audio_dir: Path | None = None,
        representation: str = "patches",
        random_crop: bool | None = None,
        mel_cache_dir: Path | None = None,
    ) -> None:
        """Load one split and its shared genre order without loading audio.

        Args:
            split: Official partition: train, validation or test.
            annotations_dir: JSON directory; defaults to data/jamendo.
            audio_dir: Extracted MP3 root; defaults to data/audio-low.
                Default paths are relative to this repository, not the cwd.
            representation: Snippet patches (default) or complete log_mel.
                Log-mels use at most 20 minutes of audio per reading.
            random_crop: Random long-song windows; defaults to True only
                for train. False uses a fixed central window. Patches
                retain their existing behavior and ignore this setting.
            mel_cache_dir: Optional power-mel cache for log_mel only.
                Validate coverage of the current partition before use.
        """
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test.")
        if representation not in ("patches", "log_mel"):
            raise ValueError("representation must be patches or log_mel.")
        self.representation = representation
        self.mel_cache: MelCache | None = None
        self.random_crop = (
            split == "train" if random_crop is None else random_crop
        )

        project_root = Path(__file__).resolve().parent.parent
        annotations_dir = (
            Path(annotations_dir) if annotations_dir is not None
            else project_root / "data" / "jamendo"
        )
        self.audio_dir = (
            Path(audio_dir) if audio_dir is not None
            else project_root / "data" / "audio-low"
        )
        with (annotations_dir / "genres.json").open(encoding="utf-8") as file:
            self.genres = json.load(file)
        with (annotations_dir / f"{split}.json").open(
            encoding="utf-8",
        ) as file:
            self.records = json.load(file)
        if mel_cache_dir is not None:
            self.enable_mel_cache(mel_cache_dir)

    def enable_mel_cache(self, directory: Path) -> None:
        """Validate coverage, sources and values before enabling the cache."""
        if self.representation != "log_mel":
            raise ValueError("Mel cache is only supported for log_mel.")
        cache = MelCache(directory, self.audio_dir)
        for record in self.records:
            cache.open(record, full_check=True)
        self.mel_cache = cache

    def cached_log_mel(self, record: dict) -> torch.Tensor:
        """Select cached frames and convert this window to peak-relative dB.

        Crop starts are frame-aligned, so edges may differ slightly from
        extracting a mel after cropping waveform samples. Sources are not
        decoded; only their size/mtime is checked for cache invalidation.
        """
        if self.mel_cache is None:
            raise RuntimeError("Mel cache is not enabled.")
        mel = self.mel_cache.open(record)
        max_frames = MAX_LOG_MEL_DURATION * SAMPLE_RATE // 512 + 1
        available = max(0, mel.shape[1] - max_frames)
        start = 0
        if available:
            start = (
                int(torch.randint(available + 1, ()).item())
                if self.random_crop else available // 2
            )
            logger.warning(
                "%s: cached %s window, frames %d–%d of %d.",
                record["track_id"],
                "random" if self.random_crop else "central",
                start, start + max_frames, mel.shape[1],
            )
        window = mel[:, start:start + max_frames]
        if not np.isfinite(window).all() or (window < 0).any():
            raise ValueError(f"{record['track_id']}: invalid cached mel.")
        log_mel = librosa.power_to_db(
            window, ref=np.max, amin=1e-10, top_db=80,
        )
        return torch.from_numpy(
            np.array(log_mel, dtype=np.float32, copy=True),
        )

    def __len__(self) -> int:
        """Return the number of songs in this partition."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU float32 features and labels [G] for one song/window.

        Features are patches [S,320,16,16] or log-mel [64,T]. Only patches
        use the snippet limit. Long log-mel inputs read one 20-minute
        window before feature extraction; original files stay untouched.

        Args:
            index: Index of the song in the selected partition.

        Raises:
            FileNotFoundError: If this song's audio is missing.
            RuntimeError: If its audio cannot be decoded.
            ValueError: If labels, decoded samples or audio length are invalid.
        """
        record = self.records[index]
        track_id = record["track_id"]
        audio_path = self.audio_dir / record["path"]
        if len(record["labels"]) != len(self.genres):
            raise ValueError(
                f"{track_id}: labels do not match the genre list."
            )
        if self.mel_cache is not None:
            return (
                self.cached_log_mel(record),
                torch.tensor(record["labels"], dtype=torch.float32),
            )
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"{track_id}: audio not found: {audio_path}"
            )

        try:
            load_options = {"sr": SAMPLE_RATE, "mono": True}
            if self.representation == "log_mel":
                duration = librosa.get_duration(path=audio_path)
                if duration > MAX_LOG_MEL_DURATION:
                    available_offset = duration - MAX_LOG_MEL_DURATION
                    offset = available_offset * (
                        torch.rand(()).item() if self.random_crop else 0.5
                    )
                    load_options.update(
                        offset=offset, duration=MAX_LOG_MEL_DURATION,
                    )
                    logger.warning(
                        "%s: using %.2f–%.2f s of %.2f s (%s window).",
                        track_id, offset, offset + MAX_LOG_MEL_DURATION,
                        duration, "random" if self.random_crop else "central",
                    )
            audio, sample_rate = librosa.load(audio_path, **load_options)
        except Exception as error:
            raise RuntimeError(
                f"{track_id}: could not decode audio: {audio_path}"
            ) from error

        labels = torch.tensor(record["labels"], dtype=torch.float32)
        if self.representation == "log_mel":
            if (
                not len(audio)
                or not torch.isfinite(torch.from_numpy(audio)).all()
            ):
                raise ValueError(f"{track_id}: audio is empty or non-finite.")
            log_mel = torch.from_numpy(
                calculate_log_mel_spectrogram(audio, sample_rate),
            ).float()
            if not torch.isfinite(log_mel).all():
                raise ValueError(f"{track_id}: log-mel is non-finite.")
            return log_mel, labels

        audio_snippets = snippets(
            audio, sample_rate,
            snippet_duration=SNIPPET_DURATION,
            max_snippets=MAX_SNIPPETS,
        )
        if not audio_snippets:
            raise ValueError(
                f"{track_id}: audio must contain a complete 15-second snippet."
            )

        log_mels = torch.stack([
            torch.from_numpy(
                calculate_log_mel_spectrogram(snippet, sample_rate),
            ).float()
            for snippet in audio_snippets
        ])
        patches = extract_patches(log_mels, patch_size=16)
        return patches, labels


def collate_jamendo(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad samples along snippets and return patches, labels and mask.

    Args:
        samples: Songs with patches [S, P, 16, 16] and labels [G].

    Returns:
        Patches [B, S_max, P, 16, 16], labels [B, G] and a boolean mask
        [B, S_max], where True identifies right-padding, not real snippets.
    """
    if not samples or any(patches.shape[0] == 0 for patches, _ in samples):
        raise ValueError(
            "Provide a nonempty batch with snippets in each song."
        )

    patches, labels = zip(*samples)
    lengths = torch.tensor([song.shape[0] for song in patches])
    padded_patches = pad_sequence(patches, batch_first=True, padding_value=0)
    positions = torch.arange(padded_patches.shape[1])
    padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    return padded_patches, torch.stack(labels), padding_mask


def collate_log_mels(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Keep complete log-mels unpadded; stack multilabel targets as [B,G]."""
    if not samples or any(
        mel.ndim != 2 or mel.shape[0] != 64 or mel.shape[1] == 0
        for mel, _ in samples
    ):
        raise ValueError("Provide a nonempty batch of log-mels [64,T].")
    log_mels, labels = zip(*samples)
    return list(log_mels), torch.stack(labels)


def create_jamendo_dataloader(
    split: str,
    batch_size: int = 2,
    annotations_dir: Path | None = None,
    audio_dir: Path | None = None,
    seed: int = 42,
    representation: str = "patches",
    mel_cache_dir: Path | None = None,
) -> DataLoader:
    """Create a padded song loader, shuffling only the training partition.

    Args:
        split: Official partition: train, validation or test.
        batch_size: Number of songs per batch; the final batch is retained.
        annotations_dir: Optional JSON directory passed to JamendoDataset.
        audio_dir: Optional MP3 root passed to JamendoDataset.
        seed: Seed for the loader's independent random generator.
        representation: patches (default) or log_mel features capped at
            20 minutes; random train windows and central evaluation windows.
        mel_cache_dir: Optional prevalidated power-mel cache for log_mel.

    Returns:
        A CPU DataLoader yielding patches/labels/mask by default, or an
        unpadded log-mel list and labels when representation is log_mel.
        Genre order remains available through loader.dataset.genres.
    """
    dataset = JamendoDataset(
        split, annotations_dir, audio_dir, representation=representation,
        mel_cache_dir=mel_cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=(collate_jamendo if representation == "patches"
                    else collate_log_mels),
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
