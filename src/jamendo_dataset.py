"""Load one Jamendo song at a time as patches and multilabel targets."""

import json
from pathlib import Path

import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.neural import (
    calculate_log_mel_spectrogram,
    extract_patches,
    snippets,
)


SAMPLE_RATE = 44100
SNIPPET_DURATION = 15
MAX_SNIPPETS = 40


class JamendoDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read annotations eagerly, but decode and process audio on demand."""

    def __init__(
        self,
        split: str,
        annotations_dir: Path | None = None,
        audio_dir: Path | None = None,
    ) -> None:
        """Load one split and its shared genre order without loading audio.

        Args:
            split: Official partition: train, validation or test.
            annotations_dir: JSON directory; defaults to data/jamendo.
            audio_dir: Extracted MP3 root; defaults to data/audio-low.
                Default paths are relative to this repository, not the cwd.
        """
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test.")

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

    def __len__(self) -> int:
        """Return the number of songs in this partition."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU float32 patches [S, 320, 16, 16] and labels [G].

        Args:
            index: Index of the song in the selected partition.

        Raises:
            FileNotFoundError: If this song's audio is missing.
            RuntimeError: If its audio cannot be decoded.
            ValueError: If labels have the wrong length or audio is too short.
        """
        record = self.records[index]
        track_id = record["track_id"]
        audio_path = self.audio_dir / record["path"]
        if len(record["labels"]) != len(self.genres):
            raise ValueError(
                f"{track_id}: labels do not match the genre list."
            )
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"{track_id}: audio not found: {audio_path}"
            )

        try:
            audio, sample_rate = librosa.load(
                audio_path, sr=SAMPLE_RATE, mono=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"{track_id}: could not decode audio: {audio_path}"
            ) from error

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
        labels = torch.tensor(
            record["labels"], dtype=torch.float32, device="cpu",
        )
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


def create_jamendo_dataloader(
    split: str,
    batch_size: int = 2,
    annotations_dir: Path | None = None,
    audio_dir: Path | None = None,
    seed: int = 42,
) -> DataLoader:
    """Create a padded song loader, shuffling only the training partition.

    Args:
        split: Official partition: train, validation or test.
        batch_size: Number of songs per batch; the final batch is retained.
        annotations_dir: Optional JSON directory passed to JamendoDataset.
        audio_dir: Optional MP3 root passed to JamendoDataset.
        seed: Seed for the loader's independent random generator.

    Returns:
        A CPU DataLoader yielding patches, labels and the padding mask.
        Genre order remains available through loader.dataset.genres.
    """
    dataset = JamendoDataset(split, annotations_dir, audio_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=collate_jamendo,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
