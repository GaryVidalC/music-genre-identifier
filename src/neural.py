from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn


def snippets(
    audio: np.ndarray,
    sr: int,
    snippet_duration: int = 15,
    max_snippets: int = 40,
) -> list[np.ndarray]:
    """Split mono audio into complete snippets, discarding the final tail.

    Args:
        audio: One-dimensional audio samples.
        sr: Sample rate in Hz.
        snippet_duration: Duration of each snippet in seconds.
        max_snippets: Maximum number of snippets to retain from the start.

    Returns:
        A list of complete audio snippets.
    """
    snippet_length = snippet_duration * sr
    num_snippets = min(len(audio) // snippet_length, max_snippets)

    snippets_list = []
    for i in range(num_snippets):
        start_idx = i * snippet_length
        end_idx = start_idx + snippet_length
        snippets_list.append(audio[start_idx:end_idx])

    return snippets_list


def calculate_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 64,
) -> np.ndarray:
    """Calculate a log-mel spectrogram relative to its maximum power.

    Args:
        audio: One-dimensional audio samples.
        sr: Sample rate in Hz.
        n_fft: Samples per FFT window.
        hop_length: Samples between successive frames.
        n_mels: Number of mel frequency bands.

    Returns:
        A NumPy array shaped (mel bands, frames), in decibels.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def extract_patches(
    log_mel: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """Split one or several log-mels into non-overlapping square patches.

    Args:
        log_mel: Tensor shaped (bands, frames) or (snippets, bands, frames).
        patch_size: Positive integer giving each patch's height and width.

    Returns:
        Tensor shaped (patches, patch_size, patch_size), optionally preceded
        by the snippets axis. Preserves dtype and device; time varies
        fastest and incomplete borders are discarded.

    Raises:
        ValueError: If the input is not 2D/3D, patch_size is invalid, or no
            complete patch fits.
    """
    if log_mel.ndim not in (2, 3):
        raise ValueError("log_mel must be a 2D or 3D tensor.")
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, (int, np.integer))
        or patch_size <= 0
    ):
        raise ValueError("patch_size must be a positive integer.")

    n_bands, n_frames = log_mel.shape[-2:]
    if n_bands < patch_size or n_frames < patch_size:
        raise ValueError("log_mel must contain at least one complete patch.")

    # Unfold appends the window axis, leaving time at dimension -2.
    patches = log_mel.unfold(-2, patch_size, patch_size)
    patches = patches.unfold(-2, patch_size, patch_size)
    return patches.flatten(start_dim=-4, end_dim=-3)


class PatchEncoder(nn.Module):
    """Project patches and contextualize them within each snippet."""

    def __init__(
        self,
        d_model: int,
        max_patches: int,
        dim_feedforward: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        """Set patch capacity, embedding/FFN widths, depth and head count."""
        super().__init__()
        self.projection = nn.Linear(16 * 16, d_model)
        self.positions = nn.Embedding(max_patches, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=0.1,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Map patches [S, P, 16, 16] to contextual embeddings [S, P, D]."""
        patch_embeddings = self.projection(patches.flatten(start_dim=-2))
        positions = torch.arange(patches.shape[1], device=patches.device)
        patch_embeddings = patch_embeddings + self.positions(positions)
        return self.transformer(patch_embeddings)


class PatchAttentionPooling(nn.Module):
    """Summarize each snippet's patches using one learned query."""

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        """Build patch cross-attention with the given width and head count."""
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool patch embeddings [S, P, D] into snippet embeddings [S, D]."""
        queries = self.query.expand(patch_embeddings.shape[0], -1, -1)
        snippet_embeddings, _ = self.attention(
            query=queries,
            key=patch_embeddings,
            value=patch_embeddings,
            need_weights=False,
        )
        return snippet_embeddings.squeeze(1)


class SnippetEncoder(nn.Module):
    """Add snippet positions and model relationships within each song."""

    def __init__(
        self,
        d_model: int,
        max_snippets: int,
        dim_feedforward: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        """Set snippet capacity, embedding/FFN widths, depth and head count."""
        super().__init__()
        self.positions = nn.Embedding(max_snippets, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=0.1,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

    def forward(
        self,
        snippet_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contextualize embeddings [B, S, D], ignoring masked snippet keys.

        padding_mask is boolean [B, S]; True marks padding. Returns [B, S, D].
        """
        positions = torch.arange(
            snippet_embeddings.shape[1], device=snippet_embeddings.device,
        )
        snippet_embeddings = snippet_embeddings + self.positions(positions)
        return self.transformer(
            snippet_embeddings, src_key_padding_mask=padding_mask,
        )


class GenreAttentionHead(nn.Module):
    """Query all snippets per genre and produce independent genre logits."""

    def __init__(
        self, d_model: int, n_genres: int, num_heads: int = 4,
    ) -> None:
        """Build genre queries/readouts with the given width and head count."""
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_genres, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False,
        )
        self.classifier = nn.Linear(d_model, n_genres)

    def forward(
        self,
        snippet_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read embeddings [B, S, D] and return genre logits [B, G].

        padding_mask is boolean [B, S]; True excludes a snippet from attention.
        """
        # Preserve sequence-first attention and the existing query weights.
        snippet_sequence = snippet_embeddings.transpose(0, 1)  # [S, B, D]
        queries = self.queries.expand(-1, snippet_embeddings.shape[0], -1)
        genre_embeddings, _ = self.attention(
            query=queries,
            key=snippet_sequence,
            value=snippet_sequence,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        genre_embeddings = genre_embeddings.transpose(0, 1)  # [B, G, D]

        # Each genre uses only its matching classifier row, not all G rows.
        logits = (genre_embeddings * self.classifier.weight).sum(dim=-1)
        return logits + self.classifier.bias


class MusicGenreTransformer(nn.Module):
    """Classify songs through patch and masked snippet attention."""

    def __init__(
        self,
        n_genres: int,
        d_model: int = 256,
        patches: int = 320,
        max_snippets: int = 40,
        dim_feedforward: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        """Build the existing architecture with configurable capacities.

        Args:
            n_genres: Number of independent genre outputs.
            d_model: Embedding width, divisible by num_heads.
            patches: Maximum number of patch positions per snippet.
            max_snippets: Maximum number of snippet positions per song.
            dim_feedforward: Hidden width of both Transformers' feedforward.
            num_layers: Number of layers in each of the two encoders.
            num_heads: Head count for all self- and cross-attention modules.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("num_heads must be positive and divide d_model.")
        self.d_model = d_model
        self.n_genres = n_genres
        self.patch_encoder = PatchEncoder(
            d_model, patches, dim_feedforward=dim_feedforward,
            num_layers=num_layers, num_heads=num_heads,
        )
        self.patch_pooling = PatchAttentionPooling(d_model, num_heads)
        self.snippet_encoder = SnippetEncoder(
            d_model, max_snippets, dim_feedforward=dim_feedforward,
            num_layers=num_layers, num_heads=num_heads,
        )
        self.genre_head = GenreAttentionHead(d_model, n_genres, num_heads)

    def forward(
        self,
        patches: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map batched patches [B, S, P, 16, 16] to raw logits [B, G].

        Also accepts a single song [S, P, 16, 16], returning logits [G].
        padding_mask must be boolean [B, S] on the input device, where True
        marks padding. If omitted, all snippets are real. Each song must
        contain at least one valid snippet. Inputs share the model's dtype
        and device. Sigmoid is left to inference, not applied here.
        """
        if patches.ndim not in (4, 5) or patches.shape[-2:] != (16, 16):
            raise ValueError(
                "patches must have shape [S, P, 16, 16] or [B, S, P, 16, 16]."
            )
        single_song = patches.ndim == 4
        if single_song:
            patches = patches.unsqueeze(0)

        batch_size, snippet_count, patch_count = patches.shape[:3]
        if (
            batch_size == 0
            or not 0 < snippet_count
            <= self.snippet_encoder.positions.num_embeddings
            or not 0 < patch_count
            <= self.patch_encoder.positions.num_embeddings
        ):
            raise ValueError(
                "Provide nonempty songs within snippet and patch capacities."
            )

        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, snippet_count),
                dtype=torch.bool, device=patches.device,
            )
        if (
            padding_mask.shape != (batch_size, snippet_count)
            or padding_mask.dtype != torch.bool
            or padding_mask.device != patches.device
        ):
            raise ValueError(
                "padding_mask must be boolean [B, S] on the patches device."
            )
        if padding_mask.all(dim=1).any():
            raise ValueError(
                "Each song must contain at least one real snippet."
            )

        # Encode only real snippets: padding never reaches patch attention.
        valid_snippets = ~padding_mask
        patch_embeddings = self.patch_encoder(patches[valid_snippets])
        pooled_snippets = self.patch_pooling(patch_embeddings)  # [valid_S, D]
        snippet_embeddings = pooled_snippets.new_zeros(
            batch_size, snippet_count, self.d_model,
        )
        snippet_embeddings[valid_snippets] = pooled_snippets
        snippet_embeddings = self.snippet_encoder(
            snippet_embeddings, padding_mask,
        )
        logits = self.genre_head(snippet_embeddings, padding_mask)  # [B, G]
        return logits.squeeze(0) if single_song else logits


# Preserve the previous constructor name for existing imports and examples.
NN_model = MusicGenreTransformer


def main() -> None:
    """Load the local sample audio and print preprocessing/output shapes."""
    root = Path(__file__).parent.parent
    sample_path = (
        root / "data" / "jamendo_sample" / "raw_30s_audio-low-00"
        / "00" / "4400.low.mp3"
    )
    sample_rate = 44100
    max_snippets = 40
    audio, sample_rate = librosa.load(
        sample_path, sr=sample_rate, mono=True,
    )
    audio_snippets = snippets(audio, sample_rate, max_snippets=max_snippets)
    if not audio_snippets:
        raise ValueError("Audio must contain at least one complete snippet.")

    log_mels = torch.stack([
        torch.from_numpy(
            calculate_log_mel_spectrogram(snippet, sample_rate),
        ).float()
        for snippet in audio_snippets
    ])
    patches = extract_patches(log_mels, patch_size=16)
    model = MusicGenreTransformer(
        n_genres=10, max_snippets=max_snippets,
    ).eval()
    with torch.inference_mode():
        logits = model(patches)

    print(f"log_mel shape: {log_mels.shape}")
    print(f"patches shape: {patches.shape}")
    print(f"logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
