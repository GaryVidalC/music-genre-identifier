from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def calculate_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 44100,
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


class AudioCNN(nn.Module):
    """Extract local features from one complete, unpadded log-mel."""

    def __init__(self, mel_bins: int = 64, conv_channels: int = 32) -> None:
        """Build three convolution blocks and an eightfold temporal pool."""
        super().__init__()
        if mel_bins <= 0 or mel_bins % 8 or conv_channels <= 0:
            raise ValueError("mel_bins must be divisible by 8; channels > 0.")
        self.mel_bins = mel_bins
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.temporal_pool = nn.AvgPool2d(
            kernel_size=(1, 8),
            stride=(1, 8),
            ceil_mode=True,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a log-mel [F,T], preserving the partial final time window.

        Returns features [1,C,F/8,ceil(T/32)] when F is divisible by eight.
        """
        if x.ndim != 2 or x.shape[0] != self.mel_bins or x.shape[1] == 0:
            raise ValueError(
                f"Expected a nonempty log-mel [{self.mel_bins},T]."
            )
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.temporal_pool(x)


class TemporalEncoder(nn.Module):
    """Contextualize padded song tokens using learned positions and CLS."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 1024,
        num_layers: int = 4, max_positions: int = 4096,
    ) -> None:
        """Set embedding/FFN dimensions, depth and capacity including CLS."""
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.position_embedding = nn.Embedding(max_positions, input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=8, dim_feedforward=hidden_dim,
                activation="relu", dropout=0.1, batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map tokens [B,L,D] to [B,L+1,D], ignoring True padding entries.

        padding_mask must be boolean [B,L] on the input device. CLS is
        prepended internally and never masked. Every song needs real tokens.
        """
        if x.ndim != 3 or x.shape[2] != self.cls_token.shape[2]:
            raise ValueError("Tokens must have shape [B,L,input_dim].")
        batch_size, length = x.shape[:2]
        if batch_size == 0 or length == 0:
            raise ValueError("Provide a nonempty batch and token sequence.")
        if length + 1 > self.position_embedding.num_embeddings:
            raise ValueError("Sequence including CLS exceeds capacity.")
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, length), dtype=torch.bool, device=x.device,
            )
        if (
            padding_mask.shape != (batch_size, length)
            or padding_mask.dtype != torch.bool
            or padding_mask.device != x.device
        ):
            raise ValueError("padding_mask must be boolean [B,L] on device.")
        if padding_mask.all(dim=1).any():
            raise ValueError("Every song needs at least one valid token.")
        # Masked values, including NaNs, must not enter key/value projections.
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        padding_mask = torch.cat((
            torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device),
            padding_mask,
        ), dim=1)
        positions = torch.arange(length + 1, device=x.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        return self.transformer(x, src_key_padding_mask=padding_mask)


class MGI(nn.Module):
    """Classify complete songs, padding tokens only after CNN extraction."""

    def __init__(
        self, mel_bins: int = 64, conv_channels: int = 32,
        n_genres: int = 87,
    ) -> None:
        """Build the current CNN and 4-layer, 8-head ReLU Transformer."""
        super().__init__()
        self.audio_cnn = AudioCNN(mel_bins, conv_channels)
        self.flatten = nn.Flatten(1, 2)
        input_dim = conv_channels * (mel_bins // 8)
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.temporal_encoder = TemporalEncoder(input_dim)
        self.classifier = nn.Linear(input_dim, n_genres)

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Return logits [B,G] for one log-mel [F,T] or a list of log-mels.

        Each input must share the model's device/dtype. Audio is never
        padded or truncated; sequences over position capacity are rejected
        before CNN extraction. Gradients flow through the token padding.
        """
        songs = [x] if isinstance(x, torch.Tensor) else x
        if not isinstance(songs, list) or not songs:
            raise ValueError("Provide a log-mel or a nonempty log-mel list.")
        capacity = self.temporal_encoder.position_embedding.num_embeddings
        for index, song in enumerate(songs):
            if (
                not isinstance(song, torch.Tensor) or song.ndim != 2
                or song.shape[0] != self.audio_cnn.mel_bins
                or song.shape[1] == 0
            ):
                raise ValueError(f"Song {index}: expected a nonempty [F,T].")
            if (song.shape[1] + 31) // 32 + 1 > capacity:
                raise ValueError(f"Song {index}: exceeds position capacity.")

        sequences = []
        for song in songs:
            features = self.audio_cnn(song)
            tokens = self.flatten(features).transpose(1, 2)
            sequences.append(self.linear_1(tokens).squeeze(0))
        padded = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            device=padded.device,
        )
        padding_mask = (
            torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        encoded = self.temporal_encoder(padded, padding_mask)
        return self.classifier(encoded[:, 0, :])


def main() -> None:
    """Load two complete sample songs and print the first song's logits."""
    audio_dir = Path(__file__).resolve().parent.parent / "data" / "audio-low"
    audio_path_1 = audio_dir / "00" / "1100.low.mp3"
    audio_path_2 = audio_dir / "00" / "1196900.low.mp3"

    audio_1, sr_1 = librosa.load(audio_path_1, sr=44100)
    audio_2, sr_2 = librosa.load(audio_path_2, sr=44100)
    log_mel_spectrogram_1 = calculate_log_mel_spectrogram(audio_1, sr_1)
    log_mel_spectrogram_2 = calculate_log_mel_spectrogram(audio_2, sr_2)
    print(f"Log-mel spectrogram for audio 1: {log_mel_spectrogram_1.shape}")
    print(f"Log-mel spectrogram for audio 2: {log_mel_spectrogram_2.shape}")

    model = MGI().eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(log_mel_spectrogram_1))
    print(f"Audio 1 output shape: {output.shape}")


if __name__ == "__main__":
    main()
