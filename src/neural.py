from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

def snippets(
        audio: np.ndarray,
        sr: int,
        snippet_duration: int=15,
        max_snippets: int=40
    ):
    """
    Function to split an audio signal into snippets of a specified duration.

    Parameters:
    audio (np.ndarray): The audio signal to be split.
    sr (int): The sample rate of the audio signal.
    snippet_duration (int): The duration of each snippet in seconds. Default is 15 seconds.

    max_snippets (int): The maximum number of snippets to generate. Default is 40.

    Returns:
    list: A list of audio snippets, each represented as a numpy array.
    """
    snippet_length = snippet_duration * sr  # Calculate the number of samples per snippet
    num_snippets = len(audio) // snippet_length  # Calculate the number of full snippets
    num_snippets = min(num_snippets, max_snippets)  # Limit the number of snippets to max_snippets

    snippets_list = []
    for i in range(num_snippets):
        start_idx = i * snippet_length
        end_idx = start_idx + snippet_length
        snippets_list.append(audio[start_idx:end_idx])  # Append the snippet to the list

    return snippets_list

def calculate_log_mel_spectrogram(
        audio: np.ndarray,
        sr: int,
        n_fft: int=2048,
        hop_length: int=512,
        n_mels: int=64
    ):
    """
    Function to calculate the log-mel spectrogram of an audio signal.

    Parameters:
    audio (np.ndarray): The audio signal for which to compute the log-mel spectrogram.
    sr (int): The sample rate of the audio signal.
    n_fft (int): The number of samples in each FFT window. Default is 2048.
    hop_length (int): The number of samples between successive frames. Default is 512.
    n_mels (int): The number of Mel bands to generate. Default is 64.

    Returns:
    np.ndarray: The log-mel spectrogram of the audio signal.
    """
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram


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


class NN_model(nn.Module):
    def __init__(self,n_genres: int, d_model: int = 256, patches: int = 320) -> None:
        """Build the snippet encoder with embedding width and patch capacity.

        Args:
            d_model: Width of each patch and snippet embedding.
            patches: Maximum number of patch positions in each snippet.
        """
        super(NN_model, self).__init__()
        self.d_model = d_model
        self.n_genres = n_genres

        # Linear layers
        self.linear_first = nn.Linear(in_features=256, out_features=d_model)
        self.linear_last = nn.Linear(in_features=d_model, out_features=n_genres)
        
        # positional embedding layer
        self.positional_embedding = nn.Embedding(patches, d_model)  # Positional embedding layer

        # encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=4,
            dropout=0.1,
            dim_feedforward=1024,
            activation='gelu',
            batch_first=True
        )  # Transformer encoder layer            

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2
        )  # Transformer encoder with 2 layers

        # pooling layer
        self.pooling_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # First linear layer for pooling
            nn.Tanh(),  # Tanh activation
            nn.Linear(d_model // 2, 1),  # Second linear layer for pooling
            nn.Softmax(dim=1)  # Softmax normalization across patches
        )  # Pooling layer to combine patch embeddings into a single snippet embedding

        # snippet attention pooling
        self.snippet_scores = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # First linear layer for snippet scoring
            nn.Tanh(),  # Tanh activation
            nn.Linear(d_model // 2, 1),  # Second linear layer for snippet scoring
        )  # Snippet scoring layer to weight snippets based on their importance
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all snippets of one song without combining snippets.

        Args:
            x: Floating-point patches shaped (snippets, patches, 16, 16),
                on the same device and with the same dtype as the model.

        Returns:
            One embedding per snippet, shaped (snippets, d_model).
        """
        if x.ndim != 4 or x.shape[-2:] != (16, 16):
            raise ValueError("x must have shape (snippets, patches, 16, 16).")
        if (
            x.shape[0] == 0
            or not 0 < x.shape[1] <= self.positional_embedding.num_embeddings
        ):
            raise ValueError("Provide snippets with a valid number of patches.")

        # Flatten each patch, retaining the snippet and patch axes.
        x = self.linear_first(x.flatten(start_dim=-2))

        # Reuse the same patch positions independently for every snippet.
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.positional_embedding(positions)

        # add encoder layer
        x = self.transformer_encoder(x)

        # pooling layer
        pl = self.pooling_layer(x)  # Apply the pooling layer

        # Weight and sum patches without reducing the snippet axis.
        x = torch.mul(x, pl)  # Element-wise multiplication of encoder output and pooling
        x = torch.sum(x, dim=1)  # Sum over the patch dimension

        scores = self.snippet_scores(x)  # Apply the snippet scoring layer
        weights = torch.softmax(scores, dim=0)  # Softmax normalization across snippets
        song_embedding = torch.sum(weights * x, dim=0)  # Weighted sum of snippet embeddings


        # return final linear layer output
        x = self.linear_last(song_embedding)  # Final linear layer to get the song

        return x

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    ruta_prueba = (
        root / "data" / "jamendo_sample" / "raw_30s_audio-low-00"
        / "00" / "4400.low.mp3"
    )
    sr = 44100
    audio, sr = librosa.load(ruta_prueba, sr=sr, mono=True)
    snips = snippets(audio, sr)
    if not snips:
        raise ValueError("Audio must contain at least one complete snippet.")

    log_mels = torch.stack([
        torch.from_numpy(calculate_log_mel_spectrogram(snip, sr)).float()
        for snip in snips
    ])
    patches = extract_patches(log_mels, patch_size=16)
    model = NN_model(n_genres=10).eval()
    with torch.inference_mode():
        nn_out = model(patches)

    print(f"log_mel shape: {log_mels.shape}")
    print(f"patches shape: {patches.shape}")
    print(f"nn out shape: {nn_out.shape}")
