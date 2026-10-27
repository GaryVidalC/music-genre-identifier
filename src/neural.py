from pathlib import Path

import librosa
import numpy as np
import torch  # noqa: F401
from torch import nn

root = Path(__file__).parent.parent
ruta_prueba = root / "data" / "jamendo_sample" / "raw_30s_audio-low-00" / "00" / "4400.low.mp3"

sr = 44100
audio, sr = librosa.load(ruta_prueba, sr=sr, mono = True)

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
    log_mel: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray:
    """Split a 2D log-mel into non-overlapping square patches.

    Args:
        log_mel: Spectrogram with shape (mel bands, time frames).
        patch_size: Positive integer giving each patch's height and width.

    Returns:
        Array shaped (patch count, patch_size, patch_size), with the input
        dtype. Time varies fastest; incomplete borders are discarded.

    Raises:
        ValueError: If the input is not 2D, patch_size is invalid, or no
            complete patch fits.
    """
    if log_mel.ndim != 2:
        raise ValueError("log_mel must be a 2D array.")
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, (int, np.integer))
        or patch_size <= 0
    ):
        raise ValueError("patch_size must be a positive integer.")

    n_bands, n_frames = log_mel.shape
    if n_bands < patch_size or n_frames < patch_size:
        raise ValueError("log_mel must contain at least one complete patch.")

    patches = []
    for band in range(0, n_bands - patch_size + 1, patch_size):
        for frame in range(0, n_frames - patch_size + 1, patch_size):
            patches.append(
                log_mel[band:band + patch_size, frame:frame + patch_size]
            )

    return np.stack(patches, axis=0)


snip = snippets(audio, sr)
mel = calculate_log_mel_spectrogram(snip[0], sr)
print(f"log_mel shape: {mel.shape}")

patches = extract_patches(mel, patch_size=16)
print(f"patches shape: {patches.shape}")


patches_reshape = np.reshape(patches, (patches.shape[0], 256))
print(f"patches_reshape shape: {patches_reshape.shape}")

linear = nn.Linear(in_features=256, out_features=256)

print(f"test linear: {linear(torch.from_numpy(patches_reshape).float()).shape}")

class NN_model(nn.Module):
    def __init__(self, d_model = 256, patches = 320):
        super(NN_model, self).__init__()
        self.d_model = d_model

        # Linear layer
        self.linear = nn.Linear(in_features=256, out_features=256)

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
        # linear layer
        self.linear_pool_1 = nn.Linear(in_features=d_model, out_features=d_model//2)  # Linear layer for pooling
        self.linear_pool_2 = nn.Linear(in_features=d_model//2, out_features=1)  # Linear layer for pooling
        
        
    def forward(self, x):
        # embed the patches
        x = np.reshape(x, (x.shape[0], self.d_model))  # Reshape to (batch_size, d_model)
        x = self.linear(torch.from_numpy(x).float())  # Apply linear layer

        # Add positional embeddings
        positions = torch.arange(x.shape[0], device=x.device)# Create position indices
        x = x + self.positional_embedding(positions)

        # add encoder layer
        x = self.transformer_encoder(x)

        # pooling layer
        pl = self.linear_pool_1(x)  # Apply first linear pooling layer
        pl = nn.Tanh()(pl)  # Apply Tanh activation
        pl = self.linear_pool_2(pl)  # Apply second linear pooling layer
        pl = nn.Softmax(dim=0)(pl)  # Apply softmax activation

        # muultiply the pooling layer with the encoder output
        x = torch.mul(x, pl)  # Element-wise multiplication of encoder output and pooling
        x = torch.sum(x, dim=0)  # Sum over the patch dimension
        return x

nn_out = NN_model()(patches)
print(f"test NN_model: {nn_out.shape}")