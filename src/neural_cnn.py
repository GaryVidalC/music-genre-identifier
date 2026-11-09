from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

audio_path_1 = Path(__file__).parent.parent/ "data"/ "audio-low" / "00"/"1100.low.mp3"
audio_path_2 = Path(__file__).parent.parent/ "data"/ "audio-low" / "00"/"1196900.low.mp3"

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


audio_1, sr_1 = librosa.load(audio_path_1, sr=44100)
audio_2, sr_2 = librosa.load(audio_path_2, sr=44100)

log_mel_spectrogram_1 = calculate_log_mel_spectrogram(audio_1, sr_1)
log_mel_spectrogram_2 = calculate_log_mel_spectrogram(audio_2, sr_2)
print(f"Log-mel spectrogram for audio 1: {log_mel_spectrogram_1.shape}")
print(f"Log-mel spectrogram for audio 2: {log_mel_spectrogram_2.shape}")

class AudioCNN(nn.Module):
    def __init__(self, mel_bins: int = 64, conv_channels: int =32) -> None:
        """Build three convolution blocks and an eightfold temporal pool."""
        super(AudioCNN, self).__init__()
        self.mel_bins = mel_bins
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
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
        # set dimension to [Batch, convolution_channel, mel_bins, frames]
        _, frames = x.shape
        x = x.view(1, 1, self.mel_bins, frames)  
        x = self.layer_1(x)
        print(f"After layer 1: {x.shape}")
        x = self.layer_2(x)
        print(f"After layer 2: {x.shape}")
        x = self.layer_3(x)
        print(f"After layer 3: {x.shape}")
        x = self.temporal_pool(x)
        print(f"After temporal reduction x8: {x.shape}")
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, num_layers: int = 4):
        super(TemporalEncoder, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim)*0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, x):
        batch_size = x.size(0)
        cls= self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.transformer(x)
        return x

class MGI(nn.Module):
    def __init__(self, mel_bins: int = 64, conv_channels: int =32):
        super(MGI, self).__init__()
        self.audio_cnn = AudioCNN(mel_bins=mel_bins, conv_channels=conv_channels)
        self.flatten = nn.Flatten(1,2)
        self.linear_1 = nn.Linear(in_features=conv_channels * (mel_bins // 8), out_features=conv_channels * (mel_bins // 8))

        self.temporal_encoder = TemporalEncoder(input_dim=conv_channels * (mel_bins // 8), hidden_dim=1024, num_layers=4)

        self.classifier = nn.Linear(in_features=conv_channels * (mel_bins // 8), out_features=87)

    def forward(self, x):
        x = self.audio_cnn(x)
        x = self.flatten(x)
        x = x.transpose(1,2)
        print(f"After flatten: {x.shape}")
        x = self.linear_1(x)
        print(f"After linear 1: {x.shape}")
        x = self.temporal_encoder(x)
        x = x[:, 0, :]
        print(f"After temporal encoder: {x.shape}")
        x = self.classifier(x)
        print(f"After classifier: {x.shape}")
        return x
    
model = MGI()
print(f"Audio 1 output shape: {model(torch.tensor(log_mel_spectrogram_1)).shape}")
