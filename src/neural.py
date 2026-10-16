import numpy as np 
import librosa 
import torch 
from torch import nn
from pathlib import Path

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
    if num_snippets > max_snippets:
        num_snippets = max_snippets  # Limit the number of snippets to max_snippets

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

# print("audio sr", sr)
# print("log mel spectrogram shape", calculate_log_mel_spectrogram(snippets(audio, sr)[0], sr).shape)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the architecture of the CNN. Input = [N, C, H, W], where:
        # N = amount of snipets, 
        # C = channels, 
        # H = number of Mel bands, 
        # W = number of time frames
        self.architecture = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # added maxpooling to reduce H and W dimensions.

    def forward(self, x):
        return self.architecture(x)


