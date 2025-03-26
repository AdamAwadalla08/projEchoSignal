import numpy as np
import math
from src import time_domain_fncs as TD


def hz_to_mel(f):
    return 2595 * np.log10(
        1 + f / 700
    )  # Converts Hertz to Mels, from Speech Communication: Human and Machine


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)  # Same source as above


def aafft(signal: np.ndarray):
    """FFT function by Adam A.

    Args:
        signal (np.ndarray): Time Domain Signal To apply fast fourier transform.

    Returns:
        Output FFT: Frequency domain representation of signal.
        Frequency Bins
    """
    N = len(signal)
    return (np.fft.fft(signal), np.fft.fftfreq(N))


def _stft(signal: np.ndarray, n_fft: int, n_overlap: int):
    """
    Short-Time Fourier Transform

    Args:
        signal (np.ndarray): (N, M) audio signal, N is length in frames, and M is number of channels
        n_fft (int): Number of fft bins.
        n_overlap (int): overlapping bins in each fft.

    Returns:
        Linear Scale Spectrogram: shape (n_fft , N/n_fft , M)
    """
    win = TD.HANN(n_fft)  # window function
    # Determine the frames indices along the time dimension
    if signal.ndim == 1:
        frames = range(0, len(signal) - n_fft, n_overlap)
        stft_matrix = np.array(
            [np.fft.rfft(signal[i : i + n_fft] * win) for i in frames]
        )
        return stft_matrix.T  # shape: (freq bins, time bins)

    elif signal.ndim == 2:
        L, M = signal.shape
        frames = range(0, L - n_fft, n_overlap)
        # Compute the STFT for each of the M columns (channels)
        spectrograms = []
        for m in range(M):
            channel = signal[:, m]
            stft_channel = np.array(
                [np.fft.rfft(channel[i : i + n_fft] * win) for i in frames]
            )
            # Transpose so that the frequency bins become the first dimension
            spectrograms.append(stft_channel.T)  # shape: (freq bins, time bins)
        # Stack along a new third axis, resulting in shape (freq bins, time bins, M)
        return np.stack(spectrograms, axis=-1)

    else:
        raise ValueError("Signal must be a 1D or 2D array")


def _power_spectrogram(spectrogram: np.ndarray):
    return np.abs(spectrogram) ** 2  # Disregard function.


def mel_filter_bank(n_mels: int, n_fft: int, fs: np.float64):

    f_min, f_max = 0, fs // 2
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_bands = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_bands = mel_to_hz(mel_bands)

    freq_bins = np.floor(((n_fft + 1) * freq_bands) / fs).astype(int)

    filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
    for mels in range(1, n_mels + 1):
        l, c, r = freq_bins[mels - 1 : mels + 2]
        for k in range(l, c):
            filter_bank[mels - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            filter_bank[mels - 1, k] = (r - k) / (r - c)

    return filter_bank
