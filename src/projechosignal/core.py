import numpy as np
from src.projechosignal import frequency_domain_fncs as f
import wave
from numpy.typing import ArrayLike
from typing import NamedTuple
import matplotlib.pyplot as plt


class MelSpectrogram(NamedTuple):
    filename: str

    @property
    def audio(self):
        return _read_wav_file(self.filename)

    def make(
        self, fs: float, audio: ArrayLike, n_fft: int, n_mels: int, n_overlap: int
    ):

        spec = f._stft(audio, n_fft=n_fft, n_overlap=n_overlap)
        spec = f._power_spectrogram(spec)

        return _mel_spectrogram(spec, n_mels, fs)

    def plot(self, spectrogram: ArrayLike):
        for i in range(spectrogram.shape[-1]):
            plt.imshow(
                np.log10(spectrogram + 1e-10),
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
            )
            plt.show()


def _read_wav_file(filename: str):
    """Function to Read *.wav files:

    Args:
        filename (str): Input should be a path or relative path to a .wav audio file

    Returns:
        sampling_rate/sample frequency: Returns the sampling frequency in Hz, usually it's 44.1 kHz
        audio_array: audio file as a np.ndarray of L-N: L is num_frames and N is num_channels
    """
    with wave.open(filename, "rb") as wav_file:

        metadata = wav_file.getparams()
        num_channels, sample_width, frame_rate, num_frames = (
            metadata.nchannels,
            metadata.sampwidth,
            metadata.framerate,
            metadata.nframes,
        )
        # Number of audio channels: 1 for Mono, 2 for stereo Etc.
        # Sample width, based on file properties, 8-bit audio is 1 wide, 16-bit 2 wide etc.
        # Sample rate (Hz)
        # Length of Sample L

        audio_data = wav_file.readframes(num_frames)

        dtype_map = {
            1: np.uint8,
            2: np.int16,
            4: np.int32,
        }  # Hashmap to find data type without boolean checking
        dtype = dtype_map[sample_width]
        audio_array = np.frombuffer(
            audio_data, dtype=dtype
        )  # Converts the binary data type from raw sample to int datatype

        if num_channels > 1:
            audio_array = audio_array.reshape(
                -1, num_channels
            )  # if more than 1 channel then reshapes

        return frame_rate, audio_array


def _mel_spectrogram(spectrogram: np.ndarray, n_mels: int, fs: np.float64):
    """
    Convert a spectrogram to a mel spectrogram using a mel filter bank.
    The function infers the FFT length from the spectrogram's first dimension.

    Parameters:
        spectrogram (np.ndarray): Input spectrogram with shape (freq bins, time bins)
                                  or (freq bins, time bins, M) for multiple channels.
        n_mels (int): Number of mel bands.
        fs (np.float64): Sampling frequency.

    Returns:
        np.ndarray: Mel spectrogram with the same time (and channel) dimensions as the input.
                    Shape is (n_mels, time bins) or (n_mels, time bins, M).
    """
    # Infer n_fft from spectrogram frequency bins.
    # spectrogram frequency bins = n_fft // 2 + 1  =>  n_fft = (freq_bins - 1) * 2
    freq_bins = spectrogram.shape[0]
    n_fft = (freq_bins - 1) * 2

    # Create the mel filter bank using the provided function.
    filter_bank = f.mel_filter_bank(n_mels, n_fft, fs)

    # Convert the spectrogram to a mel spectrogram by applying the filter bank.
    if spectrogram.ndim == 2:
        mel_spec = filter_bank @ spectrogram  # (n_mels, time bins)
    elif spectrogram.ndim == 3:
        mel_spec = np.tensordot(filter_bank, spectrogram, axes=([1], [0]))
        # After tensordot, shape is (n_mels, time bins, M)
    else:
        raise ValueError("Spectrogram must be a 2D or 3D array.")

    return mel_spec
