import numpy as np
from src import time_domain_fncs as td
from src import frequency_domain_fncs as fd
import wave

def read_wav_file(filename:str):
    """Function to Read *.wav files:

    Args:
        filename (str): Input should be a path or relative path to a .wav audio file

    Returns:
        sampling_rate/sample frequency: Returns the sampling frequency in Hz, usually it's 44.1 kHz
        audio_array: audio file as a np.ndarray of L-N: L is num_frames and N is num_channels
    """
    with wave.open(filename, 'rb') as wav_file:

        metadata = wav_file.getparams()
        num_channels,sample_width,frame_rate,num_frames = (metadata.nchannels,
                                                           metadata.sampwidth,
                                                           metadata.framerate,
                                                           metadata.nframes) 
        # Number of audio channels: 1 for Mono, 2 for stereo Etc.
        # Sample width, based on file properties, 8-bit audio is 1 wide, 16-bit 2 wide etc.
        # Sample rate (Hz)
        # Length of Sample L
        
        audio_data = wav_file.readframes(num_frames)
        
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32} # Hashmap to find data type without boolean checking
        dtype = dtype_map[sample_width]
        audio_array = np.frombuffer(audio_data, dtype=dtype) # Converts the binary data type from raw sample to int datatype
        
      
        if num_channels > 1:
            audio_array = audio_array.reshape(-1, num_channels) # if more than 1 channel then reshapes
        
        return frame_rate, audio_array
    

def make_mel_spectrogram():
    pass