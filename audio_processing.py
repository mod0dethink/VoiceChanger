import numpy as np
import librosa
from scipy import signal

def process_audio_data(data, samplerate, n_steps):
    min_length = 2048
    
    if len(data) < min_length:
        data = np.pad(data, (0, min_length - len(data)), 'constant')

    processed_data = librosa.effects.pitch_shift(data, sr=samplerate, n_steps=n_steps)

    # ローパスフィルタでノイズ除去
    nyquist_rate = samplerate / 2
    low_cutoff = 4000 / nyquist_rate
    low_filter = signal.butter(4, low_cutoff, btype='lowpass', output='sos', analog=False)
    processed_data = signal.sosfilt(low_filter, processed_data)

    nyquist_rate = samplerate / 2
    high_cutoff = 150 / nyquist_rate
    high_filter = signal.butter(4, high_cutoff, btype='highpass', output='sos', analog=False)
    processed_data = signal.sosfilt(high_filter, processed_data)

    processed_data = processed_data * 2

    # eq_coefficientsの長さを調整
    stft = librosa.core.stft(processed_data)
    n_fft = stft.shape[0]
    eq_coefficients = np.array([0.8, 1.2, 0.9, 1.1])
    eq_coefficients = np.tile(eq_coefficients, (n_fft // len(eq_coefficients)) + 1)[:n_fft]

    processed_data = librosa.core.istft(stft * eq_coefficients[:, np.newaxis])

    return processed_data