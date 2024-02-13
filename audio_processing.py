import numpy as np
import librosa

def process_audio_data(data, samplerate, n_steps):
    """
    リアルタイムで音声データを処理し、ピッチを変更
    :param data: 音声データのnumpy配列
    :param samplerate: サンプリングレート
    :param n_steps: ピッチを変更するステップ数
    """
    # FFT処理に必要な最小長を定義
    min_length = 2048
    
    # 入力データが最小長に満たない場合、ゼロパディングを行う
    if len(data) < min_length:
        data = np.pad(data, (0, min_length - len(data)), 'constant')

    # Librosaを使用してピッチを変更
    processed_data = librosa.effects.pitch_shift(data, sr=samplerate, n_steps=n_steps)

    return processed_data