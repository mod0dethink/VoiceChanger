import sounddevice as sd
from audio_processing import process_audio_data
import numpy as np

# オーバーラップ部分を保持するためのグローバル変数
overlap_data = np.array([])

def list_audio_devices():
    print("利用可能なインプットデバイス:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # 入力チャンネルがあるデバイスのみをリストアップ
            print(f"{i}: {device['name']}")

def select_input_device():
    list_audio_devices()
    device_index = int(input("マイクを選択してください: "))
    return device_index

def list_audio_output_devices():
    print("利用可能なアウトプットデバイス:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:  # 出力チャンネルがあるデバイスのみをリストアップ
            print(f"{i}: {device['name']}")

def select_output_device():
    list_audio_output_devices()
    device_index = int(input("仮想マイクを選択してください(出力先のマイク): "))
    return device_index

def callback(indata, outdata, frames, time, status, overlap):
    global overlap_data
    if status:
        print(status)
    
    # 入力データにオーバーラップデータを追加
    indata_with_overlap = np.concatenate((overlap_data, indata[:, 0]))
    
    processed_data = process_audio_data(indata_with_overlap, 44100,3)

    # 出力データのサイズを調整
    length_difference = len(outdata) - len(processed_data)
    if length_difference > 0:
        # 処理後のデータが短い場合、ゼロでパディング
        processed_data = np.pad(processed_data, (0, length_difference), 'constant')
    else:
        # 処理後のデータが長い場合、適切な長さにトリミング
        processed_data = processed_data[:len(outdata)]

    # 次フレームのためにオーバーラップデータを更新
    overlap_length = overlap 
    if len(indata[:, 0]) > overlap_length:
        overlap_data = indata[-overlap_length:, 0]
    else:
        overlap_data = indata[:, 0]

    outdata[:] = processed_data.reshape(-1, 1)

def main():
    input_device_index = select_input_device()
    output_device_index = select_output_device()

    print("音声処理の設定を選択してください: 0: デフォルト設定, 1: カスタム設定")
    setting_choice = input("選択: ")
    
    # デフォルト設定
    overlap = 1700
    blocksize = 2300
    
    # カスタム設定の入力
    if setting_choice == "1":
        overlap = int(input("オーバーラップのサイズを入力してください (デフォルト: 1700): "))
        blocksize = int(input("ブロックサイズを入力してください (デフォルト 2300): "))
    
    # ストリームを開始
    with sd.Stream(device=(input_device_index, output_device_index),
                   samplerate=44100, blocksize=blocksize,
                   dtype='float32', channels=1,
                   callback=lambda indata, outdata, frames, time, status: callback(indata, outdata, frames, time, status, overlap)):
        print("音声処理中...(Ctrl+Cで停止)")
        input()  # Ctrl+Cで停止

if __name__ == "__main__":
    main()