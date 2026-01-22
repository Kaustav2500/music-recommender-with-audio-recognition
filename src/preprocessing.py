import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T

# ffmpeg path
os.add_dll_directory(r"C:\ffmpeg\bin")

# cache path for processed dataframe
cache_path = "../data/processed_audio_df.pkl"

# path for the songs folder
folder_path = "../data"

# check if cache exists
if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}...")
    df = pd.read_pickle(cache_path)
else:
    print("Processing audio files...")
    files = os.listdir(folder_path)
    song_data = []

    for file in files:
        if file.endswith((".mp3", ".wav", ".flac")):
            path = os.path.join(folder_path, file)

            # load audio
            waveform, sample_rate = torchaudio.load(path)

            # convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # limit to first 10 seconds
            max_samples = min(waveform.shape[1], 10 * sample_rate)
            waveform = waveform[:, :max_samples]

            # mel spectrogram
            mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, n_mels=128, f_max=8000)
            mel_spec = mel_transform(waveform)

            # convert power spectrogram to dB
            mel_spec_db = T.AmplitudeToDB(stype="power")(mel_spec)

            # remove channel dimension and convert to numpy
            mel_spec_db = mel_spec_db[0].numpy()

            song_data.append({
                "file_name": file,
                "file_path": path,
                "audio_data": mel_spec_db
            })

    # convert list to dataframe
    df = pd.DataFrame(song_data)

    # save to pickle
    df.to_pickle(cache_path)
    print(f"Data saved to {cache_path}")

print(df.head())

