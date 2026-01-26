import os
import re
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

# ffmpeg path
os.add_dll_directory(r"C:\ffmpeg\bin")

# get the directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# cache path for processed dataframe
cache_path = os.path.join(project_root, "data", "processed_audio_df.pkl")

# path for the songs folder
folder_path = os.path.join(project_root, "data")

# check if cache exists
if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}...")
    df = pd.read_pickle(cache_path)
else:
    print("Processing audio files...")

    # create data folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created data folder at {folder_path}")
        print("Please add your .mp3, .wav, or .flac files to this folder and run again.")
        exit()

    files = os.listdir(folder_path)

    # filter only audio files
    audio_files = [f for f in files if f.endswith((".mp3", ".wav", ".flac"))]

    if len(audio_files) == 0:
        print(f"No audio files found in {folder_path}")
        print("Please add .mp3, .wav, or .flac files to the data folder.")
        exit()

    print(f"Found {len(audio_files)} audio files")
    song_data = []

    for file in audio_files:
        path = os.path.join(folder_path, file)

        # clean the file name
        name_no_ext = os.path.splitext(file)[0]
        clean_name = re.sub(r'[^a-zA-Z\s,-]', '', name_no_ext)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        clean_name = re.sub(r'^[\s-]+', '', clean_name)

        # load audio
        try:
            waveform, sample_rate = torchaudio.load(path)
            print(f"Processing: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        # convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # define target length (10 seconds)
        target_length = 10 * sample_rate

        # pad with zeros if too short, cut if too long
        if waveform.shape[1] < target_length:
            padding_amount = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_amount))
        else:
            waveform = waveform[:, :target_length]

        # mel spectrogram
        mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, n_mels=128, f_max=8000)
        mel_spec = mel_transform(waveform)

        # convert power spectrogram to dB
        mel_spec_db = T.AmplitudeToDB(stype="power")(mel_spec)

        # remove channel dimension and convert to numpy
        mel_spec_db = mel_spec_db[0].numpy()

        song_data.append({
            "file_name": clean_name,
            "file_path": path,
            "audio_data": mel_spec_db
        })

    # convert list to dataframe
    df = pd.DataFrame(song_data)

    # save to pickle
    df.to_pickle(cache_path)
    print(f"Data saved to {cache_path}")

print(f"\nProcessed {len(df)} songs")
print(df.head())

