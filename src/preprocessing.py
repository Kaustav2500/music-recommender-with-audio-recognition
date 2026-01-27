import os
import re
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import requests
import time

# ffmpeg path
os.add_dll_directory(r"C:\ffmpeg\bin")

# get the directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# cache path for processed dataframe
cache_path = os.path.join(project_root, "data", "processed_audio_df.pkl")
folder_path = os.path.join(project_root, "data")

# settings
MAX_SONGS = 500
DOWNLOAD_LIMIT = 200
INCLUDE_ONLY = ["Weeknd", "Arijit Singh", "KK"]
FORCE_REPROCESS = True


def download_from_itunes(queries, output_folder, limit=200):
    base_url = "https://itunes.apple.com/search"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # use session for better connectivity
    session = requests.Session()

    for query in queries:
        try:
            print(f"Searching for: {query}")
            params = {"term": query, "entity": "song", "limit": limit}

            # search request with timeout
            response = session.get(base_url, params=params, timeout=10)
            data = response.json()

            results = data.get("results", [])
            print(f"Found {len(results)} songs")

            count = 0
            for i, item in enumerate(results):
                preview_url = item.get("previewUrl")
                if not preview_url:
                    continue

                # get metadata
                artist = item.get("artistName", "Unknown")
                track = item.get("trackName", "Unknown")

                # sanitize filename
                safe_name = f"{artist} - {track}"
                safe_name = re.sub(r'[\\/*?:"<>|]', "", safe_name)
                filename = f"{safe_name}.m4a"
                filepath = os.path.join(output_folder, filename)

                # download if not exists
                if not os.path.exists(filepath):
                    try:
                        # download with timeout to prevent hanging
                        doc = session.get(preview_url, timeout=15)
                        with open(filepath, 'wb') as f:
                            f.write(doc.content)

                        print(f"[{count + 1}/{len(results)}] downloaded: {filename}")
                        count += 1

                        # small sleep to be polite
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"Failed to download {filename}: {e}")
                else:
                    count += 1

        except Exception as e:
            print(f"Error searching {query}: {e}")


# start main logic
if (FORCE_REPROCESS or not os.path.exists(cache_path)) and INCLUDE_ONLY:
    print("Starting itunes download...")
    download_from_itunes(INCLUDE_ONLY, folder_path, limit=DOWNLOAD_LIMIT)

# processing phase
if os.path.exists(cache_path) and not FORCE_REPROCESS:
    print(f"Loading cached data from {cache_path}...")
    df = pd.read_pickle(cache_path)
else:
    if FORCE_REPROCESS:
        print("Ignoring cache, Processing files...")

    files = os.listdir(folder_path)

    # filter audio files
    audio_files = [f for f in files if f.endswith((".mp3", ".wav", ".flac", ".m4a"))]

    # filter by name
    if INCLUDE_ONLY:
        audio_files = [f for f in audio_files if any(x.lower() in f.lower() for x in INCLUDE_ONLY)]
        print(f"Total matching files: {len(audio_files)}")

    if not audio_files:
        print("No songs found")
        exit()

    # limit songs
    if len(audio_files) > MAX_SONGS:
        print(f"Limiting to {MAX_SONGS} songs")
        audio_files = audio_files[:MAX_SONGS]

    song_data = []

    print("Converting audio to spectrograms...")
    for file in audio_files:
        path = os.path.join(folder_path, file)

        # clean file name
        name_no_ext = os.path.splitext(file)[0]
        clean_name = re.sub(r'[^a-zA-Z\s,-]', '', name_no_ext)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        clean_name = re.sub(r'^[\s-]+', '', clean_name)

        try:
            # load audio
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            print(f"error loading {file}: {e}")
            continue

        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # target length
        target_length = 10 * sample_rate
        total_samples = waveform.shape[1]

        # center crop logic
        if total_samples > target_length:
            # calculate start point to get the middle
            start = (total_samples - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif total_samples < target_length:
            # pad if too short
            padding = target_length - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # mel spectrogram
        mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, n_mels=128, f_max=8000)
        mel_spec = mel_transform(waveform)

        # convert to db
        mel_spec_db = T.AmplitudeToDB(stype="power")(mel_spec)
        mel_spec_db = mel_spec_db[0].numpy()

        song_data.append({
            "file_name": clean_name,
            "file_path": path,
            "audio_data": mel_spec_db
        })

    # save dataframe
    df = pd.DataFrame(song_data)
    df.to_pickle(cache_path)
    print(f"data saved to {cache_path}")

print(f"processed {len(df)} songs")