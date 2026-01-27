import numpy as np
import torch
from tqdm import tqdm
from preprocessing import df
import os
import sys
import requests
import time

# add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from database.db_connect import save_song

def search_itunes(song_name, artist_name=None):
    """
    Search iTunes API for song metadata
    """
    try:
        if artist_name and artist_name != "Unknown":
            query = f"{song_name} {artist_name}"
        else:
            query = song_name

        url = "https://itunes.apple.com/search"
        params = {"term": query, "entity": "song", "limit": 1}

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get('resultCount', 0) > 0:
            result = data['results'][0]
            itunes_artist = result.get('artistName', 'Unknown')

            # verify artist match
            if artist_name and artist_name != "Unknown":
                if (artist_name.lower() not in itunes_artist.lower() and
                    itunes_artist.lower() not in artist_name.lower()):
                    return None

            return {
                'artist': itunes_artist,
                'album': result.get('collectionName'),
                'duration': result.get('trackTimeMillis', 0) // 1000 if result.get('trackTimeMillis') else None,
                'artwork_url': result.get('artworkUrl100')
            }
    except Exception as e:
        print(f"iTunes API error for {song_name}: {e}")
    return None

def extract_statistical_features(spectrogram):
    """
    Turns a [128, Time] spectrogram into a [256] feature vector
    without using a Neural Network.
    """
    # Ensure it's a tensor
    if not isinstance(spectrogram, torch.Tensor):
        spec_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    else:
        spec_tensor = spectrogram

    # mean: captures the tonal balance (bass vs treble)
    mean_features = torch.mean(spec_tensor, dim=1)

    # std dev: captures the dynamic range (variation in each frequency)
    std_features = torch.std(spec_tensor, dim=1)

    # concatenate to get a 256 dimensional vector
    combined_vector = torch.cat((mean_features, std_features))

    return combined_vector.numpy()

print(f"\nExtracting statistical features from {len(df)} songs...")
print("Fetching metadata from iTunes API...\n")

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing songs"):
    # get spectrogram
    spec = row['audio_data']

    # extract Features
    vector_numpy = extract_statistical_features(spec)

    # extract metadata from filename
    parts = row['file_name'].split(' - ')
    if len(parts) > 1:
        artist = parts[0].strip()
        song_name = parts[1].strip()
    else:
        song_name = row['file_name']
        artist = None

    # fetch metadata
    itunes_data = search_itunes(song_name, artist)

    if itunes_data:
        final_artist = itunes_data['artist']
        album = itunes_data['album']
        duration = itunes_data['duration']
    else:
        final_artist = artist if artist else "Unknown"
        album = None
        duration = None

    # save to db
    try:
        save_song(song_name, final_artist, vector_numpy, album, duration)
    except Exception as e:
        print(f"\nError saving {song_name}: {e}")
        continue

    # rate limit
    time.sleep(0.1)

print("\nFeatures extracted and saved to MySQL database.")