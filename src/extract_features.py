import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Returns: dict with song info or None
    """
    try:
        # build search query
        if artist_name and artist_name != "Unknown":
            query = f"{song_name} {artist_name}"
        else:
            query = song_name

        url = "https://itunes.apple.com/search"
        params = {
            "term": query,
            "entity": "song",
            "limit": 1
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get('resultCount', 0) > 0:
            result = data['results'][0]

            itunes_artist = result.get('artistName', 'Unknown')

            # verify artist match
            if artist_name and artist_name != "Unknown":
                # check for match
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


# redefine class locally
class AudioAutoencoder(nn.Module):
    """
    Autoencoder model for audio spectrogram
    It is redefined here to load the pretrained weights
    """

    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(16384, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )
        # decoder (unused)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 16384),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
checkpoint_path = os.path.join(project_root, "models", "audio_autoencoder.pth")

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AudioAutoencoder().to(device)

if not os.path.exists(checkpoint_path):
    print(f"Model file not found at {checkpoint_path}")
    print("Please train the autoencoder first by running: cd src && python autoencoder.py")
    exit()

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded from {checkpoint_path}")

# extraction loop
print(f"\nExtracting features from {len(df)} songs...")
print("Fetching metadata from iTunes API...\n")

with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing songs"):
        # get the spectrogram
        spec = row['audio_data']

        # convert to tensor : [1, 1, 128, Time]
        input_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # resize to 128 to 128
        input_tensor = F.interpolate(input_tensor, size=(128, 128), mode='bilinear', align_corners=False)

        # move to device
        input_tensor = input_tensor.to(device)

        # normalise to [0, 1]
        input_tensor = (input_tensor + 80) / 80
        input_tensor = torch.clamp(input_tensor, 0.0, 1.0)

        # run encoder to get 256-dimensional latent vector
        latent_vec = model.encoder(input_tensor)

        # convert to numpy and flatten
        vector_numpy = latent_vec.cpu().numpy().flatten().astype(np.float32)

        # extract metadata from filename
        # FIXED: Logic updated to match preprocessing.py format (Artist - Song)
        parts = row['file_name'].split(' - ')
        if len(parts) > 1:
            artist = parts[0].strip()
            song_name = parts[1].strip()
        else:
            song_name = row['file_name']
            artist = None

        # fetch metadata from iTunes API
        itunes_data = search_itunes(song_name, artist)

        if itunes_data:
            # use iTunes data if found
            final_artist = itunes_data['artist']
            album = itunes_data['album']
            duration = itunes_data['duration']
        else:
            # fallback to parsed filename
            final_artist = artist if artist else "Unknown"
            album = None
            duration = None

        # save to db with metadata
        try:
            save_song(song_name, final_artist, vector_numpy, album, duration)
        except Exception as e:
            print(f"\nError saving {song_name}: {e}")
            continue

        # rate limit: iTunes API allows ~20 requests per minute
        time.sleep(0.1)

print("Features extracted and saved to MySQL database.")
print(f"Total songs processed: {len(df)}")