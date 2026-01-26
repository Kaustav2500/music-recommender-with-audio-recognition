import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.preprocessing import df
from database.db_connect import save_song


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


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioAutoencoder().to(device)

checkpoint_path = "../models/audio_autoencoder.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# extraction loop
latent_vectors = []
print("Extracting features from songs...")
with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
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

        # run encoder
        latent_vec = model.encoder(input_tensor)

        # convert to numpy and flatten
        vector_numpy = latent_vec.cpu().numpy().flatten().astype(np.float32)

        # save to db
        save_song(row['file_name'], vector_numpy)


print("Features extracted and saved to MySQL database.")