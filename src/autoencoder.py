import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.audiopipeline import train_loader, test_loader

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()

        # encoder: compresses 128x128 mel spectrogram into a compact latent vector
        self.encoder = nn.Sequential(
            # input: [batch, 1, 128, 128]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # output: [batch, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # output: [batch, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # output: [batch, 64, 16, 16]
            nn.ReLU(),
            nn.Flatten(),                                          # output: [batch, 64*16*16 = 16384]
            nn.Linear(16384, 128)                                  # output: [batch, 128] - latent space
        )

        # decoder: reconstructs the 128x128 mel spectrogram from the latent vector
        self.decoder = nn.Sequential(
            # input: [batch, 128]
            nn.Linear(128, 16384),                                 # output: [batch, 16384]
            nn.Unflatten(1, (64, 16, 16)),                        # output: [batch, 64, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [batch, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [batch, 16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 1, 128, 128]
            nn.Sigmoid()  # normalize output to [0, 1] range
        )

    def forward(self, x):
        # encode input to latent space
        encoded = self.encoder(x)
        # decode latent vector back to spectrogram
        decoded = self.decoder(encoded)
        return decoded

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# initialize model and move to device
model = AudioAutoencoder().to(device)
print(model)

# define loss function and optimizer
criterion = nn.MSELoss()  # mean squared error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
