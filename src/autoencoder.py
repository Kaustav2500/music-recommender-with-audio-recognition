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

from tqdm import tqdm
from src.audiopipeline import train_loader, test_loader
from src.autoencoder import model, device, criterion, optimizer

# training loop
num_epochs = 20
train_losses = []

print("\nStarting Training...")
for epoch in range(num_epochs):
    model.train()  # set model to training mode
    batch_losses = []

    # iterate through batches
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

        # extract spectrograms from tuple (spectrograms, labels)
        if isinstance(data, list) or isinstance(data, tuple):
            data = data[0]

        # move data to device
        data = data.to(device)

        # forward pass: reconstruct the input
        output = model(data)
        loss = criterion(output, data)

        # backward pass: compute gradients and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    # calculate and store average loss for this epoch
    avg_loss = sum(batch_losses) / len(batch_losses)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("\nTraining Complete!")

# save the trained model
# torch.save(model.state_dict(), "audio_autoencoder.pth")
# print("Model saved to audio_autoencoder.pth")