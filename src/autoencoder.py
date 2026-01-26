import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from audiopipeline import train_loader, test_loader

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()

        # encoder: compresses 128x128 mel spectrogram into a compact latent vector
        self.encoder = nn.Sequential(
            # input: [batch, 1, 128, 128]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # output: [batch, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2), # optimization: leaky relu prevents dead neurons
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # output: [batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # output: [batch, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # output: [batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),                                                                   # output: [batch, 256*8*8 = 16384]
            nn.Linear(16384, 512),                                    # output: [batch, 512]
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)  # final latent representation
        )

        # decoder: reconstructs the 128x128 mel spectrogram from the latent vector
        self.decoder = nn.Sequential(
            # input: [batch, 256]
            nn.Linear(256, 512),  # expand latent space gradually
            nn.LeakyReLU(0.2),
            nn.Linear(512, 16384),  # output: [batch, 16384]
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),  # output: [batch, 256, 8, 8]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 1, 128, 128]
            nn.Sigmoid()                                # normalize output to [0, 1] range
        )

    def forward(self, x):
        # encode input to latent space
        encoded = self.encoder(x)
        # decode latent vector back to spectrogram
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # initialize model and move to device
    model = AudioAutoencoder().to(device)
    print(model)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # training loop
    num_epochs = 50
    train_losses = []
    val_losses = []

    print("\nStarting Training...")
    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        batch_losses = []

        # iterate through batches
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

            # extract spectrogram from tuples or list
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]

            # move data to device
            data = data.to(device)

            # optimization: global normalization
            # map typical dB range (-80 to 0) to (0 to 1)
            data = (data + 80) / 80
            data = torch.clamp(data, 0.0, 1.0) # ensure strict [0, 1] range

            # forward pass: reconstruct the input
            output = model(data)
            loss = criterion(output, data)

            # backward pass: compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()

            # add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            batch_losses.append(loss.item())

        # calculate and store average loss for this epoch
        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_train_loss)

        # validation phase
        model.eval()  # set model to evaluation mode
        val_batch_losses = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # extract spectrogram from tuple or list
                if isinstance(data, list) or isinstance(data, tuple):
                    data = data[0]

                data = data.to(device)

                # normalize input (using same global stats as training)
                data = (data + 80) / 80
                data = torch.clamp(data, 0.0, 1.0)

                output = model(data)
                loss = criterion(output, data)
                val_batch_losses.append(loss.item())

        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        val_losses.append(avg_val_loss)

        # step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # early stopping if validation loss stops improving
        if epoch > 10 and val_losses[-1] > min(val_losses[:-1]):
            patience_counter = sum([val_losses[-i] > min(val_losses[:-i]) for i in range(1, min(6, len(val_losses)))])
            if patience_counter >= 5:
                print("Early stopping triggered - validation loss not improving")
                break

    print("\nTraining Complete!")

    # save the trained model
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses, 'val_losses': val_losses, 'epoch': epoch + 1
    }, "../models/audio_autoencoder.pth")
    print("Model saved to audio_autoencoder.pth")