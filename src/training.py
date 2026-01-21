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