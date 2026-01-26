import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import df


class AudioPipeline(Dataset):
    """
    Custom Dataset for loading pre-processed audio data and labels
    1. Converts numpy arrays to PyTorch tensors
    2. Resizes tensors to 128x128
    3. Adds a channel dimension
    4. Returns the processed tensor and corresponding label
    """
    def __init__(self, audio_data, labels):
        self.audio_data = list(audio_data)
        self.labels = list(labels)

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        # get the pre-processed numpy array
        spec_array = self.audio_data[idx]

        # convert to tensor
        spec_tensor = torch.tensor(spec_array, dtype=torch.float32)

        # add channel dimension [1, 128, time]
        spec_tensor = spec_tensor.unsqueeze(0)

        # resize to 128x128
        spec_resized = F.interpolate(spec_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)

        # remove the fake batch dimension
        spec_resized = spec_resized.squeeze(0)

        return spec_resized, self.labels[idx]


# create the dataset using the dataframe columns
full_dataset = AudioPipeline(df['audio_data'], df['file_name'])

# splitting the dataset
train_size = int(0.90 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)