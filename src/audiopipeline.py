import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import df

class AudioPipeline(Dataset):
    def __init__(self, audio_file_labels, audio_file_path, target_sr=44100, duration_sec=5.0):
        self.labels = list(audio_file_labels)
        self.audio_files = list(audio_file_path)
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration_sec)
        self.mel_transform = T.MelSpectrogram(sample_rate=target_sr, n_mels=128, n_fft=2048, hop_length=512)
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.audio_files[idx])

        # convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample if sr doesn't match target
        if sr != self.target_sr:
             resampler = T.Resample(sr, self.target_sr)
             waveform = resampler(waveform)

        # ensure fixed length
        if waveform.shape[1] > self.num_samples:    # if long
            waveform = waveform[:, :self.num_samples]
        else:                                       # if short
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        mel_spec = self.mel_transform(waveform)
        mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)

        return mel_spec, self.labels[idx]


from torch.utils.data import random_split

full_dataset = AudioPipeline(df['file_name'], df['file_path'])

# splitting the dataset
train_size = int(0.90 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)