import torch
import torch.nn as nn
import torchaudio

class MelSpectrogramCNN(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128):
        super(MelSpectrogramCNN, self).__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # B x 1 x n_mels x time
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is B x 1 x Samples
        mel_spec = self.mel_transform(x)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        features = self.conv_layers(mel_spec)
        features = features.view(features.size(0), -1)
        
        out = self.fc(features)
        return out
