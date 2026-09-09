import torch
import torchaudio
import random

class AudioAugmenter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def apply_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
        
    def apply_speed_change(self, waveform, speed_factor=1.1):
        # Simplistic speed change using resample (changes pitch too)
        new_sr = int(self.sample_rate * speed_factor)
        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=new_sr)
        waveform_changed = resampler(waveform)
        # Resample back to maintain original sample rate
        resampler_back = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=self.sample_rate)
        return resampler_back(waveform_changed)
        
    def apply_vol_change(self, waveform, vol_factor=0.8):
        return waveform * vol_factor

    def __call__(self, waveform):
        # Randomly apply augmentations
        if random.random() > 0.5:
            waveform = self.apply_noise(waveform, noise_level=random.uniform(0.001, 0.01))
        if random.random() > 0.5:
            speed_factor = random.choice([0.9, 1.1])
            waveform = self.apply_speed_change(waveform, speed_factor)
        if random.random() > 0.5:
            vol_factor = random.uniform(0.5, 1.5)
            waveform = self.apply_vol_change(waveform, vol_factor)
            
        return waveform
