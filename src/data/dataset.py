import os
import torch
from torch.utils.data import Dataset
from src.audio.preprocessing import AudioPreprocessor
from src.audio.augmentation import AudioAugmenter

class DeepfakeVoiceDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, chunk_length_s=3.0, overlap_s=1.0, augment=False):
        self.data_dir = data_dir
        self.preprocessor = AudioPreprocessor(sample_rate, chunk_length_s, overlap_s)
        self.augmenter = AudioAugmenter(sample_rate) if augment else None
        
        self.samples = []
        # Expected structure: data_dir / real / *.wav and data_dir / fake / *.wav
        for label_dir, label_idx in [("real", 0), ("fake", 1)]:
            dir_path = os.path.join(data_dir, label_dir)
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.endswith(".wav"):
                        self.samples.append((os.path.join(dir_path, filename), label_idx))
                        
        print(f"Loaded {len(self.samples)} full files from {data_dir}")
        # Note: For large datasets, you might not want to pre-chunk everything in memory.
        # This implementation assumes chunks are created on the fly in __getitem__.
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load and preprocess
        waveform = self.preprocessor.load_and_preprocess(file_path)
        
        # In a real scenario, you might want to return all chunks, 
        # or randomly sample one chunk per file during training.
        # Here we randomly sample one chunk for simplicity during training.
        chunks = self.preprocessor.chunk_audio(waveform)
        
        if not chunks:
            # Fallback for empty audio
            chunk = torch.zeros((1, self.preprocessor.chunk_samples))
        else:
            import random
            chunk, _, _ = random.choice(chunks)
            
        if self.augmenter:
            chunk = self.augmenter(chunk)
            
        return chunk, torch.tensor(label, dtype=torch.float32)
