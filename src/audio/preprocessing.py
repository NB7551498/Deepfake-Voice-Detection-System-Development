import torch
import librosa
import numpy as np
import io

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, chunk_length_s=3.0, overlap_s=1.0):
        self.sample_rate = sample_rate
        self.chunk_length_s = chunk_length_s
        self.overlap_s = overlap_s
        self.chunk_samples = int(sample_rate * chunk_length_s)
        self.step_samples = int(sample_rate * (chunk_length_s - overlap_s))

    def load_and_preprocess(self, file_path_or_bytes):
        """Loads audio, converts to mono, resamples to target sample rate, and normalizes."""
        if isinstance(file_path_or_bytes, bytes):
            audio_np, sr = librosa.load(io.BytesIO(file_path_or_bytes), sr=self.sample_rate, mono=True)
        else:
            audio_np, sr = librosa.load(file_path_or_bytes, sr=self.sample_rate, mono=True)
        waveform = torch.from_numpy(audio_np).unsqueeze(0).float()

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample
        if sr != self.sample_rate:
            if HAS_TORCHAUDIO:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            else:
                audio_resampled = librosa.resample(waveform.squeeze().numpy(), orig_sr=sr, target_sr=self.sample_rate)
                waveform = torch.from_numpy(audio_resampled).unsqueeze(0).float()
            
        # Peak Normalization
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Trim silence (simple energy-based approach using librosa)
        waveform_np = waveform.squeeze().numpy()
        waveform_trimmed, _ = librosa.effects.trim(waveform_np, top_db=30)
        waveform = torch.tensor(waveform_trimmed).unsqueeze(0)
        
        return waveform
        
    def chunk_audio(self, waveform):
        """Splits the audio into overlapping chunks."""
        num_samples = waveform.shape[1]
        
        chunks = []
        if num_samples < self.chunk_samples:
            # Pad if shorter than chunk size
            pad_amount = self.chunk_samples - num_samples
            chunk = torch.nn.functional.pad(waveform, (0, pad_amount))
            chunks.append((chunk, 0.0, num_samples / self.sample_rate))
        else:
            for start_idx in range(0, num_samples - self.chunk_samples + 1, self.step_samples):
                end_idx = start_idx + self.chunk_samples
                chunk = waveform[:, start_idx:end_idx]
                start_s = start_idx / self.sample_rate
                end_s = end_idx / self.sample_rate
                chunks.append((chunk, start_s, end_s))
                
            # Handle the last chunk if there's a remainder
            if (num_samples - self.chunk_samples) % self.step_samples != 0:
                start_idx = num_samples - self.chunk_samples
                chunk = waveform[:, start_idx:]
                start_s = start_idx / self.sample_rate
                end_s = num_samples / self.sample_rate
                chunks.append((chunk, start_s, end_s))
                
        return chunks
