import os
import torch
import torchaudio

def generate_dummy_audio(directory, num_files, label):
    os.makedirs(os.path.join(directory, label), exist_ok=True)
    for i in range(num_files):
        # Generate 4 seconds of random noise (16kHz)
        waveform = torch.randn(1, 16000 * 4)
        file_path = os.path.join(directory, label, f"dummy_{label}_{i}.wav")
        torchaudio.save(file_path, waveform, 16000)
        print(f"Generated {file_path}")

if __name__ == "__main__":
    print("Generating dummy training data...")
    generate_dummy_audio("data/processed/train", 10, "real")
    generate_dummy_audio("data/processed/train", 10, "fake")
    
    print("Generating dummy validation data...")
    generate_dummy_audio("data/processed/validation", 4, "real")
    generate_dummy_audio("data/processed/validation", 4, "fake")
    print("Dummy data generation complete!")
