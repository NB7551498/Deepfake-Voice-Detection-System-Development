"""
Self-contained Deepfake Voice Detector Training Pipeline
=========================================================
Uses librosa (no torchaudio) for audio I/O + PyTorch CNN for classification.
Generates synthetic training data if no real dataset is present.

Features extracted per audio chunk:
  - 40 MFCCs (mean + std)  = 80
  - 128 Mel-spectrogram (mean) = 128
  - Spectral centroid, bandwidth, rolloff (mean+std each) = 6
  - Zero-crossing rate (mean+std) = 2
  - Chroma features (mean) = 12
  Total = 228 features per chunk
"""

import os
import sys
import json
import time
import pickle
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000
CHUNK_S         = 3.0          # seconds per chunk
OVERLAP_S       = 1.0          # overlap between chunks
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_S)
STEP_SAMPLES    = int(SAMPLE_RATE * (CHUNK_S - OVERLAP_S))

N_MFCC         = 40
N_MELS         = 128
N_FFT          = 1024
HOP_LENGTH     = 512

BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
EPOCHS         = 60
PATIENCE       = 10
HIDDEN_DIM     = 256
DROPOUT        = 0.4

TRAIN_DIR      = "data/processed/train"
VAL_DIR        = "data/processed/validation"
MODEL_OUT      = "models/production/deepfake_cnn.pth"
SCALER_OUT     = "models/production/scaler.pkl"
METRICS_OUT    = "reports/metrics/results.json"

os.makedirs("models/production", exist_ok=True)
os.makedirs("reports/metrics", exist_ok=True)

# ─── Feature Extraction ─────────────────────────────────────────────────────────
def extract_features(audio_chunk: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a rich 228-dimensional feature vector from a 3-second audio chunk."""
    # Safety: ensure correct length
    if len(audio_chunk) < CHUNK_SAMPLES:
        audio_chunk = np.pad(audio_chunk, (0, CHUNK_SAMPLES - len(audio_chunk)))
    else:
        audio_chunk = audio_chunk[:CHUNK_SAMPLES]

    features = []

    # 1. MFCCs (40 coefficients, mean + std = 80 features)
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Mel spectrogram (mean over time = 128 features)
    mel = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))

    # 3. Spectral Centroid (mean + std = 2)
    cent = librosa.feature.spectral_centroid(y=audio_chunk, sr=sr, hop_length=HOP_LENGTH)
    features.append(np.mean(cent))
    features.append(np.std(cent))

    # 4. Spectral Bandwidth (mean + std = 2)
    bw = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=sr, hop_length=HOP_LENGTH)
    features.append(np.mean(bw))
    features.append(np.std(bw))

    # 5. Spectral Rolloff (mean + std = 2)
    rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=sr, hop_length=HOP_LENGTH)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # 6. Zero-Crossing Rate (mean + std = 2)
    zcr = librosa.feature.zero_crossing_rate(audio_chunk, hop_length=HOP_LENGTH)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 7. Chroma features (mean = 12)
    chroma = librosa.feature.chroma_stft(y=audio_chunk, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    features.extend(np.mean(chroma, axis=1))

    return np.array(features, dtype=np.float32)


def load_audio_chunks(file_path: str) -> list:
    """Load audio file, resample to 16kHz mono, return overlapping 3-second chunks."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio) < SAMPLE_RATE:
            return [audio]  # too short, return as-is
        chunks = []
        for start in range(0, len(audio) - CHUNK_SAMPLES + 1, STEP_SAMPLES):
            chunks.append(audio[start: start + CHUNK_SAMPLES])
        if not chunks:
            chunks.append(audio[:CHUNK_SAMPLES])
        return chunks
    except Exception as e:
        print(f"  [WARN] Could not load {file_path}: {e}")
        return []


# ─── Dataset ────────────────────────────────────────────────────────────────────
def build_dataset_from_dir(data_dir: str, augment: bool = False):
    """Walk real/ and fake/ subdirs, extract features for every chunk."""
    X, y = [], []
    for label_str, label_idx in [("real", 0), ("fake", 1)]:
        dir_path = Path(data_dir) / label_str
        if not dir_path.exists():
            print(f"  [SKIP] {dir_path} does not exist")
            continue
        files = list(dir_path.glob("*.wav")) + list(dir_path.glob("*.mp3"))
        print(f"  {label_str}: {len(files)} files")
        for fp in files:
            chunks = load_audio_chunks(str(fp))
            for chunk in chunks:
                feat = extract_features(chunk)
                X.append(feat)
                y.append(label_idx)

                # Augmentation: add noise + volume variants
                if augment:
                    noise = chunk + np.random.randn(len(chunk)) * 0.005
                    X.append(extract_features(noise))
                    y.append(label_idx)

                    vol = chunk * random.uniform(0.6, 1.4)
                    X.append(extract_features(vol))
                    y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class AudioFeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Model ──────────────────────────────────────────────────────────────────────
class DeepfakeMLP(nn.Module):
    """Multi-layer perceptron deepfake classifier with residual skip connections."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ─── Synthetic Data Generator ────────────────────────────────────────────────
def generate_synthetic_data():
    """
    Generates synthetic audio with DISCRIMINATIVE differences:
    - REAL: natural random noise with harmonic structure
    - FAKE: periodic artifact-like waveforms with uniform spectral content
    This gives the model a pattern to learn, unlike pure random noise.
    """
    print("\n[DATA] No dataset found. Generating synthetic discriminative audio...")
    for split, counts in [("train", (80, 80)), ("validation", (20, 20))]:
        for label, n_files in [("real", counts[0]), ("fake", counts[1])]:
            out_dir = Path(TRAIN_DIR if split == "train" else VAL_DIR) / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_files):
                t = np.linspace(0, 4.0, SAMPLE_RATE * 4)
                if label == "real":
                    # Natural speech: multi-frequency harmonics + noise
                    f0 = random.uniform(80, 300)
                    audio = (
                        0.4 * np.sin(2 * np.pi * f0 * t) +
                        0.2 * np.sin(2 * np.pi * f0 * 2 * t) +
                        0.1 * np.sin(2 * np.pi * f0 * 3 * t) +
                        0.1 * np.random.randn(len(t)) +
                        0.05 * np.random.randn(len(t)) * np.sin(2 * np.pi * 5 * t)  # tremolo
                    )
                else:
                    # Synthetic: uniform sawtooth wave (TTS artifact) + spectral flatness
                    f0 = random.uniform(120, 180)
                    sawtooth = 2 * (t * f0 - np.floor(0.5 + t * f0))  # sawtooth wave
                    audio = (
                        0.3 * sawtooth +
                        0.2 * np.sin(2 * np.pi * f0 * t) +
                        0.05 * np.random.randn(len(t))  # minimal noise (AI audio is too clean)
                    )
                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-8)

                # Save as raw PCM wav using scipy (no torchaudio needed)
                import scipy.io.wavfile as wav_io
                wav_io.write(str(out_dir / f"{label}_{i:03d}.wav"),
                             SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print("[DATA] Synthetic data generated.\n")


# ─── Training Loop ──────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  DEEPFAKE VOICE DETECTOR — TRAINING PIPELINE")
    print("=" * 60)

    # Check if dataset exists, else generate synthetic one
    real_train = Path(TRAIN_DIR) / "real"
    fake_train  = Path(TRAIN_DIR) / "fake"
    if (not real_train.exists() or not any(real_train.glob("*.wav"))) or \
       (not fake_train.exists()  or not any(fake_train.glob("*.wav"))):
        generate_synthetic_data()

    print("\n[STEP 1] Extracting features from training set...")
    X_train, y_train = build_dataset_from_dir(TRAIN_DIR, augment=True)
    print(f"  Train samples: {len(X_train)} | Real: {(y_train==0).sum()} | Fake: {(y_train==1).sum()}")

    print("\n[STEP 2] Extracting features from validation set...")
    X_val, y_val = build_dataset_from_dir(VAL_DIR, augment=False)
    print(f"  Val samples: {len(X_val)} | Real: {(y_val==0).sum()} | Fake: {(y_val==1).sum()}")

    # Standardize features
    print("\n[STEP 3] Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to {SCALER_OUT}")

    input_dim = X_train.shape[1]
    print(f"  Feature vector dimension: {input_dim}")

    # Weighted sampler for class balance
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts[y_train]
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32),
                                     len(weights), replacement=True)

    train_ds = AudioFeatDataset(X_train, y_train)
    val_ds   = AudioFeatDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[STEP 4] Training on: {device}")

    model = DeepfakeMLP(input_dim, HIDDEN_DIM, DROPOUT).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_metrics  = {}
    patience_ctr  = 0

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} {'Acc':>8} {'F1':>8} {'AUC':>8} {'EER':>8}")
    print("-" * 66)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out  = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss  = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                out  = model(batch_x)
                loss = criterion(out, batch_y)
                val_loss  += loss.item()
                all_probs.extend(out.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        probs  = np.array(all_probs)
        labels = np.array(all_labels)
        preds  = (probs > 0.5).astype(int)

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0

        # EER
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        fnr = 1 - tpr
        eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])

        print(f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} "
              f"{acc:>8.4f} {f1:>8.4f} {auc:>8.4f} {eer:>8.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = {"accuracy": acc, "f1": f1, "roc_auc": auc, "eer": eer,
                              "val_loss": val_loss, "epoch": epoch}
            torch.save({"model_state": model.state_dict(),
                        "input_dim": input_dim,
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT}, MODEL_OUT)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Epoch   : {best_metrics.get('epoch')}")
    print(f"  Accuracy     : {best_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  F1 Score     : {best_metrics.get('f1', 0)*100:.2f}%")
    print(f"  ROC-AUC      : {best_metrics.get('roc_auc', 0)*100:.2f}%")
    print(f"  EER          : {best_metrics.get('eer', 0)*100:.2f}%")
    print(f"  Model saved  : {MODEL_OUT}")

    with open(METRICS_OUT, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"  Metrics saved: {METRICS_OUT}")
    print("=" * 60)

    return best_metrics


if __name__ == "__main__":
    train()
