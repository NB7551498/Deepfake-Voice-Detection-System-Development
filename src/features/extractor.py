"""
Feature Extraction and Caching module for lightweight deepfake voice detection.

Extracts 228 acoustic features per 3-second audio chunk:
  - MFCC: 40 coefficients x 2 (mean, std) = 80
  - Mel Spectrogram: 128 bands (mean)     = 128
  - Spectral Centroid: (mean, std)        = 2
  - Spectral Bandwidth: (mean, std)       = 2
  - Spectral Rolloff: (mean, std)         = 2
  - Zero-Crossing Rate: (mean, std)       = 2
  - Chroma STFT: 12 bins (mean)           = 12
  Total: 228 features
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List, Optional

SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_DURATION = 1.0
STEP_SAMPLES = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP_DURATION))

N_MFCC = 40
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


def extract_chunk_features(chunk: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a 228-dimensional acoustic feature vector from a single chunk."""
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode='constant')
    else:
        chunk = chunk[:CHUNK_SAMPLES]

    feats = []

    # 1. MFCC (mean & std) -> 80 dims
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))

    # 2. Mel Spectrogram (mean log power) -> 128 dims
    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend(np.mean(mel_db, axis=1))

    # 3. Spectral Centroid -> 2 dims
    cent = librosa.feature.spectral_centroid(y=chunk, sr=sr, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(cent)))
    feats.append(float(np.std(cent)))

    # 4. Spectral Bandwidth -> 2 dims
    bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(bw)))
    feats.append(float(np.std(bw)))

    # 5. Spectral Rolloff -> 2 dims
    rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(rolloff)))
    feats.append(float(np.std(rolloff)))

    # 6. Zero-Crossing Rate -> 2 dims
    zcr = librosa.feature.zero_crossing_rate(chunk, hop_length=HOP_LENGTH)
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.std(zcr)))

    # 7. Chroma STFT -> 12 dims
    chroma = librosa.feature.chroma_stft(y=chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.extend(np.mean(chroma, axis=1))

    return np.array(feats, dtype=np.float32)


def chunk_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Slice audio into overlapping 3s chunks with start/end timestamps in seconds."""
    chunks = []
    timestamps = []

    if len(audio) < CHUNK_SAMPLES:
        chunks.append(audio)
        timestamps.append((0.0, len(audio) / sr))
    else:
        for start in range(0, len(audio) - CHUNK_SAMPLES + 1, STEP_SAMPLES):
            end = start + CHUNK_SAMPLES
            chunks.append(audio[start:end])
            timestamps.append((start / sr, end / sr))

    return chunks, timestamps


def cache_dataset_features(
    file_label_pairs: List[Tuple[str, int]],
    cache_file: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all files and cache to .npz for fast reuse.
    Returns (X, y) where X is (N, 228) and y is (N,).
    """
    X_list, y_list = [], []

    for path, label in file_label_pairs:
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio, _ = librosa.effects.trim(audio, top_db=30)
            chunks, _ = chunk_audio(audio)
            for ch in chunks:
                X_list.append(extract_chunk_features(ch))
                y_list.append(label)
        except Exception as e:
            print(f"[WARN] Error extracting {path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y)
    print(f"[CACHE] Saved {len(X)} feature vectors to {cache_file}")

    return X, y


def load_cached_features(cache_file: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load cached features if present, otherwise None."""
    p = Path(cache_file)
    if not p.exists():
        return None
    data = np.load(cache_file)
    return data['X'], data['y']
