"""
Real inference using the trained DeepfakeMLP model.
Uses librosa for audio I/O (no torchaudio dependency).
"""

import io
import pickle
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ─── Model Definition (must match training) ─────────────────────────────────
class DeepfakeMLP(nn.Module):
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


# ─── Config ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_S       = 3.0
OVERLAP_S     = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_S)
STEP_SAMPLES  = int(SAMPLE_RATE * (CHUNK_S - OVERLAP_S))
N_MFCC        = 40
N_MELS        = 128
N_FFT         = 1024
HOP_LENGTH    = 512
MODEL_PATH    = "models/production/deepfake_cnn.pth"
SCALER_PATH   = "models/production/scaler.pkl"

# ─── Load Model & Scaler ─────────────────────────────────────────────────────
print("[INIT] Loading trained model...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model = DeepfakeMLP(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    dropout=checkpoint["dropout"]
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open("reports/metrics/results.json") as f:
    training_metrics = json.load(f)

print(f"[INIT] Model loaded. Training accuracy: {training_metrics['accuracy']*100:.2f}%")


# ─── Feature Extraction ──────────────────────────────────────────────────────
def extract_features(audio_chunk: np.ndarray) -> np.ndarray:
    if len(audio_chunk) < CHUNK_SAMPLES:
        audio_chunk = np.pad(audio_chunk, (0, CHUNK_SAMPLES - len(audio_chunk)))
    else:
        audio_chunk = audio_chunk[:CHUNK_SAMPLES]

    features = []
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    mel = librosa.feature.melspectrogram(y=audio_chunk, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))

    cent = librosa.feature.spectral_centroid(y=audio_chunk, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    features.append(np.mean(cent)); features.append(np.std(cent))

    bw = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    features.append(np.mean(bw)); features.append(np.std(bw))

    rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    features.append(np.mean(rolloff)); features.append(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(audio_chunk, hop_length=HOP_LENGTH)
    features.append(np.mean(zcr)); features.append(np.std(zcr))

    chroma = librosa.feature.chroma_stft(y=audio_chunk, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(np.mean(chroma, axis=1))

    return np.array(features, dtype=np.float32)


def analyze_audio(audio_bytes: bytes):
    """Load audio bytes, chunk it, run model on each chunk, aggregate."""
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Build overlapping chunks
    chunks, timestamps = [], []
    if len(audio) < CHUNK_SAMPLES:
        chunks = [audio]
        timestamps = [(0.0, len(audio) / SAMPLE_RATE)]
    else:
        for start in range(0, len(audio) - CHUNK_SAMPLES + 1, STEP_SAMPLES):
            chunks.append(audio[start: start + CHUNK_SAMPLES])
            timestamps.append((start / SAMPLE_RATE, (start + CHUNK_SAMPLES) / SAMPLE_RATE))

    if not chunks:
        chunks = [audio[:CHUNK_SAMPLES]]
        timestamps = [(0.0, CHUNK_S)]

    # Extract features + scale + infer
    feats = np.array([extract_features(c) for c in chunks], dtype=np.float32)
    feats_scaled = scaler.transform(feats)
    tensor = torch.tensor(feats_scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = model(tensor).squeeze(1).numpy()

    segments = []
    for (start_s, end_s), prob in zip(timestamps, probs):
        segments.append({"start": round(start_s, 2), "end": round(end_s, 2),
                         "fake_probability": round(float(prob), 4)})

    # Aggregate: use weighted mean (later segments count more if earlier are uncertain)
    overall = float(np.mean(probs))

    if overall > 0.65:
        prediction = "DEEPFAKE"
    elif overall < 0.35:
        prediction = "REAL"
    else:
        prediction = "UNCERTAIN"

    return {
        "prediction": prediction,
        "fake_probability": round(overall, 4),
        "real_probability": round(1 - overall, 4),
        "confidence": round(abs(overall - 0.5) * 2, 4),
        "segments_analyzed": len(segments),
        "segments": segments,
        "model_info": {
            "accuracy": round(training_metrics["accuracy"] * 100, 2),
            "f1": round(training_metrics["f1"] * 100, 2),
            "roc_auc": round(training_metrics["roc_auc"] * 100, 2),
            "eer": round(training_metrics["eer"] * 100, 2)
        }
    }


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="Deepfake Voice Detector — Real Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "deepfake_cnn_v1",
            "accuracy": f"{training_metrics['accuracy']*100:.2f}%"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, detail="Unsupported audio format.")
    audio_bytes = await file.read()
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, detail="File too large (max 20MB).")
    try:
        return analyze_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    print("[API] Starting Real Model API on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
