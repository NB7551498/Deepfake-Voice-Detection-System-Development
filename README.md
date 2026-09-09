# 🎙️ Deepfake Voice Detection System

> A production-grade, end-to-end AI system for detecting synthetic and deepfake speech using audio signal processing and deep learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What It Does

Takes any audio file (WAV, MP3, M4A, FLAC) and outputs:

- ✅ `REAL` or ⚠️ `DEEPFAKE` prediction
- 📊 Fake probability score (0–100%)
- 🎯 Confidence dial
- 🕐 Segment-by-segment timeline analysis (3-second overlapping chunks)
- 📈 Model performance metrics (Accuracy, F1, ROC-AUC, EER)

---

## 🏗️ Architecture

```
User Audio → Preprocessing → VAD → 3s Overlapping Chunks
     → Feature Extraction (228 features: MFCC + Mel + Spectral)
     → Ensemble (CNN + SSL Branch + Spectral Branch)
     → Calibration → Aggregation
     → REAL / DEEPFAKE + Confidence + Segment Timeline
```

**Feature Vector (228 dims per chunk):**
| Feature | Dims |
|---|---|
| MFCCs (40 coeff × mean+std) | 80 |
| Mel Spectrogram (128 bands, mean) | 128 |
| Spectral Centroid, Bandwidth, Rolloff (mean+std) | 6 |
| Zero-Crossing Rate (mean+std) | 2 |
| Chroma (mean) | 12 |

---

## 📁 Project Structure

```
deepfake-voice-detector/
├── configs/                  # YAML configuration files
├── data/
│   ├── raw/real/             # Real human speech recordings
│   ├── raw/fake/             # AI-generated speech recordings
│   └── processed/            # Preprocessed train/val/test splits
├── src/
│   ├── audio/                # Preprocessing, VAD, augmentation
│   ├── models/
│   │   ├── baseline/         # CNN on Mel-Spectrogram
│   │   ├── ssl/              # WavLM / HuBERT fine-tuning
│   │   └── ensemble/         # Fusion of all models
│   ├── training/             # Training loop, scheduler, callbacks
│   ├── evaluation/           # EER, ROC-AUC, F1, confusion matrix
│   ├── inference/            # Predictor + segment aggregator
│   └── api/                  # FastAPI backend
├── frontend/                 # React + Tailwind CSS dashboard
├── scripts/
│   ├── train_local.py        # 🚀 Main training script (no GPU needed)
│   ├── prepare_dataset.py    # Dataset preparation
│   └── evaluate.py           # Evaluation runner
├── models/production/        # Saved model weights + scaler
├── reports/metrics/          # Training results JSON
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/NB7551498/Deepfake-Voice-Detection-System-Development.git
cd Deepfake-Voice-Detection-System-Development
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your audio files into:
```
data/raw/real/   ← real human speech (WAV/MP3)
data/raw/fake/   ← AI-generated/deepfake speech (WAV/MP3)
```

> **Recommended datasets:** ASVspoof 2021, WaveFake, MLAAD

### 3. Train the Model

```bash
python scripts/train_local.py
```

This automatically:
- Extracts 228 acoustic features per 3-second chunk
- Applies augmentation (noise + volume variation)
- Trains a deep MLP classifier with balanced sampling
- Reports Accuracy, F1, ROC-AUC, EER per epoch
- Saves the best model to `models/production/deepfake_cnn.pth`

### 4. Start the API

```bash
python src/api/real_api.py
# Runs on http://127.0.0.1:8001
```

### 5. Docker (full stack)

```bash
docker-compose up
# API: http://localhost:8000
# UI:  http://localhost:3000
```

---

## 📊 Evaluation Metrics

The system reports:
| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **F1 Score** | Harmonic mean of precision + recall |
| **ROC-AUC** | Area under the ROC curve |
| **EER** | Equal Error Rate (key deepfake detection metric) |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status + model info |
| `POST` | `/predict` | Upload audio → get prediction + segments |

**Example response:**
```json
{
  "prediction": "DEEPFAKE",
  "fake_probability": 0.92,
  "real_probability": 0.08,
  "confidence": 0.84,
  "segments_analyzed": 5,
  "segments": [
    { "start": 0.0, "end": 3.0, "fake_probability": 0.21 },
    { "start": 2.0, "end": 5.0, "fake_probability": 0.89 }
  ],
  "model_info": {
    "accuracy": 100.0,
    "f1": 100.0,
    "roc_auc": 100.0,
    "eer": 0.0
  }
}
```

---

## ⚠️ Important Notes

- The model trained on **synthetic data** achieves 100% accuracy on that data. For real-world deepfake detection, retrain with actual datasets like **ASVspoof** or **WaveFake**.
- Model confidence ≠ ground truth. Results should be treated as indicators, not proof.
- Thresholds: **<35%** = Real, **35–65%** = Uncertain, **>65%** = Deepfake.

---

## 📌 Roadmap

- [x] Phase 1 — Audio pipeline (preprocessing, VAD, chunking)
- [x] Phase 2 — Baseline CNN (Mel Spectrogram)
- [x] Phase 3 — SSL model (WavLM/HuBERT wrapper)
- [x] Phase 4 — Augmentation & robustness
- [x] Phase 5 — Training & evaluation (EER, ROC-AUC)
- [x] Phase 6 — FastAPI backend + segment inference
- [x] Phase 7 — React dashboard with timeline visualization
- [ ] Phase 8 — Real dataset integration (ASVspoof / WaveFake)
- [ ] Phase 9 — GPU training + WavLM fine-tuning
- [ ] Phase 10 — Deployment to cloud

---

## 📄 License

MIT License © 2026
