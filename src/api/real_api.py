"""
Deepfake Voice Detector 2.0 — Production API.

Features:
  - Two-Stage Hybrid Inference: Fast Scan (<30ms) & Deep Verification (Fusion 2.0 with Frozen SSL)
  - Production Security: File size (<=25MB), duration (<=300s), magic bytes validation
  - Rate Limiting: 40 requests/min per IP
  - API Authentication: Optional X-API-Key with public demo support
  - Performance & Drift Monitoring: Latency p50/p95 tracking (GET /metrics)
  - Benchmark Reports: GET /benchmarks
"""

import io
import json
import time
import librosa
import numpy as np
import sys
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.two_stage import TwoStageDetector
from src.api.middleware.security import validate_audio_payload, validate_api_key
from src.api.middleware.rate_limiter import rate_limiter
from src.api.middleware.monitoring import monitor

# Initialize Two-Stage Detector 2.0
detector = TwoStageDetector(
    model_path="models/production/deepfake_cnn.pth",
    scaler_path="models/production/scaler.pkl",
    rf_path="models/production/rf_model.pkl",
    fusion_path="models/production/fusion_v2.pth",
    fusion_scaler_path="models/production/fusion_scaler_ac.pkl",
)

# Load metrics and benchmark reports
BENCHMARK_PATH = Path("reports/metrics/benchmark_results.json")
ABLATION_PATH = Path("reports/metrics/ablation_study.json")
ROBUSTNESS_PATH = Path("reports/metrics/robustness_study.json")
REGISTRY_PATH = Path("models/registry/registry.json")


def load_json(p: Path, default):
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default


benchmark_data = load_json(BENCHMARK_PATH, {})
ablation_metrics = load_json(ABLATION_PATH, [])
robustness_metrics = load_json(ROBUSTNESS_PATH, [])
model_registry = load_json(REGISTRY_PATH, [])

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Voice Detector 2.0 — Production API",
    description="Research-Grade & Production-Hardened Two-Stage Voice Anti-Spoofing System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "system": "Deepfake Voice Detector 2.0",
        "architecture": "Two-Stage Hybrid (228 Acoustic Features + 768 Frozen SSL)",
        "model_version": "Ensemble-v2.0",
        "active_models": [
            "DeepfakeFusion 2.0 (PyTorch)",
            "RandomForest (100 trees)",
            "DeepfakeMLP (PyTorch)",
            "Calibrated Ensembler",
        ],
        "modes_available": ["fast", "deep"],
        "benchmark_summary": {
            "mean_eer": "4.61%",
            "unseen_generator_generalization": "92.1%",
        },
    }


@app.get("/metrics")
def get_metrics():
    """Production latency p50/p95 and drift monitoring metrics."""
    return monitor.get_metrics()


@app.get("/benchmarks")
def get_benchmarks():
    """Cross-dataset evaluation results across ASVspoof, WaveFake, and MLAAD."""
    return benchmark_data


@app.get("/experiments")
def get_experiments():
    return {
        "ablation_study": ablation_metrics,
        "robustness_study": robustness_metrics,
        "model_registry": model_registry,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_version": "Ensemble-v2.0",
        "architecture": "Deepfake Voice Detector 2.0 (Two-Stage + Frozen SSL)",
        "features": {
            "acoustic_dims": 228,
            "ssl_dims": 768,
            "total_fused_dims": 996,
        },
        "thresholds": {
            "real_bound": "< 0.35",
            "uncertain_range": "0.35 - 0.65",
            "fake_bound": "> 0.65",
        },
        "benchmarks": benchmark_data.get("summary_comparison", {}),
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("fast", description="Detection mode: 'fast' or 'deep'"),
    authorized: bool = Depends(validate_api_key),
):
    # 1. Rate limiting check
    rate_limiter.check(request)

    # 2. Payload size and validation
    audio_bytes = await file.read()
    validate_audio_payload(audio_bytes, file.filename or "audio.wav")

    # 3. Safe decoding
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=30)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio decoding failure: {str(e)}")

    # 4. Inference
    try:
        result = detector.predict_audio(audio, sr=16000, mode=mode)

        # Record metrics
        monitor.record_inference(
            prediction=result.get("prediction", "UNCERTAIN"),
            fake_probability=result.get("fake_probability", 0.5),
            confidence=result.get("confidence", 0.0),
            latency_ms=result.get("latency_ms", 0.0),
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")


if __name__ == "__main__":
    print("[API] Starting Deepfake Voice Detector 2.0 on http://127.0.0.1:8001 ...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
